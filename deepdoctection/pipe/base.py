# -*- coding: utf-8 -*-
# File: base.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Module for the base class for building pipelines
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Optional, Set, Union

from ..dataflow import DataFlow, MapData
from ..datapoint.image import Image
from ..extern.base import ImageTransformer, ObjectDetector, PdfMiner, TextRecognizer
from ..mapper.laylmstruct import LayoutLMFeatures
from ..utils.context import timed_operation
from ..utils.detection_types import JsonDict
from .anngen import DatapointManager


class PipelineComponent(ABC):
    """
    Base class for pipeline components. Pipeline components are the parts that make up a pipeline. They contain the
    abstract :meth:`serve`, in which the component steps are defined. Within pipelines, pipeline components take an
    image, enrich these with annotations or transform existing annotation and transfer the image again. The pipeline
    component should be implemented in such a way that the pythonic approach of passing arguments via assignment is used
    well. To support the pipeline component, an intrinsic datapoint manager is provided, which can perform operations on
    the image datapoint that are common for pipeline components. This includes the creation of an image, sub-category
    and similar annotations.

    Pipeline components do not necessarily have to contain predictors but can also contain rule-based transformation
    steps. (For pipeline components with predictors see :class:`PredictorPipelineComponent`.)

    The sequential execution of pipeline components is carried out with dataflows. In the case of components with
    predictors, this allows the predictor graph to be set up first and then to be streamed to the processed data points.

    Caution: Currently, predictors can only process single images. Processing higher number of batches is not planned.
    """

    def __init__(self, name: str):
        """
        :param name: The name of the pipeline component. The name will be used to identify a pipeline component in a
                     pipeline. Use something that describe the task of the pipeline.
        """
        self.name = name
        self._meta_has_all_types()
        self.dp_manager = DatapointManager()
        self.timer_on = False

    @abstractmethod
    def serve(self, dp: Image) -> None:
        """
        Processing an image through the whole pipeline component. Abstract method that contains all processing steps of
        the component. Please note that dp is already available to the dp_manager and operations for this can be carried
        out via it.

        dp was transferred to the dp_manager via an assignment. This means that operations on dp directly or operations
        via dp_manager are equivalent.

        As a simplified interface :meth:`serve` does not have to return a dp. The data point is passed on within
        pipelines internally (via :meth:`pass_datapoint`).
        """
        raise NotImplementedError

    def pass_datapoint(self, dp: Image) -> Image:
        """
        Acceptance, handover to dp_manager, transformation and forwarding of dp. To measure the time, use

        .. code-block:: python

            self.timer_on = True

        :param dp: datapoint
        :return: datapoint
        """
        if self.timer_on:
            with timed_operation(self.__class__.__name__):
                self.dp_manager.datapoint = dp
                self.serve(dp)
        else:
            self.dp_manager.datapoint = dp
            self.serve(dp)
        return self.dp_manager.datapoint

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via :meth:`pass_datapoint` within a dataflow pipeline

        :param df: An input dataflow
        :return: A output dataflow
        """
        return MapData(df, self.pass_datapoint)

    @abstractmethod
    def clone(self) -> "PipelineComponent":
        """
        Clone an instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_meta_annotation(self) -> JsonDict:
        """
        Get a dict of list of annotation type. The dict must contain
        "image_annotation" with values: a list of category names,
        "sub_categories" with values: a dict with category names as keys and a list of the generated sub categories
        "relationships" with values: a dict with category names as keys and a list of the generated relationships
        "summaries" with values: A list of summary sub categories
        :return: Dict with meta infos as just described
        """
        raise NotImplementedError

    def _meta_has_all_types(self) -> None:
        if not {"image_annotations", "sub_categories", "relationships", "summaries"}.issubset(
            set(self.get_meta_annotation().keys())
        ):
            raise TypeError(
                f" 'get_meta_annotation' must return dict with all required keys. "
                f"Got {self.get_meta_annotation().keys()}"
            )


class PredictorPipelineComponent(PipelineComponent, ABC):
    """
    Lightweight abstract pipeline component class with :attr:`predictor`. Object detectors that only read in images as
    numpy array and return DetectResults are currently permitted.
    """

    def __init__(
        self,
        name: str,
        predictor: Union[ObjectDetector, PdfMiner, TextRecognizer],
    ) -> None:
        """
        :param name: Will be passed to base class
        :param predictor: An Object detector for predicting
        """
        self.predictor = predictor
        super().__init__(name)

    @abstractmethod
    def clone(self) -> "PredictorPipelineComponent":
        raise NotImplementedError


class LanguageModelPipelineComponent(PipelineComponent, ABC):
    """
    Abstract pipeline component class with two attributes :attr:`tokenizer` and :attr:`language_model` .
    """

    def __init__(
        self,
        name: str,
        tokenizer: Any,
        mapping_to_lm_input_func: Callable[..., Callable[[Image], Optional[LayoutLMFeatures]]],
    ):
        """
        :param name: Will be passed to base class
        :param tokenizer: Tokenizer, typing allows currently anything. This will be changed in the future
        :param mapping_to_lm_input_func: Function mapping image to layout language model features
        """

        self.tokenizer = tokenizer
        self.mapping_to_lm_input_func = mapping_to_lm_input_func
        super().__init__(name)

    @abstractmethod
    def clone(self) -> "LanguageModelPipelineComponent":
        """
        Clone an instance
        """
        raise NotImplementedError


class ImageTransformPipelineComponent(PipelineComponent, ABC):
    """
    Abstract pipeline component class with one model to transform images. This component is meant to be used at the
    beginning of a pipeline
    """

    def __init__(self, name: str, transform_predictor: ImageTransformer):
        """
        :param name: Will be passed to base class
        :param transform_predictor: Am ImageTransformer for image transformation
        """

        self.transform_predictor = transform_predictor
        super().__init__(name)

    @abstractmethod
    def clone(self) -> "ImageTransformPipelineComponent":
        """
        Clone an instance
        """
        raise NotImplementedError


class Pipeline(ABC):
    """
    Abstract base class for creating pipelines. Pipelines represent the framework with which documents can be processed
    by reading individual pages, processing the pages through the pipeline infrastructure and returning the extracted
    information.

    The infrastructure, as the backbone of the pipeline, consists of a list of pipeline components in which images can
    be passed through via dataflows. The order of the pipeline components in the list determines the processing order.
    The components for the pipeline backbone are composed in :meth:`_build_pipe`.

    The pipeline is set up via: meth:`analyze` for a directory with single pages or a document with multiple pages. A
    data flow is returned that is triggered via a for loop and starts the actual processing.

    This creates a pipeline using the following command arrangement:

    **Example:**

        .. code-block:: python

            layout = LayoutPipeComponent(layout_detector ...)
            text = TextExtractPipeComponent(text_detector ...)
            simple_pipe = MyPipeline (pipeline_component = [layout, text])
            doc_dataflow = simple_pipe.analyze(input = path / to / dir)

            for page in doc_dataflow:
                print(page)

    In doing so, page contains all document structures determined via the pipeline (either directly from the Image core
    model or already processed further).

    In addition to :meth:`analyze`, the internal :meth:`_entry` is used to bundle preprocessing steps.
    """

    def __init__(self, pipeline_component_list: List[PipelineComponent]) -> None:
        """
        :param pipeline_component_list: A list of pipeline components.
        """
        self.pipe_component_list = pipeline_component_list

    @abstractmethod
    def _entry(self, **kwargs: Any) -> DataFlow:
        """
        Use this method to bundle all preprocessing, such as loading one or more documents, so that a dataflow is
        provided as a return value that can be passed on to the pipeline backbone.

        :param kwargs: Arguments, for dynamic customizing of the processing or for the transfer of processing types
        """
        raise NotImplementedError

    def _build_pipe(self, df: DataFlow) -> DataFlow:
        """
        Composition of the backbone
        """
        for component in self.pipe_component_list:
            component.timer_on = True
            df = component.predict_dataflow(df)
        return df

    @abstractmethod
    def analyze(self, **kwargs: Any) -> DataFlow:
        """
        Try to keep this method as the only one necessary for the user. All processing steps, such as preprocessing,
        setting up the backbone and post-processing are to be bundled. A dataflow generator df is returned, which is
        generated via

        .. code-block:: python

            doc = iter(df)
            page = next(doc)

        can be triggered.
        """
        raise NotImplementedError

    def get_meta_annotation(self) -> JsonDict:
        """
        Collects meta annotations from all pipeline components and summarizes the returned results

        :return: Meta annotations with information about image annotations (list), sub categories (dict with category
                 names and generated sub categories), relationships (dict with category names and generated
                 relationships) as well as summaries (list with sub categories)
        """
        pipeline_populations: Dict[str, Union[List[str], DefaultDict[str, Set[str]]]] = {
            "image_annotations": [],
            "sub_categories": defaultdict(set),
            "relationships": defaultdict(set),
            "summaries": [],
        }
        for component in self.pipe_component_list:
            meta_anns = deepcopy(component.get_meta_annotation())
            pipeline_populations["image_annotations"].extend(meta_anns["image_annotations"])  # type: ignore
            for key, value in meta_anns["sub_categories"].items():
                pipeline_populations["sub_categories"][key].update(value)
            for key, value in meta_anns["relationships"].items():
                pipeline_populations["relationships"][key].update(value)
            pipeline_populations["summaries"].extend(meta_anns["summaries"])  # type: ignore

        return pipeline_populations

    def get_pipeline_info(
        self, position: Optional[int] = None, name: Optional[str] = None
    ) -> Union[Mapping[int, str], str, int]:
        """Get pipeline information: Returns a dictionary with a description of each pipeline component
        :param position: position of the pipeline component in the pipeline
        :param name: name of the pipeline component to search for
        :return: Either a full dictionary with position and name of all pipeline components or the name, if the position
                 has been passed or the position if the name has been passed.
        """
        comp_info = {key + 1: comp.name for key, comp in enumerate(self.pipe_component_list)}
        comp_info_name_as_key = {value: key for key, value in comp_info.items()}
        if position:
            return comp_info[position]
        if name:
            return comp_info_name_as_key[name]
        return comp_info
