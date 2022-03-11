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
from copy import copy
from typing import Any, Dict, List, Optional, Union

from ..dataflow import DataFlow, MapData  # type: ignore
from ..datapoint.image import Image
from ..extern.base import LMTokenClassifier, ObjectDetector, PdfMiner
from ..mapper import DefaultMapper
from .anngen import DatapointManager


class PipelineComponent(ABC):  # pylint: disable=R0903
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

    Caution: Currently, predictors can only process single images. A higher number of batches is not intended!
    """

    def __init__(self, category_id_mapping: Optional[Dict[int, int]]):
        """
        :param category_id_mapping: Reassignment of category ids. Handover via dict
        """
        self.dp_manager = DatapointManager(category_id_mapping)

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
        Acceptance, handover to dp_manager, transformation and forwarding of dp.

        :param dp: datapoint
        :return: datapoint
        """
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

    def clone(self) -> "PipelineComponent":
        """
        Clone an instance
        """
        raise NotImplementedError


class PredictorPipelineComponent(PipelineComponent, ABC):
    """
    Lightweight abstract pipeline component class with :attr:`predictor`. Object detectors that only read in images as
    numpy array and return DetectResults are currently permitted.
    """

    def __init__(
        self, predictor: Union[ObjectDetector, PdfMiner], category_id_mapping: Optional[Dict[int, int]]
    ) -> None:
        """
        :param predictor: An Object detector for predicting
        """
        super().__init__(category_id_mapping)
        self.predictor = predictor

    def clone(self) -> PipelineComponent:
        predictor = self.predictor.clone()
        assert isinstance(predictor, (ObjectDetector, PdfMiner))
        return self.__class__(predictor, copy(self.dp_manager.category_id_mapping))


class LanguageModelPipelineComponent(PipelineComponent, ABC):
    """
    Abstract pipeline component class with two attributes :attr:`tokenizer` and :attr:`language_model` .
    """

    def __init__(self, tokenizer: Any, language_model: LMTokenClassifier, mapping_to_lm_input_func: DefaultMapper):
        """
        :param tokenizer: Token classifier, typing allows currently anything. This will be changed in the future
        :param language_model: Language model for token classification
        """
        super().__init__(None)
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.mapping_to_lm_input_func = mapping_to_lm_input_func

    def clone(self) -> PipelineComponent:
        return self.__class__(copy(self.tokenizer), copy(self.language_model), copy(self.mapping_to_lm_input_func))


class Pipeline(ABC):  # pylint: disable=R0903
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
    def _entry(self, **kwargs: str) -> DataFlow:
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
            df = component.predict_dataflow(df)
        return df

    @abstractmethod
    def analyze(self, **kwargs: str) -> DataFlow:
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
