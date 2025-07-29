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
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Mapping, Optional, Union
from uuid import uuid1

from ..dataflow import DataFlow, MapData
from ..datapoint.image import Image, MetaAnnotation
from ..mapper.misc import curry
from ..utils.context import timed_operation
from ..utils.identifier import get_uuid_from_str
from ..utils.settings import ObjectTypes
from ..utils.types import DP
from .anngen import DatapointManager


class PipelineComponent(ABC):
    """
    Base class for pipeline components.

    Pipeline components are the parts that make up a pipeline. They contain the
    abstract `serve`, in which the component steps are defined. Within pipelines,
    pipeline components take an image, enrich these with annotations or transform
    existing annotation and transfer the image again. The pipeline component should
    be implemented in such a way that the pythonic approach of passing arguments via
    assignment is used well. To support the pipeline component, an intrinsic
    datapoint manager is provided, which can perform operations on the image
    datapoint that are common for pipeline components. This includes the creation of
    an image, sub-category and similar annotations.

    Pipeline components do not necessarily have to contain predictors but can also
    contain rule-based transformation steps. (For pipeline components with
    predictors see `PredictorPipelineComponent`.)

    The sequential execution of pipeline components is carried out with dataflows.
    In the case of components with predictors, this allows the predictor graph to be
    set up first and then to be streamed to the processed data points.

    Note:
        Currently, predictors can only process single images. Processing higher number of batches is not planned.
    """

    def __init__(self, name: str, model_id: Optional[str] = None) -> None:
        """
        Initializes a `PipelineComponent`.

        Args:
            name: The name of the pipeline component. The name will be used to identify a pipeline component in a
                  pipeline. Use something that describes the task of the pipeline.
            model_id: Optional model identifier.
        """
        self.name = name
        self.service_id = self.get_service_id()
        self.dp_manager = DatapointManager(self.service_id, model_id)
        self.timer_on = False
        self.filter_func: Callable[[DP], bool] = lambda dp: False

    def set_inbound_filter(self, filter_func: Callable[[DP], bool]) -> None:
        """
        Set a filter function to decide if an image of the inbound dataflow should be passed to `self.serve`.

        The filter function should return a boolean value. If the function returns True, the image will not be
        processed by this pipeline component.

        Example:
            ```python
            def do_not_process_tables(dp: Image) -> bool:
                if "table" not in dp.get_categories_from_current_state():
                    return True
                return False

            layout_component = ImageLayoutService(...)
            layout_component.set_inbound_filter(do_not_process_tables)
            ```

        Args:
            filter_func: A function that takes an image datapoint and returns a boolean value.
        """
        self.filter_func = filter_func  # type: ignore

    @abstractmethod
    def serve(self, dp: Image) -> None:
        """
        Processing an image through the whole pipeline component.

        Abstract method that contains all processing steps of the component. Please note that `dp` is already available
        to the `dp_manager` and operations for this can be carried out via it.

        `dp` was transferred to the `dp_manager` via an assignment. This means that operations on `dp` directly or
        operations via `dp_manager` are equivalent.

        As a simplified interface `serve` does not have to return a `dp`. The data point is passed on within pipelines
        internally (via `pass_datapoint`).

        Args:
            dp: The image datapoint to process.
        """
        raise NotImplementedError()

    def _pass_datapoint(self, dp: Image) -> None:
        self.dp_manager.datapoint = dp
        if not self.filter_func(dp):
            self.serve(dp)

    def pass_datapoint(self, dp: Image) -> Image:
        """
        Acceptance, handover to `dp_manager`, transformation and forwarding of `dp`.

        To measure the time, use `self.timer_on = True`.

        Args:
            dp: Datapoint.

        Returns:
            Datapoint.
        """
        if self.timer_on:
            with timed_operation(self.__class__.__name__):
                self._pass_datapoint(dp)
        else:
            self._pass_datapoint(dp)
        return self.dp_manager.datapoint

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via `pass_datapoint` within a dataflow pipeline.

        Args:
            df: An input dataflow.

        Returns:
            An output dataflow.
        """
        return MapData(df, self.pass_datapoint)

    @abstractmethod
    def clone(self) -> PipelineComponent:
        """
        Clone an instance.

        Returns:
            A cloned instance of `PipelineComponent`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_meta_annotation(self) -> MetaAnnotation:
        """
        Get a dict of list of annotation type.

        The dict must contain:
        - `image_annotation` with values: a list of category names,
        - `sub_categories` with values: a dict with category names as keys and a list of the generated sub categories,
        - `relationships` with values: a dict with category names as keys and a list of the generated relationships,
        - `summaries` with values: A list of summary sub categories.

        Returns:
            Dict with meta infos as just described.
        """
        raise NotImplementedError()

    def get_service_id(self) -> str:
        """
        Get the generating service id.

        Returns:
            The service id as a string.
        """
        return get_uuid_from_str(self.name)[:8]

    def clear_predictor(self) -> None:
        """
        Clear the predictor of the pipeline component if it has one.

        Needed for model updates during training.

        Note:
            Maybe you forgot to implement this method in your pipeline component. This might be the case when you run
            evaluation during training and need to update the trained model in your pipeline component.
        """
        raise NotImplementedError(
            "Maybe you forgot to implement this method in your pipeline component. This might "
            "be the case when you run evaluation during training and need to update the "
            "trained model in your pipeline component."
        )

    def has_predictor(self) -> bool:
        """
        Check if the pipeline component has a predictor.

        Returns:
            `True` if the pipeline component has a predictor, otherwise `False`.
        """
        if hasattr(self, "predictor"):
            if self.predictor is not None:
                return True
        return False

    def _undo(self, dp: Image) -> Image:
        """
        Undo the processing of the pipeline component.

        It will remove `ImageAnnotation`, `CategoryAnnotation` and `ContainerAnnotation` with the `service_id` of the
        pipeline component.

        Args:
            dp: The image datapoint.

        Returns:
            The modified image datapoint.
        """
        if self.timer_on:
            with timed_operation(self.__class__.__name__):
                self.dp_manager.datapoint = dp
                dp.remove(service_ids=self.service_id)
        else:
            self.dp_manager.datapoint = dp
            dp.remove(service_ids=self.service_id)
        return self.dp_manager.datapoint

    def undo(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via `_undo` within a dataflow pipeline.

        Args:
            df: An input dataflow of Images.

        Returns:
            An output dataflow of Images.
        """
        return MapData(df, self._undo)


class Pipeline(ABC):
    """
    Abstract base class for creating pipelines.

    Pipelines represent the framework with which documents can be processed by reading individual pages, processing the
     pages through the pipeline infrastructure and returning the extracted information.

    The infrastructure, as the backbone of the pipeline, consists of a list of pipeline components in which images can
     be passed through via dataflows. The order of the pipeline components in the list determines the processing order.
      The components for the pipeline backbone are composed in `_build_pipe`.

    The pipeline is set up via: `analyze` for a directory with single pages or a document with multiple pages. A data
     flow is returned that is triggered via a for loop and starts the actual processing.

    This creates a pipeline using the following command arrangement:

    Example:
        ```python
        layout = LayoutPipeComponent(layout_detector ...)
        text = TextExtractPipeComponent(text_detector ...)
        simple_pipe = MyPipeline(pipeline_component = [layout, text])
        doc_dataflow = simple_pipe.analyze(input = path / to / dir)

        for page in doc_dataflow:
            print(page)
        ```

    In doing so, `page` contains all document structures determined via the pipeline (either directly from the `Image`
     core model or already processed further).

    In addition to `analyze`, the internal `_entry` is used to bundle preprocessing steps.

    It is possible to set a session id for the pipeline. This is useful for logging purposes. The session id can be
     either passed to the pipeline via the `analyze` method or generated automatically.

    To generate a `session_id` automatically:

    Example:
        ```python
        pipe = MyPipeline(pipeline_component = [layout, text])
        pipe.set_session_id = True

        df = pipe.analyze(input = "path/to/dir") # session_id is generated automatically
        ```
    """

    def __init__(self, pipeline_component_list: list[PipelineComponent]) -> None:
        """
        Initializes a `Pipeline`.

        Args:
            pipeline_component_list: A list of pipeline components.
        """
        self.pipe_component_list = pipeline_component_list
        self.set_session_id = False

    @abstractmethod
    def _entry(self, **kwargs: Any) -> DataFlow:
        """
        Use this method to bundle all preprocessing, such as loading one or more documents, so that a dataflow is
         provided as a return value that can be passed on to the pipeline backbone.

        Args:
            kwargs: Arguments, for dynamic customizing of the processing or for the transfer of processing types.

        Returns:
            A dataflow for further processing.
        """
        raise NotImplementedError()

    @staticmethod
    @curry
    def _undo(dp: Image, service_ids: Optional[list[str]] = None) -> Image:
        """
        Remove annotations from a datapoint.

        Args:
            dp: The image datapoint.
            service_ids: Optional list of service ids to remove.

        Returns:
            The modified image datapoint.
        """
        dp.remove(service_ids=service_ids)
        return dp

    def undo(self, df: DataFlow, service_ids: Optional[set[str]] = None) -> DataFlow:
        """
        Mapping a datapoint via `_undo` within a dataflow pipeline.

        Args:
            df: An input dataflow of Images.
            service_ids: A set of service ids to remove.

        Returns:
            An output dataflow of Images.
        """
        return MapData(df, self._undo(service_ids=service_ids))

    @abstractmethod
    def analyze(self, **kwargs: Any) -> DataFlow:
        """
        Try to keep this method as the only one necessary for the user.

        All processing steps, such as preprocessing, setting up the backbone and post-processing are to be bundled. A
        dataflow generator `df` is returned, which is generated via

        Example:
            ```python
            df = pipe.analyze(path="path/to/dir")
            df.reset_state()
            doc = iter(df)
            page = next(doc)
            ```

        can be triggered.

        Args:
            kwargs: Arguments for analysis.

        Returns:
            A dataflow generator.
        """
        raise NotImplementedError()

    def _build_pipe(self, df: DataFlow, session_id: Optional[str] = None) -> DataFlow:
        """
        Composition of the backbone.

        Args:
            df: The input dataflow.
            session_id: Optional session id.

        Returns:
            The processed dataflow.
        """
        if session_id is None and self.set_session_id:
            session_id = self.get_session_id()
        for component in self.pipe_component_list:
            component.timer_on = True
            component.dp_manager.session_id = session_id
            df = component.predict_dataflow(df)
        return df

    def get_meta_annotation(self) -> MetaAnnotation:
        """
        Collects meta annotations from all pipeline components and summarizes the returned results.

        Returns:
            Meta annotations with information about image annotations (list), sub categories (dict with category
            names and generated sub categories), relationships (dict with category names and generated relationships)
            as well as summaries (list with sub categories).
        """
        image_annotations: list[ObjectTypes] = []
        sub_categories: dict[ObjectTypes, dict[ObjectTypes, set[ObjectTypes]]] = {}
        relationships = defaultdict(set[ObjectTypes])  # type: ignore
        summaries: list[ObjectTypes] = []
        for component in self.pipe_component_list:
            meta_anns = component.get_meta_annotation()
            image_annotations.extend(meta_anns.image_annotations)
            for key, value in meta_anns.sub_categories.items():
                sub_dict = meta_anns.sub_categories[key]
                for sub_cat, sub_cat_value in value.items():
                    if sub_cat in sub_dict:
                        sub_dict[sub_cat].update(sub_cat_value)
                    else:
                        sub_dict[sub_cat] = {sub_cat_value}  # type: ignore
                if key in sub_categories:
                    sub_categories[key].update(sub_dict)
                else:
                    sub_categories[key] = sub_dict
            for key, value in meta_anns.relationships.items():  # type: ignore
                relationships[key].update(value)
            summaries.extend(meta_anns.summaries)
        return MetaAnnotation(
            image_annotations=tuple(image_annotations),
            sub_categories=dict(sub_categories),
            relationships=dict(relationships),
            summaries=tuple(summaries),
        )

    def get_service_id_to_meta_annotation(self) -> Mapping[str, MetaAnnotation]:
        """
        Collects meta annotations from all pipeline components and return a dict of service id to its meta annotation.

        Returns:
            `service_id` to `MetaAnnotation` with information about image annotations (list), sub categories (dict with
            category names and generated sub categories), relationships (dict with category names and generated
            relationships) as well as summaries (list with sub categories).
        """
        service_id_to_meta_annotation = {}
        for component in self.pipe_component_list:
            meta_anns = component.get_meta_annotation()
            service_id_to_meta_annotation[component.service_id] = meta_anns
        return service_id_to_meta_annotation

    def get_pipeline_info(
        self, service_id: Optional[str] = None, name: Optional[str] = None
    ) -> Union[str, Mapping[str, str]]:
        """
        Get pipeline information.

        Returns a dictionary with a description of each pipeline component.

        Args:
            service_id: Service id of the pipeline component to search for.
            name: Name of the pipeline component to search for.

        Returns:
            Either a full dictionary with position and name of all pipeline components or the name, if
            the position has been passed or the position if the name has been passed.
        """
        comp_info = {comp.service_id: comp.name for comp in self.pipe_component_list}
        comp_info_name_as_key = {value: key for key, value in comp_info.items()}
        if service_id is not None:
            return comp_info[service_id]
        if name is not None:
            return comp_info_name_as_key[name]
        return comp_info

    def get_pipeline_component(self, service_id: Optional[str] = None, name: Optional[str] = None) -> PipelineComponent:
        """
        Get a pipeline component by `service_id` or `name`.

        Args:
            service_id: Service id of the pipeline component to search for.
            name: Name of the pipeline component to search for.

        Returns:
            The pipeline component if found, otherwise raises ValueError.
        """
        for comp in self.pipe_component_list:
            if comp.service_id == service_id or comp.name == name:
                return comp
        raise ValueError(f"Pipeline component not found with service_id={service_id} or name={name}")

    @staticmethod
    def get_session_id() -> str:
        """
        Get the generating a session id.

        Returns:
            The session id as a string.
        """
        return str(uuid1())[:8]
