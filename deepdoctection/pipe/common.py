# -*- coding: utf-8 -*-
# File: common.py

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
Module for common pipeline components
"""
from copy import deepcopy
from typing import List, Literal, Mapping, Optional, Sequence, Union

import numpy as np

from ..dataflow import DataFlow, MapData
from ..datapoint.image import Image
from ..datapoint.view import Page
from ..mapper.maputils import MappingContextManager
from ..mapper.match import match_anns_by_intersection
from ..mapper.misc import to_image
from ..utils.detection_types import JsonDict
from ..utils.file_utils import detectron2_available, pytorch_available, tf_available
from ..utils.settings import LayoutType, ObjectTypes, Relationships, TypeOrStr, get_type
from .base import PipelineComponent
from .registry import pipeline_component_registry

if tf_available():
    from ..mapper.tpstruct import tf_nms_image_annotations as nms_image_annotations

elif pytorch_available() and detectron2_available():
    from ..mapper.d2struct import pt_nms_image_annotations as nms_image_annotations


@pipeline_component_registry.register("ImageCroppingService")
class ImageCroppingService(PipelineComponent):
    """
    Crop sub images given by bounding boxes of some annotations. This service is not necessary for
    `ImageLayoutService` and is more intended for saved files where sub images are
    generally not stored.
    """

    def __init__(self, category_names: Union[TypeOrStr, Sequence[TypeOrStr]]):
        """
        :param category_names: A single name or a list of category names to crop
        """

        if isinstance(category_names, str):
            category_names = [category_names]
        self.category_names = [get_type(category_name) for category_name in category_names]
        super().__init__("image_crop")

    def serve(self, dp: Image) -> None:
        for ann in dp.get_annotation(category_names=self.category_names):
            dp.image_ann_to_image(ann.annotation_id, crop_image=True)

    def clone(self) -> "PipelineComponent":
        return self.__class__(self.category_names)

    def get_meta_annotation(self) -> JsonDict:
        return dict([("image_annotations", []), ("sub_categories", {}), ("relationships", {}), ("summaries", [])])


@pipeline_component_registry.register("MatchingService")
class MatchingService(PipelineComponent):
    """
    Objects of two object classes can be assigned to one another by determining their pairwise average. If this is above
    a limit, a relation is created between them.
    The parent object class (based on its category) and the child object class are defined for the service. A child
    relation is created in the parent class if the conditions are met.

    Either `iou` (intersection-over-union) or `ioa` (intersection-over-area) can be selected as the matching rule.

            # the following will assign word annotations to text and title annotation, provided that their ioa-threshold
            # is above 0.7. words below that threshold will not be assigned.

            match = MatchingService(parent_categories=["TEXT","TITLE"],child_categories="WORD",matching_rule="ioa",
                                    threshold=0.7)

            # Assigning means that text and title annotation will receive a relationship called "CHILD" which is a list
              of annotation ids of mapped words.
    """

    def __init__(
        self,
        parent_categories: Union[TypeOrStr, List[TypeOrStr]],
        child_categories: Union[TypeOrStr, List[TypeOrStr]],
        matching_rule: Literal["iou", "ioa"],
        threshold: float,
        use_weighted_intersections: bool = False,
        max_parent_only: bool = False,
    ) -> None:
        """
        :param parent_categories: list of categories to be used a for parent class. Will generate a child-relationship
        :param child_categories: list of categories to be used for a child class.
        :param matching_rule: "iou" or "ioa"
        :param threshold: iou/ioa threshold. Value between [0,1]
        :param use_weighted_intersections: This is currently only implemented for matching_rule 'ioa'. Instead of using
                                           the ioa_matrix it will use mat weighted ioa in order to take into account
                                           that intersections with more cells will likely decrease the ioa value. By
                                           multiplying the ioa with the number of all intersection for each child this
                                           value calibrate the ioa.
        :param max_parent_only: Will assign to each child at most one parent with maximum ioa
        """
        self.parent_categories = parent_categories
        self.child_categories = child_categories
        assert matching_rule in ["iou", "ioa"], "segment rule must be either iou or ioa"
        self.matching_rule = matching_rule
        self.threshold = threshold
        self.use_weighted_intersections = use_weighted_intersections
        self.max_parent_only = max_parent_only
        super().__init__("matching")

    def serve(self, dp: Image) -> None:
        """
        - generates pairwise match-score by intersection
        - generates child relationship at parent level

        :param dp: datapoint image
        """
        child_index, parent_index, child_anns, parent_anns = match_anns_by_intersection(
            dp,
            parent_ann_category_names=self.parent_categories,
            child_ann_category_names=self.child_categories,
            matching_rule=self.matching_rule,
            threshold=self.threshold,
            use_weighted_intersections=self.use_weighted_intersections,
            max_parent_only=self.max_parent_only,
        )

        with MappingContextManager(dp_name=dp.file_name):
            matched_child_anns = np.take(child_anns, child_index)  # type: ignore
            matched_parent_anns = np.take(parent_anns, parent_index)  # type: ignore
            for idx, parent in enumerate(matched_parent_anns):
                parent.dump_relationship(Relationships.child, matched_child_anns[idx].annotation_id)

    def clone(self) -> PipelineComponent:
        return self.__class__(self.parent_categories, self.child_categories, self.matching_rule, self.threshold)

    def get_meta_annotation(self) -> JsonDict:

        return dict(
            [
                ("image_annotations", []),
                ("sub_categories", {}),
                ("relationships", {parent: {Relationships.child} for parent in self.parent_categories}),
                ("summaries", []),
            ]
        )


@pipeline_component_registry.register("PageParsingService")
class PageParsingService:
    """
    A "pseudo" pipeline component that can be added to a pipeline to convert `Image`s into `Page` formats. It allows a
    custom parsing depending on customizing options of other pipeline components.
    """

    def __init__(
        self,
        text_container: TypeOrStr,
        top_level_text_block_names: Union[TypeOrStr, Sequence[TypeOrStr]],
        text_block_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    ):
        """
        :param text_container: name of an image annotation that has a CHARS sub category. These annotations will be
                               ordered within all text blocks.
        :param top_level_text_block_names: name of image annotation that have a relation with text containers (or which
                                           might be text containers themselves).
        :param text_block_names: name of image annotation that have a relation with text containers (or which might be
                                 text containers themselves). This is only necessary, when residual text_container (e.g.
                                 words that have not been assigned to any text block) should be displayed in `page.text`
        """
        self.name = "page_parser"
        if isinstance(top_level_text_block_names, (str, ObjectTypes)):
            top_level_text_block_names = [top_level_text_block_names]
        if isinstance(text_block_names, (str, ObjectTypes)):
            text_block_names = [text_block_names]
        if text_block_names is not None:
            text_block_names = [get_type(text_block) for text_block in text_block_names]

        self._text_container = get_type(text_container)
        self._top_level_text_block_names = [get_type(text_block) for text_block in top_level_text_block_names]
        self._text_block_names = text_block_names
        self._init_sanity_checks()

    def pass_datapoint(self, dp: Image) -> Page:
        """
        converts Image to Page
        :param dp: Image
        :return: Page
        """
        return Page.from_image(
            dp, self._text_container, self._top_level_text_block_names, self._text_block_names  # type: ignore
        )

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via `pass_datapoint` within a dataflow pipeline

        :param df: An input dataflow
        :return: A output dataflow
        """
        return MapData(df, self.pass_datapoint)

    def _init_sanity_checks(self) -> None:
        assert self._text_container in [
            LayoutType.word,
            LayoutType.line,
        ], f"text_container must be either {LayoutType.word} or {LayoutType.line}"

    @staticmethod
    def get_meta_annotation() -> JsonDict:
        """
        meta annotation. We do not generate any new annotations here
        """
        return dict([("image_annotations", []), ("sub_categories", {}), ("relationships", {}), ("summaries", [])])

    def clone(self) -> "PageParsingService":
        """clone"""
        return self.__class__(
            deepcopy(self._text_container), deepcopy(self._top_level_text_block_names), deepcopy(self._text_block_names)
        )


@pipeline_component_registry.register("AnnotationNmsService")
class AnnotationNmsService(PipelineComponent):
    """
    A service to pass `ImageAnnotation` to a non-maximum suppression (NMS) process for given pairs of categories.
    `ImageAnnotation`s are subjected to NMS process in groups:
    If `nms_pairs=[[LayoutType.text, LayoutType.table],[LayoutType.title, LayoutType.table]]` all `ImageAnnotation`
    subject to these categories are being selected and identified as one category.
    After NMS the discarded image annotation will be deactivated.
    """

    def __init__(
        self,
        nms_pairs: Sequence[Sequence[TypeOrStr]],
        thresholds: Union[float, List[float]],
        priority: Optional[List[Union[Optional[TypeOrStr]]]] = None,
    ):
        """
        :param nms_pairs: Groups of categories, either as string or by `ObjectType`.
        :param thresholds: Suppression threshold. If only one value is provided, it will apply the threshold to all
                           pairs. If a list is provided, make sure to add as many list elements as `nms_pairs`.
        """
        self.nms_pairs = [[get_type(val) for val in pair] for pair in nms_pairs]
        if isinstance(thresholds, float):
            self.threshold = [thresholds for _ in self.nms_pairs]
        else:
            assert len(self.nms_pairs) == len(thresholds), "Sequences of nms_pairs and thresholds must have same length"
            self.threshold = thresholds
        if priority:
            assert len(self.nms_pairs) == len(priority), "Sequences of nms_pairs and priority must have same length"

            def _get_type(val: Optional[str]) -> Union[ObjectTypes, str]:
                if val is None:
                    return ""
                return get_type(val)

            self.priority = [_get_type(val) for val in priority]
        else:
            self.priority = [None for _ in self.nms_pairs]  # type: ignore
        super().__init__("nms")

    def serve(self, dp: Image) -> None:
        for pair, threshold, prio in zip(self.nms_pairs, self.threshold, self.priority):
            anns = dp.get_annotation(category_names=pair)
            ann_ids_to_keep = nms_image_annotations(anns, threshold, dp.image_id, prio)
            for ann in anns:
                if ann.annotation_id not in ann_ids_to_keep:
                    self.dp_manager.deactivate_annotation(ann.annotation_id)

    def clone(self) -> "PipelineComponent":
        return self.__class__(deepcopy(self.nms_pairs), self.threshold)

    def get_meta_annotation(self) -> JsonDict:
        return dict([("image_annotations", []), ("sub_categories", {}), ("relationships", {}), ("summaries", [])])


@pipeline_component_registry.register("ImageParsingService")
class ImageParsingService:
    """
    A super light service that calls `to_image` when processing datapoints. Might be useful if you build a pipeline that
    is not derived from `DoctectionPipe`.
    """

    def __init__(self, dpi: Optional[int] = None):
        """
        :param dpi: dpi resolution when converting PDFs into pixel values
        """
        self.name = "image"
        self.dpi = dpi

    def pass_datapoint(self, dp: Union[str, Mapping[str, Union[str, bytes]]]) -> Optional[Image]:
        """pass a datapoint"""
        return to_image(dp, self.dpi)

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via `pass_datapoint` within a dataflow pipeline

        :param df: An input dataflow
        :return: A output dataflow
        """
        return MapData(df, self.pass_datapoint)

    def clone(self) -> "ImageParsingService":
        """clone"""
        return self.__class__(self.dpi)

    @staticmethod
    def get_meta_annotation() -> JsonDict:
        """
        meta annotation. We do not generate any new annotations here
        """
        return dict([("image_annotations", []), ("sub_categories", {}), ("relationships", {}), ("summaries", [])])
