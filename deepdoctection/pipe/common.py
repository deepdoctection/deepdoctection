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

from typing import List, Literal, Optional, Sequence, Union

import numpy as np

from ..dataflow import DataFlow, MapData
from ..datapoint.image import Image
from ..datapoint.view import Page
from ..mapper.maputils import MappingContextManager
from ..mapper.match import match_anns_by_intersection
from ..utils.detection_types import JsonDict
from ..utils.settings import LayoutType, ObjectTypes, Relationships, TypeOrStr, get_type
from .base import PipelineComponent
from .registry import pipeline_component_registry


@pipeline_component_registry.register("ImageCroppingService")
class ImageCroppingService(PipelineComponent):
    """
    Crop sub images given by bounding boxes of some annotations. This service is not necessary for
    :class::`ImageLayoutService` and is more intended for saved files where sub images are
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

    Either iou (intersection-over-union) or ioa (intersection-over-area) can be selected as the matching rule.

        .. code-block:: python

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
    A "pseudo" pipeline component that can be added to a pipeline to convert Images into Page formats. It allows a
    custom parsing depending on customizing options of other pipeline components
    """

    def __init__(
        self,
        text_container: TypeOrStr,
        text_block_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    ):
        """
        :param text_container: name of an image annotation that has a CHARS sub category. These annotations will be
                               ordered within all text blocks.
        :param text_block_names: name of image annotation that have a relation with text containers (or which might be
                                 text containers themselves).
        """
        if isinstance(text_block_names, (str, ObjectTypes)):
            text_block_names = [text_block_names]
        elif text_block_names is None:
            text_block_names = []

        self._text_container = get_type(text_container)
        self._text_block_names = [get_type(text_block) for text_block in text_block_names]
        self._init_sanity_checks()

    def pass_datapoint(self, dp: Image) -> Page:
        """
        converts Image to Page
        :param dp: Image
        :return: Page
        """
        return Page.from_image(
            dp,
            self._text_container,
            self._text_block_names,
        )

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Mapping a datapoint via :meth:`pass_datapoint` within a dataflow pipeline

        :param df: An input dataflow
        :return: A output dataflow
        """
        return MapData(df, self.pass_datapoint)

    def _init_sanity_checks(self) -> None:
        assert self._text_container in [
            LayoutType.word,
            LayoutType.line,
        ], f"text_container must be either {LayoutType.word} or {LayoutType.line}"
