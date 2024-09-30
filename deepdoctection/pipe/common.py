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
from __future__ import annotations

import os
from copy import deepcopy
from typing import Literal, Mapping, Optional, Sequence, Union

import numpy as np

from ..dataflow import DataFlow, MapData
from ..datapoint.image import Image
from ..datapoint.view import IMAGE_DEFAULTS, Page
from ..mapper.match import match_anns_by_distance, match_anns_by_intersection
from ..mapper.misc import to_image
from ..utils.settings import LayoutType, ObjectTypes, Relationships, TypeOrStr, get_type
from .base import MetaAnnotation, PipelineComponent
from .registry import pipeline_component_registry

if os.environ.get("DD_USE_TORCH"):
    from ..mapper.d2struct import pt_nms_image_annotations as nms_image_annotations
elif os.environ.get("DD_USE_TF"):
    from ..mapper.tpstruct import tf_nms_image_annotations as nms_image_annotations


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

        self.category_names = (
            (category_names,)
            if isinstance(category_names, str)
            else tuple(get_type(category_name) for category_name in category_names)
        )
        super().__init__("image_crop")

    def serve(self, dp: Image) -> None:
        for ann in dp.get_annotation(category_names=self.category_names):
            dp.image_ann_to_image(ann.annotation_id, crop_image=True)

    def clone(self) -> ImageCroppingService:
        return self.__class__(self.category_names)

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(image_annotations=(), sub_categories={}, relationships={}, summaries=())

    def clear_predictor(self) -> None:
        pass


class IntersectionMatcher:
    """
    Objects of two object classes can be assigned to one another by determining their pairwise intersection. If this is
    above a limit, a relation is created between them.
    The parent object class (based on its category) and the child object class are defined for the service.

    Either `iou` (intersection-over-union) or `ioa` (intersection-over-area) can be selected as the matching rule.

            # the following will assign word annotations to text and title annotation, provided that their ioa-threshold
            # is above 0.7. words below that threshold will not be assigned.

            matcher = IntersectionMatcher(matching_rule="ioa", threshold=0.7)

            match_service = MatchingService(parent_categories=["text","title"],
                                    child_categories="word",
                                    matcher=matcher,
                                    relationship_key=Relationships.CHILD)

            # Assigning means that text and title annotation will receive a relationship called "CHILD" which is a list
              of annotation ids of mapped words.
    """

    def __init__(
        self,
        matching_rule: Literal["iou", "ioa"],
        threshold: float,
        use_weighted_intersections: bool = False,
        max_parent_only: bool = False,
    ) -> None:
        """
        :param matching_rule: "iou" or "ioa"
        :param threshold: iou/ioa threshold. Value between [0,1]
        :param use_weighted_intersections: This is currently only implemented for matching_rule 'ioa'. Instead of using
                                           the ioa_matrix it will use mat weighted ioa in order to take into account
                                           that intersections with more cells will likely decrease the ioa value. By
                                           multiplying the ioa with the number of all intersection for each child this
                                           value calibrate the ioa.
        :param max_parent_only: Will assign to each child at most one parent with maximum ioa"""

        if matching_rule not in ("iou", "ioa"):
            raise ValueError("segment rule must be either iou or ioa")
        self.matching_rule = matching_rule
        self.threshold = threshold
        self.use_weighted_intersections = use_weighted_intersections
        self.max_parent_only = max_parent_only

    def match(
        self,
        dp: Image,
        parent_categories: Union[TypeOrStr, Sequence[TypeOrStr]],
        child_categories: Union[TypeOrStr, Sequence[TypeOrStr]],
    ) -> list[tuple[str, str]]:
        """
        The matching algorithm

        :param dp: datapoint image
        :param parent_categories: list of categories to be used a for parent class. Will generate a child-relationship
        :param child_categories: list of categories to be used for a child class.

        :return: A list of tuples with parent and child annotation ids
        """
        child_index, parent_index, child_anns, parent_anns = match_anns_by_intersection(
            dp,
            parent_ann_category_names=parent_categories,
            child_ann_category_names=child_categories,
            matching_rule=self.matching_rule,
            threshold=self.threshold,
            use_weighted_intersections=self.use_weighted_intersections,
            max_parent_only=self.max_parent_only,
        )

        matched_child_anns = np.take(child_anns, child_index)  # type: ignore
        matched_parent_anns = np.take(parent_anns, parent_index)  # type: ignore

        all_parent_child_relations = []
        for idx, parent in enumerate(matched_parent_anns):
            all_parent_child_relations.append((parent.annotation_id, matched_child_anns[idx].annotation_id))

        return all_parent_child_relations


class NeighbourMatcher:
    """
    Objects of two object classes can be assigned to one another by determining their pairwise distance.

        # the following will assign caption annotations to figure annotation

        matcher = NeighbourMatcher()

        match_service = MatchingService(parent_categories=["figure"],
                                        child_categories="caption",
                                        matcher=matcher,
                                        relationship_key=Relationships.LAYOUT_LINK)

    """

    def match(
        self,
        dp: Image,
        parent_categories: Union[TypeOrStr, Sequence[TypeOrStr]],
        child_categories: Union[TypeOrStr, Sequence[TypeOrStr]],
    ) -> list[tuple[str, str]]:
        """
        The matching algorithm

        :param dp: datapoint image
        :param parent_categories: list of categories to be used a for parent class. Will generate a child-relationship
        :param child_categories: list of categories to be used for a child class.

        :return: A list of tuples with parent and child annotation ids
        """

        return [
            (pair[0].annotation_id, pair[1].annotation_id)
            for pair in match_anns_by_distance(dp, parent_categories, child_categories)
        ]


@pipeline_component_registry.register("MatchingService")
class MatchingService(PipelineComponent):
    """
    A service to match annotations of two categories by intersection or distance. The matched annotations will be
    assigned a relationship. The parent category will receive a relationship to the child category.
    """

    def __init__(
        self,
        parent_categories: Union[TypeOrStr, Sequence[TypeOrStr]],
        child_categories: Union[TypeOrStr, Sequence[TypeOrStr]],
        matcher: Union[IntersectionMatcher, NeighbourMatcher],
        relationship_key: Relationships,
    ) -> None:
        """
        :param parent_categories: list of categories to be used a for parent class. Will generate a child-relationship
        :param child_categories: list of categories to be used for a child class.

        """
        self.parent_categories = (
            (get_type(parent_categories),)
            if isinstance(parent_categories, str)
            else tuple(get_type(category_name) for category_name in parent_categories)
        )
        self.child_categories = (
            (get_type(child_categories),)
            if isinstance(child_categories, str)
            else (tuple(get_type(category_name) for category_name in child_categories))
        )
        self.matcher = matcher
        self.relationship_key = relationship_key
        super().__init__("matching")

    def serve(self, dp: Image) -> None:
        """
        - generates pairwise match-score by intersection
        - generates child relationship at parent level

        :param dp: datapoint image
        """

        matched_pairs = self.matcher.match(dp, self.parent_categories, self.child_categories)

        for pair in matched_pairs:
            self.dp_manager.set_relationship_annotation(self.relationship_key, pair[0], pair[1])

    def clone(self) -> PipelineComponent:
        return self.__class__(self.parent_categories, self.child_categories, self.matcher, self.relationship_key)

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(
            image_annotations=(),
            sub_categories={},
            relationships={parent: {Relationships.CHILD} for parent in self.parent_categories},
            summaries=(),
        )

    def clear_predictor(self) -> None:
        pass


@pipeline_component_registry.register("PageParsingService")
class PageParsingService(PipelineComponent):
    """
    A "pseudo" pipeline component that can be added to a pipeline to convert `Image`s into `Page` formats. It allows a
    custom parsing depending on customizing options of other pipeline components.
    """

    def __init__(
        self,
        text_container: TypeOrStr,
        floating_text_block_categories: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
        include_residual_text_container: bool = True,
    ):
        """
        :param text_container: name of an image annotation that has a CHARS sub category. These annotations will be
                               ordered within all text blocks.
        :param floating_text_block_categories: name of image annotation that have a relation with text containers.
        """
        self.name = "page_parser"
        if isinstance(floating_text_block_categories, (str, ObjectTypes)):
            floating_text_block_categories = (get_type(floating_text_block_categories),)
        if floating_text_block_categories is None:
            floating_text_block_categories = IMAGE_DEFAULTS["floating_text_block_categories"]

        self.text_container = get_type(text_container)
        self.floating_text_block_categories = tuple(
            (get_type(text_block) for text_block in floating_text_block_categories)
        )
        self.include_residual_text_container = include_residual_text_container
        self._init_sanity_checks()
        super().__init__(self.name)

    def serve(self, dp: Image) -> None:
        raise NotImplementedError("PageParsingService is not meant to be used in serve method")

    def pass_datapoint(self, dp: Image) -> Page:
        """
        converts Image to Page
        :param dp: Image
        :return: Page
        """
        return Page.from_image(
            dp,
            text_container=self.text_container,
            floating_text_block_categories=self.floating_text_block_categories,
            include_residual_text_container=self.include_residual_text_container,
        )

    def _init_sanity_checks(self) -> None:
        assert self.text_container in (
            LayoutType.WORD,
            LayoutType.LINE,
        ), f"text_container must be either {LayoutType.WORD} or {LayoutType.LINE}"

    def get_meta_annotation(self) -> MetaAnnotation:
        """
        meta annotation. We do not generate any new annotations here
        """
        return MetaAnnotation(image_annotations=(), sub_categories={}, relationships={}, summaries=())

    def clone(self) -> PageParsingService:
        """clone"""
        return self.__class__(
            deepcopy(self.text_container),
            deepcopy(self.floating_text_block_categories),
            self.include_residual_text_container,
        )

    def clear_predictor(self) -> None:
        pass


@pipeline_component_registry.register("AnnotationNmsService")
class AnnotationNmsService(PipelineComponent):
    """
    A service to pass `ImageAnnotation` to a non-maximum suppression (NMS) process for given pairs of categories.
    `ImageAnnotation`s are subjected to NMS process in groups:
    If `nms_pairs=[[LayoutType.text, LayoutType.table],[LayoutType.title, LayoutType.table]]` all `ImageAnnotation`
    subject to these categories are being selected and identified as one category.
    After NMS the discarded image annotation will be deactivated.

    **Example**

        AnnotationNmsService(nms_pairs=[[LayoutType.text, LayoutType.table],[LayoutType.title, LayoutType.table]],
                             thresholds=[0.7,0.7])   # for each pair a threshold has to be provided

    For a pair of categories, one can also select a category which has always priority even if the score is lower.
    This is useful if one expects some categories to be larger and want to keep them.


    """

    def __init__(
        self,
        nms_pairs: Sequence[Sequence[TypeOrStr]],
        thresholds: Union[float, list[float]],
        priority: Optional[list[Union[Optional[TypeOrStr]]]] = None,
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

    def clone(self) -> PipelineComponent:
        return self.__class__(deepcopy(self.nms_pairs), self.threshold)

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(image_annotations=(), sub_categories={}, relationships={}, summaries=())

    def clear_predictor(self) -> None:
        pass


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

    def clone(self) -> ImageParsingService:
        """clone"""
        return self.__class__(self.dpi)

    @staticmethod
    def get_meta_annotation() -> MetaAnnotation:
        """
        meta annotation. We do not generate any new annotations here
        """
        return MetaAnnotation(image_annotations=(), sub_categories={}, relationships={}, summaries=())

    def clear_predictor(self) -> None:
        """clear predictor. Will do nothing"""
