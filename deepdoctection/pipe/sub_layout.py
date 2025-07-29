# -*- coding: utf-8 -*-
# File: sub_layout.py

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
Sub layout detection pipeline component
"""
from __future__ import annotations

from collections import Counter
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import crop_box_from_image
from ..datapoint.image import Image, MetaAnnotation
from ..extern.base import DetectionResult, ObjectDetector, PdfMiner
from ..utils.settings import ObjectTypes, Relationships, TypeOrStr, get_type
from ..utils.transform import PadTransform
from ..utils.types import PixelValues
from .base import PipelineComponent
from .registry import pipeline_component_registry


class DetectResultGenerator:
    """
    Use `DetectResultGenerator` to refine raw detection results.

    Certain pipeline components depend on, for example, at least one object being detected. If this is not the
    case, the generator can generate a `DetectResult` with a default setting. If no object was discovered for a
    category, a `DetectResult` with the dimensions of the original image is generated and added to the remaining
    `DetectResults`.
    """

    def __init__(
        self,
        categories_name_as_key: Mapping[ObjectTypes, int],
        group_categories: Optional[list[list[ObjectTypes]]] = None,
        exclude_category_names: Optional[Sequence[ObjectTypes]] = None,
        absolute_coords: bool = True,
    ) -> None:
        """
        Args:
            categories_name_as_key: The dict of all possible detection categories.
            group_categories: If you only want to generate only one `DetectResult` for a group of categories, provided
                that the sum of the group is less than one, then you can pass a list of list for grouping category ids.
            exclude_category_names: List of category names to exclude from result generation.
            absolute_coords: Value to be set in `DetectionResult` for `absolute_coords`.

        """
        self.categories_name_as_key = MappingProxyType(dict(categories_name_as_key.items()))
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        if group_categories is None:
            group_categories = [[cat_name] for cat_name in self.categories_name_as_key]
        self.group_categories = group_categories
        if exclude_category_names is None:
            exclude_category_names = []
        self.exclude_category_names = exclude_category_names
        self.dummy_for_group_generated = [False for _ in self.group_categories]
        self.absolute_coords = absolute_coords

    def create_detection_result(self, detect_result_list: list[DetectionResult]) -> list[DetectionResult]:
        """
        Adds `DetectResults` for which no object was detected to the list.

        Args:
            detect_result_list: `DetectResults` of a previously run `ObjectDetector`.

        Returns:
            Refined list of `DetectionResult`.

        Raises:
            ValueError: If `width` and `height` are not initialized.
        """

        if self.width is None and self.height is None:
            raise ValueError("Initialize height and width first")
        detect_result_list = self._detection_result_sanity_check(detect_result_list)
        count = self._create_condition(detect_result_list)
        for category_name in self.categories_name_as_key:
            if category_name not in self.exclude_category_names:
                if count[category_name] < 1:
                    if not self._dummy_for_group_generated(category_name):
                        detect_result_list.append(
                            DetectionResult(
                                box=[0.0, 0.0, float(self.width), float(self.height)],  # type: ignore
                                class_name=category_name,
                                score=0.0,
                                absolute_coords=self.absolute_coords,
                            )
                        )
        # resetting before finishing this sample
        self.dummy_for_group_generated = self._initialize_dummy_for_group_generated()
        return detect_result_list

    def _create_condition(self, detect_result_list: list[DetectionResult]) -> dict[ObjectTypes, int]:
        count = Counter([ann.class_name for ann in detect_result_list])
        cat_to_group_sum = {}
        for group in self.group_categories:
            group_sum = 0
            for el in group:
                group_sum += count[el]
            for el in group:
                cat_to_group_sum[el] = group_sum
        return cat_to_group_sum

    @staticmethod
    def _detection_result_sanity_check(detect_result_list: list[DetectionResult]) -> list[DetectionResult]:
        """
        Go through each `detect_result` in the list and check if the `box` argument has sensible coordinates:
        `ulx >= 0` and `lrx - ulx >= 0` (same for y coordinate). Remove the detection result if this condition is not
        satisfied. We need this check because if some detection results are not sane, we might end up with some
        non-existing categories.

        Args:
            detect_result_list: List of `DetectionResult` to check.

        Returns:
            List of `DetectionResult` with only valid boxes.
        """
        sane_detect_results = []
        for detect_result in detect_result_list:
            if detect_result.box is not None:
                ulx, uly, lrx, lry = detect_result.box
                if ulx >= 0 and lrx - ulx >= 0 and uly >= 0 and lry - uly >= 0:
                    sane_detect_results.append(detect_result)
        return sane_detect_results

    def _dummy_for_group_generated(self, category_name: ObjectTypes) -> bool:
        for idx, group in enumerate(self.group_categories):
            if category_name in group:
                is_generated = self.dummy_for_group_generated[idx]
                self.dummy_for_group_generated[idx] = True
                return is_generated
        return False

    def _initialize_dummy_for_group_generated(self) -> list[bool]:
        return [False for _ in self.group_categories]


@pipeline_component_registry.register("SubImageLayoutService")
class SubImageLayoutService(PipelineComponent):
    """
    Component in which the selected `ImageAnnotation` can be selected with cropped images and presented to a detector.

    The detected `DetectResults` are transformed into `ImageAnnotations` and stored both in the cache of the parent
    image and in the cache of the sub image.

    If no objects are discovered, artificial objects can be added by means of a refinement process.

    Example:
        ```python
        detect_result_generator = DetectResultGenerator(categories_items)
        d_items = TPFrcnnDetector(item_config_path, item_weights_path, {1: LayoutType.row,
        2: LayoutType.column})
        item_component = SubImageLayoutService(d_items, LayoutType.table, detect_result_generator)
        ```
    """

    def __init__(
        self,
        sub_image_detector: ObjectDetector,
        sub_image_names: Union[str, Sequence[TypeOrStr]],
        service_ids: Optional[Sequence[str]] = None,
        detect_result_generator: Optional[DetectResultGenerator] = None,
        padder: Optional[PadTransform] = None,
    ):
        """
        Args:
            sub_image_detector: `ObjectDetector`.
            sub_image_names: Category names of `ImageAnnotations` to be presented to the detector.
                Attention: The selected `ImageAnnotations` must have `image` and `image.image` not None.
            service_ids: List of service ids to be used for filtering the `ImageAnnotations`. If None, all
                `ImageAnnotations` will be used.
            detect_result_generator: `DetectResultGenerator` instance. `categories` attribute has to be the same as
                the `categories` attribute of the `sub_image_detector`. The generator will be
                responsible to create `DetectionResult` for some categories, if they have not
                been detected by `sub_image_detector`.
            padder: `PadTransform` to pad an image before passing to a predictor. Will be also responsible for
                inverse coordinate transformation.

        Raises:
            ValueError: If the categories of the `detect_result_generator` do not match the categories of the
                        `sub_image_detector`.
        """

        self.sub_image_name = (
            (get_type(sub_image_names),)
            if isinstance(sub_image_names, str)
            else tuple((get_type(cat) for cat in sub_image_names))
        )
        self.service_ids = service_ids
        self.detect_result_generator = detect_result_generator
        self.padder = padder
        self.predictor = sub_image_detector
        super().__init__(self._get_name(sub_image_detector.name), self.predictor.model_id)
        if self.detect_result_generator is not None:
            if self.detect_result_generator.categories_name_as_key != self.predictor.categories.get_categories(
                as_dict=True, name_as_key=True
            ):
                raise ValueError(
                    f"The categories of the 'detect_result_generator' must be the same as the categories of the "
                    f"'sub_image_detector'. Got {self.detect_result_generator.categories_name_as_key} #"
                    f"and {self.predictor.categories.get_categories()}."
                )

    def serve(self, dp: Image) -> None:
        """
        - Selection of `ImageAnnotation` to present to the detector.
        - Invoke the detector.
        - Optionally invoke the `DetectResultGenerator`.
        - Generate `ImageAnnotations` and dump to parent image and sub image.

        Args:
            dp: `Image` to process.
        """
        sub_image_anns = dp.get_annotation(category_names=self.sub_image_name, service_ids=self.service_ids)
        for sub_image_ann in sub_image_anns:
            np_image = self.prepare_np_image(sub_image_ann)
            detect_result_list = self.predictor.predict(np_image)
            if self.padder and detect_result_list:
                boxes = np.array([detect_result.box for detect_result in detect_result_list])
                boxes_orig = self.padder.inverse_apply_coords(boxes)
                for idx, detect_result in enumerate(detect_result_list):
                    detect_result.box = boxes_orig[idx, :].tolist()
            if self.detect_result_generator and sub_image_ann.image:
                self.detect_result_generator.width = sub_image_ann.image.width
                self.detect_result_generator.height = sub_image_ann.image.height
                detect_result_list = self.detect_result_generator.create_detection_result(detect_result_list)

            for detect_result in detect_result_list:
                self.dp_manager.set_image_annotation(detect_result, sub_image_ann.annotation_id)

    def get_meta_annotation(self) -> MetaAnnotation:
        if not isinstance(self.predictor, (ObjectDetector, PdfMiner)):
            raise ValueError(f"predictor must be of type ObjectDetector but is of type {type(self.predictor)}")
        return MetaAnnotation(
            image_annotations=self.predictor.get_category_names(),
            sub_categories={},
            relationships={get_type(parent): {Relationships.CHILD} for parent in self.sub_image_name},
            summaries=(),
        )

    @staticmethod
    def _get_name(predictor_name: str) -> str:
        return f"sub_image_{predictor_name}"

    def clone(self) -> SubImageLayoutService:
        predictor = self.predictor.clone()
        padder_clone = None
        if self.padder:
            padder_clone = self.padder.clone()
        if not isinstance(predictor, ObjectDetector):
            raise ValueError(f"predictor must be of type ObjectDetector but is of type {type(predictor)}")
        return self.__class__(
            predictor,
            self.sub_image_name,
            self.service_ids,
            self.detect_result_generator,
            padder_clone,
        )

    def prepare_np_image(self, sub_image_ann: ImageAnnotation) -> PixelValues:
        """
        Maybe crop and pad a `np_array` before passing it to the predictor.

        Note:
            We currently assume a two level hierarchy of images, e.g. we can crop a sub-image from the base
            image, e.g. the original input but we cannot crop a sub-image from an image which is itself a sub-image.

        Args:
            sub_image_ann: `ImageAnnotation` to be processed.

        Returns:
            Processed `np_image`.

        Raises:
            ValueError: If `sub_image_ann.image` is `None`.
        """
        if sub_image_ann.image is None:
            raise ValueError("sub_image_ann.image is None, but must be an datapoint.Image")
        np_image = sub_image_ann.image.image
        if np_image is None and self.dp_manager.datapoint.image is not None:
            np_image = crop_box_from_image(
                self.dp_manager.datapoint.image,
                sub_image_ann.get_bounding_box(self.dp_manager.datapoint.image_id),
                self.dp_manager.datapoint.width,
                self.dp_manager.datapoint.height,
            )
        if self.padder:
            np_image = self.padder.apply_image(np_image)
        return np_image

    def clear_predictor(self) -> None:
        self.predictor.clear_model()
