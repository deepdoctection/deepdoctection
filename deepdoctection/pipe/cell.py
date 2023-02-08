# -*- coding: utf-8 -*-
# File: cell.py

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
Module for cell detection pipeline component
"""
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..datapoint.image import Image
from ..extern.base import DetectionResult, ObjectDetector, PdfMiner
from ..utils.detection_types import JsonDict
from ..utils.settings import ObjectTypes, Relationships
from ..utils.transform import PadTransform
from .base import PredictorPipelineComponent
from .registry import pipeline_component_registry


class DetectResultGenerator:
    """
    Use:  `DetectResultGenerator` to refine raw detection results.

    Certain pipeline components depend on, for example, at least one object being detected. If this is not the
    case, the generator can generate a DetectResult with a default setting. If no object was discovered for a
    category, a DetectResult with the dimensions of the original image is generated and added to the remaining
    DetectResults.
    """

    def __init__(
        self,
        categories: Mapping[str, ObjectTypes],
        group_categories: Optional[List[List[str]]] = None,
        exclude_category_ids: Optional[Sequence[str]] = None,
        absolute_coords: bool = True,
    ) -> None:
        """
        :param categories: The dict of all possible detection categories
        :param group_categories: If you only want to generate only one DetectResult for a group of categories, provided
                                 that the sum of the group is less than one, then you can pass a list of list for
                                 grouping category ids.
        :param absolute_coords: 'absolute_coords' value to be set in 'DetectionResult'
        """
        self.categories = categories
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        if group_categories is None:
            group_categories = [[idx] for idx in self.categories]
        self.group_categories = group_categories
        if exclude_category_ids is None:
            exclude_category_ids = []
        self.exclude_category_ids = exclude_category_ids
        self.dummy_for_group_generated = [False for _ in self.group_categories]
        self.absolute_coords = absolute_coords

    def create_detection_result(self, detect_result_list: List[DetectionResult]) -> List[DetectionResult]:
        """
        Adds DetectResults for which no object was detected to the list.

        :param detect_result_list: DetectResults of a previously run ObjectDetector
        :return: refined list
        """

        if self.width is None and self.height is None:
            raise ValueError("Initialize height and width first")

        count = self._create_condition(detect_result_list)
        for category_id in self.categories:
            if category_id not in self.exclude_category_ids:
                if count[category_id] < 1:
                    if not self._dummy_for_group_generated(category_id):
                        detect_result_list.append(
                            DetectionResult(
                                box=[0.0, 0.0, float(self.width), float(self.height)],  # type: ignore
                                class_id=int(category_id),
                                class_name=self.categories[category_id],
                                score=0.0,
                                absolute_coords=self.absolute_coords,
                            )
                        )
        return detect_result_list

    def _create_condition(self, detect_result_list: List[DetectionResult]) -> Dict[str, int]:
        count = Counter([str(ann.class_id) for ann in detect_result_list])
        cat_to_group_sum = {}
        for group in self.group_categories:
            group_sum = 0
            for el in group:
                group_sum += count[el]
            for el in group:
                cat_to_group_sum[el] = group_sum
        return cat_to_group_sum

    def _dummy_for_group_generated(self, category_id: str) -> bool:
        for idx, group in enumerate(self.group_categories):
            if category_id in group:
                is_generated = self.dummy_for_group_generated[idx]
                self.dummy_for_group_generated[idx] = True
                return is_generated
        return False


@pipeline_component_registry.register("SubImageLayoutService")
class SubImageLayoutService(PredictorPipelineComponent):
    """
    Component in which the selected ImageAnnotation can be selected with cropped images and presented to a detector.

    The detected DetectResults are transformed into ImageAnnotations and stored both in the cache of the parent image
    and in the cache of the sub image.

    If no objects are discovered, artificial objects can be added by means of a refinement process.

    **Example**

            detect_result_generator = DetectResultGenerator(categories_items)
            d_items = TPFrcnnDetector(item_config_path, item_weights_path, {"1": LayoutType.row,
            "2": LayoutType.column})
            item_component = SubImageLayoutService(d_items, LayoutType.table, {1: 7, 2: 8}, detect_result_generator)
    """

    def __init__(
        self,
        sub_image_detector: ObjectDetector,
        sub_image_names: Union[str, List[str]],
        category_id_mapping: Optional[Dict[int, int]] = None,
        detect_result_generator: Optional[DetectResultGenerator] = None,
        padder: Optional[PadTransform] = None,
    ):
        """
        :param sub_image_detector: object detector.
        :param sub_image_names: Category names of ImageAnnotations to be presented to the detector.
                                Attention: The selected ImageAnnotations must have: attr:`image` and: attr:`image.image`
                                not None.
        :param category_id_mapping: Mapping of category IDs. Usually, the category ids start with 1.
        :param detect_result_generator: 'DetectResultGenerator' instance. 'categories' attribute has to be the same as
                                        the 'categories' attribute of the 'sub_image_detector'. The generator will be
                                        responsible to create 'DetectionResult' for some categories, if they have not
                                        been detected by 'sub_image_detector'.
        :param padder: 'PadTransform' to pad an image before passing to a predictor. Will be also responsible for
                        inverse coordinate transformation.
        """

        if isinstance(sub_image_names, str):
            sub_image_names = [sub_image_names]

        self.sub_image_name = sub_image_names
        self.category_id_mapping = category_id_mapping
        self.detect_result_generator = detect_result_generator
        self.padder = padder
        super().__init__(self._get_name(sub_image_detector.name), sub_image_detector)
        if self.detect_result_generator is not None:
            assert self.detect_result_generator.categories == self.predictor.categories  # type: ignore

    def serve(self, dp: Image) -> None:
        """
        - Selection of ImageAnnotation to present to the detector.
        - Invoke the detector
        - Optionally invoke the DetectResultGenerator
        - Generate ImageAnnotations and dump to parent image and sub image.
        """
        sub_image_anns = dp.get_annotation_iter(category_names=self.sub_image_name)
        for sub_image_ann in sub_image_anns:
            if sub_image_ann.image is None:
                raise ValueError("sub_image_ann.image is None, but must be an image")
            np_image = sub_image_ann.image.image
            if self.padder:
                np_image = self.padder.apply_image(np_image)
            detect_result_list = self.predictor.predict(np_image)
            if self.padder and detect_result_list:
                boxes = np.array([detect_result.box for detect_result in detect_result_list])
                boxes_orig = self.padder.inverse_apply_coords(boxes)
                for idx, detect_result in enumerate(detect_result_list):
                    detect_result.box = boxes_orig[idx, :].tolist()
            if self.detect_result_generator:
                self.detect_result_generator.width = sub_image_ann.image.width
                self.detect_result_generator.height = sub_image_ann.image.height
                detect_result_list = self.detect_result_generator.create_detection_result(detect_result_list)

            for detect_result in detect_result_list:
                if self.category_id_mapping:
                    if detect_result.class_id:
                        detect_result.class_id = self.category_id_mapping.get(
                            detect_result.class_id, detect_result.class_id
                        )
                self.dp_manager.set_image_annotation(detect_result, sub_image_ann.annotation_id)

    def get_meta_annotation(self) -> JsonDict:
        assert isinstance(self.predictor, (ObjectDetector, PdfMiner))
        return dict(
            [
                ("image_annotations", self.predictor.possible_categories()),
                ("sub_categories", {}),
                # implicit setup of relations by using set_image_annotation with explicit annotation_id
                ("relationships", {parent: {Relationships.child} for parent in self.sub_image_name}),
                ("summaries", []),
            ]
        )

    @staticmethod
    def _get_name(predictor_name: str) -> str:
        return f"sub_image_{predictor_name}"

    def clone(self) -> "PredictorPipelineComponent":
        predictor = self.predictor.clone()
        padder_clone = None
        if self.padder:
            padder_clone = self.padder.clone()
        if not isinstance(predictor, ObjectDetector):
            raise ValueError(f"predictor must be of type ObjectDetector but is of type {type(predictor)}")
        return self.__class__(
            predictor,
            deepcopy(self.sub_image_name),
            deepcopy(self.category_id_mapping),
            deepcopy(self.detect_result_generator),
            padder_clone,
        )
