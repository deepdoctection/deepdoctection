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
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Dict, List, Mapping, Optional, Union

from ..datapoint.image import Image
from ..extern.base import DetectionResult, ObjectDetector, PdfMiner
from ..utils.detection_types import JsonDict
from ..utils.settings import ObjectTypes, Relationships
from .base import PredictorPipelineComponent
from .registry import pipeline_component_registry


class DetectResultGenerator:
    """
    Use: class: `DetectResultGenerator` to refine raw detection results.

    Certain pipeline components depend on, for example, at least one object being detected. If this is not the
    case, the generator can generate a DetectResult with a default setting. If no object was discovered for a
    category, a DetectResult with the dimensions of the original image is generated and added to the remaining
    DetectResults.
    """

    def __init__(
        self,
        categories: Mapping[str, ObjectTypes],
        image_width: float,
        image_height: float,
        group_categories: List[List[str]],
    ) -> None:
        """
        :param categories: The dict of all possible detection categories
        :param image_width: Used for generating a DetectResult of image_width
        :param image_height: Used for generating a DetectResult of image_height
        :param group_categories: If you only want to generate only one DetectResult for a group of categories, provided
                                 that the sum of the group is less than one, then you can pass a list of list for
                                 grouping category ids.
        """
        self.categories = categories
        self.width = image_width
        self.height = image_height
        self.group_categories = group_categories
        self.dummy_for_group_generated = [False for _ in self.group_categories]

    def create_detection_result(self, detect_result_list: List[DetectionResult]) -> List[DetectionResult]:
        """
        Adds DetectResults for which no object was detected to the list.

        :param detect_result_list: DetectResults of a previously run ObjectDetector
        :return: refined list
        """
        count = self._create_condition(detect_result_list)
        for category_id in self.categories:
            if count[category_id] < 1:
                if not self._dummy_for_group_generated(category_id):
                    detect_result_list.append(
                        DetectionResult(
                            box=[0.0, 0.0, self.width, self.height],
                            class_id=int(category_id),
                            class_name=self.categories[category_id],
                            score=0.0,
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

        .. code-block:: python

            d_items = TPFrcnnDetector(item_config_path, item_weights_path, {"1": "ROW", "2": "COLUMNS"})
            item_component = SubImageLayoutService(d_items, names.C.TAB, {1: 6}, True)
    """

    def __init__(
        self,
        sub_image_detector: ObjectDetector,
        sub_image_names: Union[str, List[str]],
        category_id_mapping: Optional[Dict[int, int]] = None,
        add_dummy_detection: bool = False,
    ):
        """
        :param sub_image_detector: object detector.
        :param sub_image_names: Category names of ImageAnnotations to be presented to the detector.
                                Attention: The selected ImageAnnotations must have: attr:`image` and: attr:`image.image`
                                not None.
        :param category_id_mapping: Mapping of category IDs. Usually, the category ids start with 1.
        :param add_dummy_detection: If set to True will add an ImageAnnotation for each class for which no sample have
                                    been detected.
        """

        if isinstance(sub_image_names, str):
            sub_image_names = [sub_image_names]

        self.sub_image_name = sub_image_names
        self.add_dummy_detection = add_dummy_detection
        self.dummy_generator_cls = None
        self.category_id_mapping = category_id_mapping
        if add_dummy_detection:
            if category_id_mapping is None:
                raise ValueError("Using DetectResult dummy generator requires passing a category_id_mapping")
            self.dummy_generator_cls = DetectResultGenerator
            group_categories_dict = defaultdict(list)
            for group in category_id_mapping.items():
                group_categories_dict[group[1]].append(str(group[0]))
            group_categories = list(group_categories_dict.values())
            self.group_categories = group_categories
        super().__init__(self._get_name(sub_image_detector.name), sub_image_detector)

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
            detect_result_list = self.predictor.predict(sub_image_ann.image.image)
            if self.has_dummy_generator():
                if hasattr(self.predictor, "categories"):
                    generator = self.get_dummy_generator(
                        self.predictor.categories,  # type: ignore
                        sub_image_ann.image.width,
                        sub_image_ann.image.height,
                        self.group_categories,
                    )
                    detect_result_list = generator.create_detection_result(detect_result_list)
                else:
                    raise AttributeError("predictor must have attribute categories when using dummy generation")
            for detect_result in detect_result_list:
                self.dp_manager.set_image_annotation(detect_result, sub_image_ann.annotation_id)

    def get_dummy_generator(
        self, categories: Dict[str, str], image_width: float, image_height: float, group_categories: List[List[str]]
    ) -> DetectResultGenerator:
        """
        Create a DetectResultGenerator with some configs

        :param categories: A dict of possible categories
        :param image_width: width
        :param image_height: height
        :param group_categories: If you only want to generate only one DetectResult for a group of categories, provided
                                 that the sum of the group is less than one, then you can pass a list of list for
                                 grouping category ids.
        :return: DetectResultGenerator instance
        """
        if self.has_dummy_generator():
            return self.dummy_generator_cls(categories, image_width, image_height, group_categories)  # type: ignore
        raise ValueError("No dummy generator available")

    def has_dummy_generator(self) -> bool:
        """
        :return: True if DetectResultGenerator is available
        """
        if self.dummy_generator_cls is None:
            return False
        return True

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
        if not isinstance(predictor, ObjectDetector):
            raise ValueError(f"predictor must be of type ObjectDetector but is of type {type(predictor)}")
        return self.__class__(
            predictor,
            deepcopy(self.sub_image_name),
            deepcopy(self.category_id_mapping),
            deepcopy(self.add_dummy_detection),
        )
