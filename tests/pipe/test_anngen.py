# -*- coding: utf-8 -*-
# File: test_anngen.py

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
Testing module pipe.anngen
"""

from typing import List

from deepdoctection.datapoint import Image, ImageAnnotation
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.anngen import DatapointManager


class TestDatapointManager:
    """
    Testing DatapointManager functions
    """

    @staticmethod
    def test_set_image_annotation(
        dp_image: Image, layout_detect_results: List[DetectionResult], layout_annotations: List[ImageAnnotation]
    ) -> None:
        """
        test set_image_annotation
        """
        # Arrange
        dp_manager = DatapointManager(None)
        dp_manager.datapoint = dp_image

        # Act
        ann_id = dp_manager.set_image_annotation(layout_detect_results[0])

        # Assert
        assert ann_id is not None
        ann = dp_manager.datapoint.get_annotation()
        assert ann[0] == layout_annotations[0]
        assert ann_id == layout_annotations[0].annotation_id
        assert dp_manager._cache_anns[ann_id] == layout_annotations[0]  # pylint: disable=W0212

    @staticmethod
    def test_set_image_annotation_with_image(dp_image: Image, layout_detect_results: List[DetectionResult]) -> None:
        """
        test set_image_annotation with image_ann_to_image
        """
        # Arrange
        dp_manager = DatapointManager(None)
        dp_manager.datapoint = dp_image

        # Act
        dp_manager.set_image_annotation(layout_detect_results[0], to_image=True)

        # Assert
        ann = dp_manager.datapoint.get_annotation()
        assert ann[0].image is not None
        assert ann[0].bounding_box == ann[0].image.get_embedding(dp_image.image_id)

    @staticmethod
    def test_set_image_annotation_to_image_ann(dp_image: Image, layout_detect_results: List[DetectionResult]) -> None:
        """
        test set_image_annotation with ann to image ann
        """

        # Arrange
        dp_manager = DatapointManager(None)
        dp_manager.datapoint = dp_image
        img_ann_id = dp_manager.set_image_annotation(layout_detect_results[0], to_image=True)

        # Act
        ann_id = dp_manager.set_image_annotation(layout_detect_results[2], to_annotation_id=img_ann_id)

        # Assert
        ann = dp_manager.datapoint.get_annotation(annotation_ids=ann_id)
        img_ann = dp_manager.datapoint.get_annotation(annotation_ids=img_ann_id)[0]

        assert img_ann.image is not None
        ann_from_img_ann = img_ann.image.get_annotation(annotation_ids=ann_id)
        assert ann == ann_from_img_ann
        assert dp_manager._cache_anns[ann_id]  # type:ignore # pylint: disable=W0212

    @staticmethod
    def test_set_category_annotation(dp_image: Image, layout_detect_results: List[DetectionResult]) -> None:
        """
        test set_category_annotation
        """

        # Arrange
        dp_manager = DatapointManager(None)
        dp_manager.datapoint = dp_image
        ann_id = dp_manager.set_image_annotation(layout_detect_results[0])

        # Act
        assert ann_id is not None
        dp_manager.set_category_annotation("foo", 5, "FOO", ann_id, 0.8)

        # Assert
        ann = dp_manager.datapoint.get_annotation(annotation_ids=ann_id)
        cat_ann = ann[0].get_sub_category("FOO")

        assert cat_ann.category_id == "5"
        assert cat_ann.score == 0.8
        assert cat_ann.category_name == "foo"

    @staticmethod
    def test_set_container_annotation(dp_image: Image, layout_detect_results: List[DetectionResult]) -> None:
        """
        test set_container_annotation
        """

        # Arrange
        dp_manager = DatapointManager(None)
        dp_manager.datapoint = dp_image
        ann_id = dp_manager.set_image_annotation(layout_detect_results[0])

        # Act
        assert ann_id is not None
        cont_ann_id = dp_manager.set_container_annotation("foo", 5, "FOO", ann_id, "hello world", 0.8)

        # Assert
        ann = dp_manager.datapoint.get_annotation(annotation_ids=ann_id)
        cont_ann = ann[0].get_sub_category("FOO")

        assert cont_ann.category_id == "5"
        assert cont_ann.score == 0.8
        assert cont_ann.category_name == "foo"
        assert cont_ann.value == "hello world"  # type: ignore
        assert cont_ann.annotation_id == cont_ann_id

    @staticmethod
    def test_summary_annotation(dp_image: Image, layout_detect_results: List[DetectionResult]) -> None:
        """
        test summary_annotation
        """

        # Arrange
        dp_manager = DatapointManager(None)
        dp_manager.datapoint = dp_image
        ann_id = dp_manager.set_image_annotation(layout_detect_results[0], to_image=True)

        # Act
        summ_id_1 = dp_manager.set_summary_annotation("foo", 1)
        summ_id_2 = dp_manager.set_summary_annotation("bak", 2, ann_id)
        ann = dp_manager.datapoint.get_annotation(annotation_ids=ann_id)

        # Assert
        cat_1 = dp_manager.datapoint.summary.get_sub_category("foo")  # type: ignore
        assert cat_1.annotation_id == summ_id_1
        assert cat_1.category_name == "foo"
        assert cat_1.category_id == "1"

        cat_2 = ann[0].image.summary.get_sub_category("bak")  # type: ignore
        assert cat_2.annotation_id == summ_id_2
        assert cat_2.category_name == "bak"
        assert cat_2.category_id == "2"
