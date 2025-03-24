# -*- coding: utf-8 -*-
# File: test_sub_layout.py

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
Testing module pipe.cell
"""

from typing import List, Mapping
from unittest.mock import MagicMock

from pytest import mark

from deepdoctection import ObjectTypes
from deepdoctection.datapoint import BoundingBox, Image
from deepdoctection.extern.base import DetectionResult, ObjectDetector
from deepdoctection.pipe.sub_layout import DetectResultGenerator, SubImageLayoutService
from deepdoctection.utils.settings import LayoutType


@mark.basic
def test_detect_result_generator(dp_image: Image, layout_detect_results: List[DetectionResult]) -> None:
    """
    Testing DetectResultGenerator creates DetectionResult correctly
    """

    # Arrange
    categories_name_as_key: Mapping[ObjectTypes, int] = {
        LayoutType.TEXT: 1,
        LayoutType.TITLE: 2,
        LayoutType.TABLE: 3,
        LayoutType.FIGURE: 4,
        LayoutType.LIST: 5,
    }
    detect_result_generator = DetectResultGenerator(
        categories_name_as_key,
        [[LayoutType.TEXT], [LayoutType.TITLE], [LayoutType.TABLE], [LayoutType.FIGURE], [LayoutType.LIST]],
    )

    # Act
    detect_result_generator.width = 600
    detect_result_generator.height = 400
    raw_anns = detect_result_generator.create_detection_result(layout_detect_results)

    # Assert
    raw_ann_cats = {raw_ann.class_id for raw_ann in raw_anns}
    assert raw_ann_cats == {None, 1, 2, 4, 5}

    assert raw_anns[5].box == [0.0, 0.0, dp_image.width, dp_image.height]


class TestSubImageLayoutService:
    """
    Test SubImageLayoutService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._cell_detector = MagicMock(spec=ObjectDetector)
        self._cell_detector.model_id = "test_model"
        self._cell_detector.name = "mock_cell_detector"

        self.sub_image_layout_service = SubImageLayoutService(self._cell_detector, LayoutType.TABLE)

    @mark.basic
    def test_pass_datapoint(
        self,
        dp_image_with_layout_anns: Image,
        cell_detect_results: List[List[DetectionResult]],
        global_cell_boxes: List[List[BoundingBox]],
    ) -> None:
        """
        Testing pass_datapoint
        """

        # Arrange
        self._cell_detector.predict = MagicMock(side_effect=cell_detect_results)

        # Act
        dp = self.sub_image_layout_service.pass_datapoint(dp_image_with_layout_anns)
        anns = dp.get_annotation(category_names=LayoutType.TABLE)

        # Assert
        assert len(anns) == 2

        first_table_ann = anns[0]
        second_table_ann = anns[1]

        exp_global_boxes_first_table = global_cell_boxes[0]
        exp_global_boxes_scd_table = global_cell_boxes[1]

        first_table_cell_anns = first_table_ann.image.get_annotation()  # type: ignore
        second_table_cell_anns = second_table_ann.image.get_annotation()  # type: ignore
        assert len(first_table_cell_anns) == 2
        assert len(second_table_cell_anns) == 1

        first_table_first_cell = first_table_cell_anns[0]
        first_table_second_cell = first_table_cell_anns[1]
        second_table_first_cell = second_table_cell_anns[0]

        global_box_ftfc = first_table_first_cell.get_bounding_box(dp.image_id)
        assert global_box_ftfc == exp_global_boxes_first_table[0]
        local_box_ftfc = first_table_first_cell.get_bounding_box(first_table_ann.annotation_id)
        assert local_box_ftfc == first_table_first_cell.bounding_box

        global_box_ftsc = first_table_second_cell.get_bounding_box(dp.image_id)
        assert global_box_ftsc == exp_global_boxes_first_table[1]
        local_box_ftsc = first_table_second_cell.get_bounding_box(first_table_ann.annotation_id)
        assert local_box_ftsc == first_table_second_cell.bounding_box

        global_box_stfc = second_table_first_cell.get_bounding_box(dp.image_id)
        assert global_box_stfc == exp_global_boxes_scd_table[0]
        local_box_stfc = second_table_first_cell.get_bounding_box(second_table_ann.annotation_id)
        assert local_box_stfc == second_table_first_cell.bounding_box

    def test_pass_datapoint_when_sub_images_do_not_have_a_crop(
        self,
        dp_image_with_layout_anns: Image,
        cell_detect_results: List[List[DetectionResult]],
    ) -> None:
        """If an sub image does not have a crop, a ValueError was raised previously. Now it should be fixed."""

        # Arrange
        self._cell_detector.predict = MagicMock(side_effect=cell_detect_results)
        for ann in dp_image_with_layout_anns.get_annotation():
            if ann.image is not None:
                ann.image.clear_image()

        # Act

        try:
            self.sub_image_layout_service.pass_datapoint(dp_image_with_layout_anns)
        except ValueError:
            assert False, "ValueError was raised, because the sub image does not have a crop"
