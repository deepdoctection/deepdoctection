# -*- coding: utf-8 -*-
# File: test_cell.py

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

from typing import List
from unittest.mock import MagicMock

from deepdoctection.datapoint import BoundingBox, Image
from deepdoctection.datasets import DatasetCategories
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.cell import DetectResultGenerator, SubImageLayoutService
from deepdoctection.utils.settings import names


def test_detect_result_generator(
    dataset_categories: DatasetCategories, dp_image: Image, layout_detect_results: List[DetectionResult]
) -> None:
    """
    Testing DetectResultGenerator creates DetectionResult correctly
    """

    # Arrange
    categories = dataset_categories.get_categories()
    detect_result_generator = DetectResultGenerator(
        categories, dp_image.width, dp_image.height, [["1"], ["2"], ["3"], ["4"], ["5"]]  # type: ignore
    )

    # Act
    raw_anns = detect_result_generator.create_detection_result(layout_detect_results)

    # Assert
    raw_ann_cats = {raw_ann.class_id for raw_ann in raw_anns}
    assert raw_ann_cats == {1, 2, 3, 4, 5}

    assert raw_anns[5].box == [0.0, 0.0, dp_image.width, dp_image.height]


class TestSubImageLayoutService:
    """
    Test SubImageLayoutService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._cell_detector = MagicMock()
        self.sub_image_layout_service = SubImageLayoutService(self._cell_detector, names.C.TAB)

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
        anns = dp.get_annotation(category_names=names.C.TAB)

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

        global_box_ftfc = first_table_first_cell.image.get_embedding(dp.image_id)
        assert global_box_ftfc == exp_global_boxes_first_table[0]
        local_box_ftfc = first_table_first_cell.image.get_embedding(first_table_ann.annotation_id)
        assert local_box_ftfc == first_table_first_cell.bounding_box

        global_box_ftsc = first_table_second_cell.image.get_embedding(dp.image_id)
        assert global_box_ftsc == exp_global_boxes_first_table[1]
        local_box_ftsc = first_table_second_cell.image.get_embedding(first_table_ann.annotation_id)
        assert local_box_ftsc == first_table_second_cell.bounding_box

        global_box_stfc = second_table_first_cell.image.get_embedding(dp.image_id)
        assert global_box_stfc == exp_global_boxes_scd_table[0]
        local_box_stfc = second_table_first_cell.image.get_embedding(second_table_ann.annotation_id)
        assert local_box_stfc == second_table_first_cell.bounding_box
