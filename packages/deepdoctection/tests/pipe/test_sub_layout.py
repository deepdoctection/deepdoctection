# -*- coding: utf-8 -*-
# File: test_sub_layout.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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

from typing import Mapping
from unittest.mock import MagicMock

from dd_core.datapoint import BoundingBox, ImageAnnotation
from dd_core.datapoint.image import Image
from dd_core.utils.object_types import ObjectTypes, get_type
from deepdoctection.extern.base import DetectionResult, ObjectDetector
from deepdoctection.pipe.sub_layout import DetectResultGenerator, SubImageLayoutService


def test_detect_result_generator(dp_image: Image) -> None:
    """
    Testing DetectResultGenerator creates DetectionResult correctly
    """

    layout_detect_results = [
        DetectionResult(box=[100.0, 160.0, 200.0, 260.0], score=0.63, class_id=2, class_name=get_type("title")),
        DetectionResult(box=[120.0, 120.0, 140.0, 120.0], score=0.03, class_id=5, class_name=get_type("figure")),
        DetectionResult(box=[50.0, 50.0, 150.0, 200.0], score=0.97, class_id=4, class_name=get_type("table")),
        DetectionResult(box=[100.0, 320.0, 150.0, 350.0], score=0.53, class_id=1, class_name=get_type("text")),
        DetectionResult(box=[200.0, 50.0, 250.0, 200.0], score=0.83, class_id=4, class_name=get_type("table")),
    ]

    categories_name_as_key: Mapping[ObjectTypes, int] = {
        get_type("text"): 1,
        get_type("title"): 2,
        get_type("table"): 3,
        get_type("figure"): 4,
        get_type("list"): 5,
    }
    detect_result_generator = DetectResultGenerator(
        categories_name_as_key,
        [[get_type("text")], [get_type("title")], [get_type("table")], [get_type("figure")], [get_type("list")]],
    )

    detect_result_generator.width = 600
    detect_result_generator.height = 400
    raw_anns = detect_result_generator.create_detection_result(layout_detect_results)

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

        self.sub_image_layout_service = SubImageLayoutService(self._cell_detector, get_type("table"))

    def test_pass_datapoint(
        self,
        dp_image: Image,
        layout_annotations,
        cell_detect_results: list[list[DetectionResult]],
    ) -> None:
        """
        Testing pass_datapoint
        """

        for img_ann in layout_annotations(segmentation=False):
            dp_image.dump(img_ann)
            dp_image.image_ann_to_image(img_ann.annotation_id, True)

        global_cell_boxes = [
            [
                BoundingBox(absolute_coords=False, ulx=0.11666667, uly=0.175, lrx=0.125, lry=0.2),
                BoundingBox(absolute_coords=False, ulx=0.15, uly=0.225, lrx=0.16666667, lry=0.25),
            ],
            [BoundingBox(absolute_coords=False, ulx=0.35833333, uly=0.175, lrx=0.36666667, lry=0.2)],
        ]

        self._cell_detector.predict = MagicMock(side_effect=cell_detect_results)

        dp = self.sub_image_layout_service.pass_datapoint(dp_image)
        anns = dp.get_annotation(category_names=get_type("table"))

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
        assert local_box_ftfc == first_table_first_cell.bounding_box.transform(
            first_table_ann.image.width, first_table_ann.image.height
        )

        global_box_ftsc = first_table_second_cell.get_bounding_box(dp.image_id)
        assert global_box_ftsc == exp_global_boxes_first_table[1]
        local_box_ftsc = first_table_second_cell.get_bounding_box(first_table_ann.annotation_id)
        assert local_box_ftsc == first_table_second_cell.bounding_box.transform(
            first_table_ann.image.width, first_table_ann.image.height
        )

        global_box_stfc = second_table_first_cell.get_bounding_box(dp.image_id)
        assert global_box_stfc == exp_global_boxes_scd_table[0]
        local_box_stfc = second_table_first_cell.get_bounding_box(second_table_ann.annotation_id)
        assert local_box_stfc == second_table_first_cell.bounding_box.transform(
            second_table_ann.image.width, second_table_ann.image.height
        )

    def test_pass_datapoint_when_sub_images_do_not_have_a_crop(
        self,
        dp_image: Image,
        layout_annotations,
        cell_detect_results: list[list[DetectionResult]],
    ) -> None:
        """If an sub image does not have a crop, a ValueError was raised previously. Now it should be fixed."""

        for img_ann in layout_annotations(segmentation=False):
            dp_image.dump(img_ann)
            dp_image.image_ann_to_image(img_ann.annotation_id, True)

        # Arrange
        self._cell_detector.predict = MagicMock(side_effect=cell_detect_results)
        for ann in dp_image.get_annotation():
            if ann.image is not None:
                ann.image.clear_image()

        # Act

        try:
            self.sub_image_layout_service.pass_datapoint(dp_image)
        except ValueError:
            assert False, "ValueError was raised, because the sub image does not have a crop"
