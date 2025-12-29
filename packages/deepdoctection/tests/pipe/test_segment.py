# -*- coding: utf-8 -*-
# File: test_segment.py

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

"""
Unit tests for table segmentation functionalities.

This module includes test cases for validating the behavior and correctness
of table segmentation-related processes. These tests ensure that bounding
boxes, subcategories, and annotations are correctly assigned and manipulated
during table segmentation tasks.

"""

from typing import List

from dd_core.datapoint import BoundingBox, CategoryAnnotation, Image, get_type
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.segment import (
    PubtablesSegmentationService,
    SegmentationResult,
    TableSegmentationService,
    create_intersection_cells,
    stretch_items,
    tile_tables_with_items_per_table,
)


def test_stretch_items(dp_image_tab_cell_item: Image) -> None:
    """test stretch_items"""

    dp = dp_image_tab_cell_item
    table_name = get_type("table")
    item_names = [get_type("row"), get_type("column")]

    dp = stretch_items(dp, table_name, item_names[0], item_names[1], 0.001, 0.001)

    rows = dp.get_annotation(category_names=item_names[0])
    cols = dp.get_annotation(category_names=item_names[1])

    rows_expected = [
        BoundingBox(absolute_coords=False, ulx=0.16833334, uly=0.5, lrx=0.33083333, lry=0.625),
        BoundingBox(absolute_coords=False, ulx=0.16833334, uly=0.75, lrx=0.33083333, lry=0.85),
    ]
    cols_expected = [BoundingBox(absolute_coords=False, ulx=0.18333333, uly=0.25166667, lrx=0.2, lry=0.9975)]

    for row, row_expected in zip(rows, rows_expected):
        row_embedding = row.get_bounding_box(dp.image_id)
        assert row_embedding == row_expected

    for col, col_expected in zip(cols, cols_expected):
        col_embedding = col.get_bounding_box(dp.image_id)
        assert col_embedding == col_expected


class TestTableSegmentationService:
    """
    Test TableSegmentationService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.table_segmentation_service = TableSegmentationService(
            "iou",
            0.001,
            0.001,
            False,
            0.001,
            0.001,
            get_type("table"),
            [get_type("column_header"), get_type("body"), get_type("cell")],
            [get_type("row"), get_type("column")],
            [get_type("row_number"), get_type("column_number")],
        )

    def test_pass_datapoint(self, dp_image_tab_cell_item: Image, dp_image_fully_segmented: Image) -> None:
        """test pass_datapoint"""

        # Arrange
        dp = dp_image_tab_cell_item
        dp_expected = dp_image_fully_segmented

        # Act
        dp = self.table_segmentation_service.pass_datapoint(dp)

        # Assert items have correctly assigned sub categories row/col number
        for item_name, sub_item_name in zip(
            self.table_segmentation_service.item_names,
            self.table_segmentation_service.sub_item_names,
        ):
            items = dp.get_annotation(category_names=item_name)
            items_expected = dp_expected.get_annotation(category_names=item_name)
            for el in zip(items, items_expected):
                item_cat = el[0].get_sub_category(sub_item_name)
                item_cat_expected = el[1].get_sub_category(sub_item_name)
                assert item_cat.category_name == item_cat_expected.category_name
                assert item_cat.category_id == item_cat_expected.category_id

        # Assert cells have correctly assigned sub categories row/col/rs/cs number
        cells = dp.get_annotation(category_names=self.table_segmentation_service.cell_names)
        cells_expected = dp_expected.get_annotation(category_names=self.table_segmentation_service.cell_names)

        for el in zip(cells, cells_expected):
            cell, cell_expected = el[0], el[1]
            row_sub_cat = cell.get_sub_category(get_type("row_number"))
            row_sub_cat_expected = cell_expected.get_sub_category(get_type("row_number"))
            assert row_sub_cat.category_name == row_sub_cat_expected.category_name
            assert row_sub_cat.category_id == row_sub_cat_expected.category_id

            col_sub_cat = cell.get_sub_category(get_type("column_number"))
            col_sub_cat_expected = cell_expected.get_sub_category(get_type("column_number"))
            assert col_sub_cat.category_name == col_sub_cat_expected.category_name
            assert col_sub_cat.category_id == col_sub_cat_expected.category_id

            rs_sub_cat = cell.get_sub_category(get_type("row_span"))
            rs_sub_cat_expected = cell_expected.get_sub_category(get_type("row_span"))
            assert rs_sub_cat.category_name == rs_sub_cat_expected.category_name
            assert rs_sub_cat.category_id == rs_sub_cat_expected.category_id

            cs_sub_cat = cell.get_sub_category(get_type("column_span"))
            cs_sub_cat_expected = cell_expected.get_sub_category(get_type("column_span"))
            assert cs_sub_cat.category_name == cs_sub_cat_expected.category_name
            assert cs_sub_cat.category_id == cs_sub_cat_expected.category_id


def test_tile_tables_with_items_per_table(
    dp_image_item_stretched: Image,
    row_sub_cats: List[CategoryAnnotation],
    column_sub_cats: List[CategoryAnnotation],
) -> None:
    """test tile_tables_with_items_per_table"""

    # Arrange
    dp = dp_image_item_stretched
    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))

    for row, col, row_sub_cat, col_sub_cat in zip(rows, cols, row_sub_cats, column_sub_cats):
        row.dump_sub_category(get_type("row_number"), row_sub_cat)
        col.dump_sub_category(get_type("column_number"), col_sub_cat)

    table = dp.get_annotation(category_names=get_type("table"))
    item_names = [get_type("row"), get_type("column")]  # row names must be before column name

    # Act
    dp = tile_tables_with_items_per_table(dp, table[0], item_names[0])
    dp = tile_tables_with_items_per_table(dp, table[0], item_names[1])

    # Assert
    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))

    first_row_box = rows[0].get_bounding_box(dp.image_id)
    second_row_box = rows[1].get_bounding_box(dp.image_id)

    first_col_box = cols[0].get_bounding_box(dp.image_id)
    second_col_box = cols[1].get_bounding_box(dp.image_id)

    row_box_tiling_table = [
        BoundingBox(
            absolute_coords=False,
            ulx=0.16833334,
            uly=0.2525,
            lrx=0.33083333,
            lry=0.625,
            width=0.16249999,
            height=0.3725,
        ),
        BoundingBox(
            absolute_coords=False,
            ulx=0.16833334,
            uly=0.625,
            lrx=0.33083333,
            lry=0.9975,
            width=0.16249999,
            height=0.3725,
        ),
    ]

    col_box_tiling_table = [
        BoundingBox(
            absolute_coords=False,
            ulx=0.16833334,
            uly=0.25166667,
            lrx=0.2,
            lry=0.9975,
            width=0.03166666,
            height=0.74583333,
        ),
        BoundingBox(
            absolute_coords=False,
            ulx=0.2,
            uly=0.25166667,
            lrx=0.33166666,
            lry=0.9975,
            width=0.13166666,
            height=0.74583333,
        ),
    ]

    assert first_row_box == row_box_tiling_table[0]
    assert second_row_box == row_box_tiling_table[1]

    assert first_col_box == col_box_tiling_table[0]
    assert second_col_box == col_box_tiling_table[1]


class TestTableSegmentationServiceWhenTableFullyTiled:
    """
    Test TableSegmentationService integrated when the table is fully tiled by rows and columns
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.tp_table_segmentation_service = TableSegmentationService(
            "iou",
            0.001,
            0.001,
            True,
            0.001,
            0.001,
            get_type("table"),
            [get_type("column_header"), get_type("body"), get_type("cell")],
            [get_type("row"), get_type("column")],
            [get_type("row_number"), get_type("column_number")],
        )

    def test_integration_pipeline_component(
        self, dp_image_tab_cell_item: Image, dp_image_fully_segmented_fully_tiled: Image
    ) -> None:
        """
        Integration test for pipeline component
        """

        # Arrange
        dp = dp_image_tab_cell_item
        dp_expected = dp_image_fully_segmented_fully_tiled

        # Act
        dp = self.tp_table_segmentation_service.pass_datapoint(dp)

        # Assert
        cells = dp.get_annotation(category_names=self.tp_table_segmentation_service.cell_names)

        cells_expected = dp_expected.get_annotation(category_names=self.tp_table_segmentation_service.cell_names)

        assert len(cells) == len(cells_expected)

        for cell, cell_expected in zip(cells, cells_expected):
            assert cell.get_sub_category(get_type("row_number")) == cell_expected.get_sub_category(
                get_type("row_number")
            )
            assert cell.get_sub_category(get_type("column_number")) == cell_expected.get_sub_category(
                get_type("column_number")
            )
            assert cell.get_sub_category(get_type("row_span")) == cell_expected.get_sub_category(get_type("row_span"))
            assert cell.get_sub_category(get_type("column_span")) == cell_expected.get_sub_category(
                get_type("column_span")
            )


def test_create_intersection_cells(dp_image_tab_cell_item: Image) -> None:
    """
    Test create_intersection_cells generates cells from intersecting rows and columns and creates
    """

    # Arrange
    dp = dp_image_tab_cell_item

    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))
    for idx, items in enumerate(zip(rows, cols)):
        items[0].dump_sub_category(
            get_type("row_number"), CategoryAnnotation(category_name=get_type("row_number"), category_id=idx + 1)
        )
        items[1].dump_sub_category(
            get_type("column_number"), CategoryAnnotation(category_name=get_type("column_number"), category_id=idx + 1)
        )

    table = dp.get_annotation(category_names=get_type("table"))[0]
    table_ann_id = table.annotation_id
    detect_result_cells, segment_result_cells = create_intersection_cells(
        rows, cols, table_ann_id, [get_type("row_number"), get_type("column_number")]
    )
    expected_detect_result = [
        DetectionResult(
            box=[0.15, 0.33333333, 0.2, 0.5], absolute_coords=False, class_id=None, class_name=get_type("cell")
        ),
        DetectionResult(
            box=[0.4, 0.33333333, 0.5, 0.5], absolute_coords=False, class_id=None, class_name=get_type("cell")
        ),
        DetectionResult(
            box=[0.15, 0.66666667, 0.2, 0.8], absolute_coords=False, class_id=None, class_name=get_type("cell")
        ),
        DetectionResult(
            box=[0.4, 0.66666667, 0.5, 0.8], absolute_coords=False, class_id=None, class_name=get_type("cell")
        ),
    ]
    expected_segment_result = [
        SegmentationResult(annotation_id="", row_num=1, col_num=1, rs=1, cs=1),
        SegmentationResult(annotation_id="", row_num=1, col_num=2, rs=1, cs=1),
        SegmentationResult(annotation_id="", row_num=2, col_num=1, rs=1, cs=1),
        SegmentationResult(annotation_id="", row_num=2, col_num=2, rs=1, cs=1),
    ]

    assert len(detect_result_cells) == 4
    assert detect_result_cells == expected_detect_result
    assert len(segment_result_cells) == 4
    assert expected_segment_result == segment_result_cells


class TestPubtablesSegmentationService:
    """
    Test PubtablesSegmentationService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.table_segmentation_service = PubtablesSegmentationService(
            "ioa",
            0.4,
            0.4,
            True,
            0.001,
            0.001,
            get_type("table"),
            [
                get_type("spanning"),
                get_type("row_header"),
                get_type("column_header"),
                get_type("projected_row_header"),
                get_type("cell"),
            ],
            [
                get_type("spanning"),
                get_type("row_header"),
                get_type("column_header"),
                get_type("projected_row_header"),
            ],
            [get_type("row"), get_type("column")],
            [get_type("row_number"), get_type("column_number")],
            [get_type("row_header"), get_type("column_header")],
            [0.001, 0.001],
        )

    def test_pass_datapoint(self, dp_image_tab_cell_item: Image) -> None:
        """test pass_datapoint"""

        dp = dp_image_tab_cell_item
        cells_ann_ids = [ann.annotation_id for ann in dp.get_annotation(category_names=get_type("cell"))]
        table = dp.get_annotation(category_names=get_type("table"))[0]

        dp.remove(annotation_ids=cells_ann_ids)

        tab_cells_ann_ids = [
            ann.annotation_id for ann in table.image.get_annotation(category_names=get_type("cell"))  # type: ignore
        ]

        table.image.remove(annotation_ids=tab_cells_ann_ids)  # type: ignore

        dp = self.table_segmentation_service.pass_datapoint(dp)

        rows = dp.get_annotation(category_names=get_type("row"))
        cols = dp.get_annotation(category_names=get_type("column"))
        cells = dp.get_annotation(category_names=get_type("cell"))

        assert len(rows) == 2
        assert len(cols) == 2
        assert len(cells) == 4
