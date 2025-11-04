# -*- coding: utf-8 -*-
# File: test_segment.py

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
Testing module pipe.segment
"""

from typing import List, Sequence, Union

from pytest import mark

from deepdoctection.datapoint import BoundingBox, CategoryAnnotation, Image
from deepdoctection.extern.base import DetectionResult
from deepdoctection.pipe.segment import (
    PubtablesSegmentationService,
    SegmentationResult,
    TableSegmentationService,
    create_intersection_cells,
    stretch_items,
    tile_tables_with_items_per_table,
)
from deepdoctection.utils.object_types import CellType, LayoutType


@mark.basic
def test_stretch_items(dp_image_tab_cell_item: Image, dp_image_item_stretched: Image) -> None:
    """test stretch_items"""
    # Arrange
    dp = dp_image_tab_cell_item
    dp_expected = dp_image_item_stretched
    table_name = LayoutType.TABLE
    item_names = [LayoutType.ROW, LayoutType.COLUMN]

    # Act
    dp = stretch_items(dp, table_name, item_names[0], item_names[1], 0.001, 0.001)

    # Assert
    tables = dp.get_annotation(category_names=table_name)
    assert len(tables) == 1

    rows = dp.get_annotation(category_names=item_names[0])
    cols = dp.get_annotation(category_names=item_names[1])

    rows_expected = dp_expected.get_annotation(category_names=item_names[0])
    cols_expected = dp_expected.get_annotation(category_names=item_names[1])

    for row, row_expected in zip(rows, rows_expected):
        row_embedding = row.get_bounding_box(dp.image_id)
        row_expected_embedding = row_expected.get_bounding_box(dp_expected.image_id)
        assert row_embedding == row_expected_embedding

    for col, col_expected in zip(cols, cols_expected):
        col_embedding = col.get_bounding_box(dp.image_id)
        col_expected_embedding = col_expected.get_bounding_box(dp_expected.image_id)
        assert col_embedding == col_expected_embedding


class TestTableSegmentationService:
    """
    Test TableSegmentationService
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self._segment_rule = "iou"
        self._iou_threshold_rows = 0.001
        self._iou_threshold_cols = 0.001
        self._ioa_threshold_rows = 0.4
        self._ioa_threshold_cols = 0.4
        self._remove_iou_threshold_rows = 0.001
        self._remove_iou_threshold_cols = 0.001
        self._tile_table_with_items = False
        self.table_name = LayoutType.TABLE
        self.cell_names = [CellType.COLUMN_HEADER, CellType.BODY, LayoutType.CELL]
        self.item_names = [LayoutType.ROW, LayoutType.COLUMN]
        self.sub_item_names = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]

        self.table_segmentation_service = TableSegmentationService(
            self._segment_rule,  # type: ignore
            self._iou_threshold_rows if self._segment_rule in ["iou"] else self._ioa_threshold_rows,
            self._iou_threshold_cols if self._segment_rule in ["iou"] else self._ioa_threshold_cols,
            self._tile_table_with_items,
            self._remove_iou_threshold_rows,
            self._remove_iou_threshold_cols,
            self.table_name,
            self.cell_names,
            self.item_names,
            self.sub_item_names,
        )

    @mark.basic
    def test_pass_datapoint(self, dp_image_tab_cell_item: Image, dp_image_fully_segmented: Image) -> None:
        """test pass_datapoint"""

        # Arrange
        dp = dp_image_tab_cell_item
        dp_expected = dp_image_fully_segmented

        # Act
        dp = self.table_segmentation_service.pass_datapoint(dp)

        # Assert items have correctly assigned sub categories row/col number
        for item_name, sub_item_name in zip(
            self.table_segmentation_service.item_names,  # pylint: disable=W0212
            self.table_segmentation_service.sub_item_names,  # pylint: disable=W0212
        ):
            items = dp.get_annotation(category_names=item_name)
            items_expected = dp_expected.get_annotation(category_names=item_name)
            for el in zip(items, items_expected):
                item_cat = el[0].get_sub_category(sub_item_name)
                item_cat_expected = el[1].get_sub_category(sub_item_name)
                assert item_cat.category_name == item_cat_expected.category_name
                assert item_cat.category_id == item_cat_expected.category_id

        # Assert cells have correctly assigned sub categories row/col/rs/cs number
        cells = dp.get_annotation(category_names=self.table_segmentation_service.cell_names)  # pylint: disable=W0212
        cells_expected = dp_expected.get_annotation(
            category_names=self.table_segmentation_service.cell_names  # pylint: disable=W0212
        )

        for el in zip(cells, cells_expected):
            cell, cell_expected = el[0], el[1]
            row_sub_cat = cell.get_sub_category(CellType.ROW_NUMBER)
            row_sub_cat_expected = cell_expected.get_sub_category(CellType.ROW_NUMBER)
            assert row_sub_cat.category_name == row_sub_cat_expected.category_name
            assert row_sub_cat.category_id == row_sub_cat_expected.category_id

            col_sub_cat = cell.get_sub_category(CellType.COLUMN_NUMBER)
            col_sub_cat_expected = cell_expected.get_sub_category(CellType.COLUMN_NUMBER)
            assert col_sub_cat.category_name == col_sub_cat_expected.category_name
            assert col_sub_cat.category_id == col_sub_cat_expected.category_id

            rs_sub_cat = cell.get_sub_category(CellType.ROW_SPAN)
            rs_sub_cat_expected = cell_expected.get_sub_category(CellType.ROW_SPAN)
            assert rs_sub_cat.category_name == rs_sub_cat_expected.category_name
            assert rs_sub_cat.category_id == rs_sub_cat_expected.category_id

            cs_sub_cat = cell.get_sub_category(CellType.COLUMN_SPAN)
            cs_sub_cat_expected = cell_expected.get_sub_category(CellType.COLUMN_SPAN)
            assert cs_sub_cat.category_name == cs_sub_cat_expected.category_name
            assert cs_sub_cat.category_id == cs_sub_cat_expected.category_id


@mark.basic
def test_tile_tables_with_items_per_table(
    dp_image_item_stretched: Image,
    row_box_tiling_table: List[BoundingBox],
    col_box_tiling_table: List[BoundingBox],
    row_sub_cats: List[CategoryAnnotation],
    col_sub_cats: List[CategoryAnnotation],
) -> None:
    """test tile_tables_with_items_per_table"""

    # Arrange
    dp = dp_image_item_stretched
    rows = dp.get_annotation(category_names=LayoutType.ROW)
    cols = dp.get_annotation(category_names=LayoutType.COLUMN)

    for row, col, row_sub_cat, col_sub_cat in zip(rows, cols, row_sub_cats, col_sub_cats):
        row.dump_sub_category(CellType.ROW_NUMBER, row_sub_cat)
        col.dump_sub_category(CellType.COLUMN_NUMBER, col_sub_cat)

    table = dp.get_annotation(category_names=LayoutType.TABLE)
    item_names = [LayoutType.ROW, LayoutType.COLUMN]  # row names must be before column name

    # Act
    dp = tile_tables_with_items_per_table(dp, table[0], item_names[0])
    dp = tile_tables_with_items_per_table(dp, table[0], item_names[1])

    # Assert
    rows = dp.get_annotation(category_names=LayoutType.ROW)
    cols = dp.get_annotation(category_names=LayoutType.COLUMN)

    first_row_box = rows[0].get_bounding_box(dp.image_id)
    second_row_box = rows[1].get_bounding_box(dp.image_id)

    first_col_box = cols[0].get_bounding_box(dp.image_id)
    second_col_box = cols[1].get_bounding_box(dp.image_id)

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

        self._segment_rule = "iou"
        self._iou_threshold_rows = 0.001
        self._iou_threshold_cols = 0.001
        self._ioa_threshold_rows = 0.4
        self._ioa_threshold_cols = 0.4
        self._remove_iou_threshold_rows = 0.001
        self._remove_iou_threshold_cols = 0.001
        self._tile_table_with_items = True
        self.table_name = LayoutType.TABLE
        self.cell_names = [CellType.COLUMN_HEADER, CellType.BODY, LayoutType.CELL]
        self.item_names = [LayoutType.ROW, LayoutType.COLUMN]
        self.sub_item_names = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]

        self.tp_table_segmentation_service = TableSegmentationService(
            self._segment_rule,  # type: ignore
            self._iou_threshold_rows if self._segment_rule in ["iou"] else self._ioa_threshold_rows,
            self._iou_threshold_cols if self._segment_rule in ["iou"] else self._ioa_threshold_cols,
            self._tile_table_with_items,
            self._remove_iou_threshold_rows,
            self._remove_iou_threshold_cols,
            self.table_name,
            self.cell_names,
            self.item_names,
            self.sub_item_names,
        )

    @mark.basic
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
        cells = dp.get_annotation(category_names=self.tp_table_segmentation_service.cell_names)  # pylint: disable=W0212

        cells_expected = dp_expected.get_annotation(
            category_names=self.tp_table_segmentation_service.cell_names  # pylint: disable=W0212
        )

        assert len(cells) == len(cells_expected)

        for cell, cell_expected in zip(cells, cells_expected):
            assert cell.get_sub_category(CellType.ROW_NUMBER) == cell_expected.get_sub_category(CellType.ROW_NUMBER)
            assert cell.get_sub_category(CellType.COLUMN_NUMBER) == cell_expected.get_sub_category(
                CellType.COLUMN_NUMBER
            )
            assert cell.get_sub_category(CellType.ROW_SPAN) == cell_expected.get_sub_category(CellType.ROW_SPAN)
            assert cell.get_sub_category(CellType.COLUMN_SPAN) == cell_expected.get_sub_category(CellType.COLUMN_SPAN)


@mark.basic
def test_create_intersection_cells(dp_image_tab_cell_item: Image) -> None:
    """
    Test create_intersection_cells generates cells from intersecting rows and columns and creates
    """

    # Arrange
    dp = dp_image_tab_cell_item

    rows = dp.get_annotation(category_names=LayoutType.ROW)
    cols = dp.get_annotation(category_names=LayoutType.COLUMN)
    for idx, items in enumerate(zip(rows, cols)):
        items[0].dump_sub_category(
            CellType.ROW_NUMBER, CategoryAnnotation(category_name=CellType.ROW_NUMBER, category_id=idx + 1)
        )
        items[1].dump_sub_category(
            CellType.COLUMN_NUMBER, CategoryAnnotation(category_name=CellType.COLUMN_NUMBER, category_id=idx + 1)
        )

    table = dp.get_annotation(category_names=LayoutType.TABLE)[0]
    table_ann_id = table.annotation_id
    detect_result_cells, segment_result_cells = create_intersection_cells(
        rows, cols, table_ann_id, [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]
    )
    expected_detect_result = [
        DetectionResult(box=[15.0, 100.0, 20.0, 150.0], class_id=None, class_name=LayoutType.CELL),
        DetectionResult(box=[40.0, 100.0, 50.0, 150.0], class_id=None, class_name=LayoutType.CELL),
        DetectionResult(box=[15.0, 200.0, 20.0, 240.0], class_id=None, class_name=LayoutType.CELL),
        DetectionResult(box=[40.0, 200.0, 50.0, 240.0], class_id=None, class_name=LayoutType.CELL),
    ]
    expected_segment_result = [
        SegmentationResult(row_num=1, col_num=1, rs=1, cs=1, annotation_id=""),
        SegmentationResult(row_num=1, col_num=2, rs=1, cs=1, annotation_id=""),
        SegmentationResult(row_num=2, col_num=1, rs=1, cs=1, annotation_id=""),
        SegmentationResult(row_num=2, col_num=2, rs=1, cs=1, annotation_id=""),
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

        self._ioa_threshold_rows = 0.4
        self._ioa_threshold_cols = 0.4
        self._remove_iou_threshold_rows = 0.001
        self._remove_iou_threshold_cols = 0.001
        self._tile_table_with_items = True
        self.table_name = LayoutType.TABLE
        self.cell_names: Sequence[Union[LayoutType, CellType]] = [
            CellType.SPANNING,
            CellType.ROW_HEADER,
            CellType.COLUMN_HEADER,
            CellType.PROJECTED_ROW_HEADER,
            LayoutType.CELL,
        ]
        self.spanning_cell_names = [
            CellType.SPANNING,
            CellType.ROW_HEADER,
            CellType.COLUMN_HEADER,
            CellType.PROJECTED_ROW_HEADER,
        ]
        self.item_names = [LayoutType.ROW, LayoutType.COLUMN]
        self.sub_item_names = [CellType.ROW_NUMBER, CellType.COLUMN_NUMBER]
        self.item_header_cell_names = [CellType.ROW_HEADER, CellType.COLUMN_HEADER]
        self.item_header_thresholds = [0.001, 0.001]

        self.table_segmentation_service = PubtablesSegmentationService(
            "ioa",
            self._ioa_threshold_rows,
            self._ioa_threshold_cols,
            self._tile_table_with_items,
            self._remove_iou_threshold_rows,
            self._remove_iou_threshold_cols,
            self.table_name,
            self.cell_names,
            self.spanning_cell_names,
            self.item_names,
            self.sub_item_names,
            self.item_header_cell_names,
            self.item_header_thresholds,
        )

    @mark.basic
    def test_pass_datapoint(self, dp_image_tab_cell_item: Image) -> None:
        """test pass_datapoint"""

        # Arrange
        dp = dp_image_tab_cell_item
        cells_ann_ids = [ann.annotation_id for ann in dp.get_annotation(category_names=LayoutType.CELL)]
        table = dp.get_annotation(category_names=LayoutType.TABLE)[0]

        dp.remove(annotation_ids=cells_ann_ids)

        tab_cells_ann_ids = [
            ann.annotation_id for ann in table.image.get_annotation(category_names=LayoutType.CELL)  # type: ignore
        ]

        table.image.remove(annotation_ids=tab_cells_ann_ids)  # type: ignore

        # Act
        dp = self.table_segmentation_service.pass_datapoint(dp)

        # Assert
        rows = dp.get_annotation(category_names=LayoutType.ROW)
        cols = dp.get_annotation(category_names=LayoutType.COLUMN)
        cells = dp.get_annotation(category_names=LayoutType.CELL)

        assert len(rows) == 2
        assert len(cols) == 2
        assert len(cells) == 4
