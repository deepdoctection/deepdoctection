# -*- coding: utf-8 -*-
# File: test_refine.py

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
Testing module pipe.refine
"""

from typing import List, Set, Tuple

from pytest import mark

from deepdoctection.datapoint import ContainerAnnotation, Image
from deepdoctection.pipe.refine import (
    TableSegmentationRefinementService,
    _html_table,
    connected_component_tiles,
    generate_rectangle_tiling,
    rectangle_cells,
)

# from deepdoctection.utils.settings import names
from deepdoctection.utils.settings import CellType, LayoutType, TableType


@mark.basic
@mark.parametrize(
    "tiles_to_cells,expected_rectangle_cells_list",
    [
        (
            [
                ((1, 1), "a"),
                ((1, 2), "b"),
                ((1, 3), "b"),
                ((2, 1), "c"),
                ((2, 2), "d"),
                ((2, 3), "e"),
                ((3, 1), "f"),
                ((3, 2), "f"),
                ((3, 3), "f"),
            ],
            [{"d"}, {"e"}, {"f"}, {"b"}, {"a"}, {"c"}],
        ),
        (
            [
                ((1, 1), "a"),
                ((1, 2), "b"),
                ((1, 3), "b"),
                ((2, 1), "c"),
                ((2, 2), "d"),
                ((2, 3), "e"),
                ((2, 1), "f"),
                ((2, 2), "f"),
                ((2, 3), "f"),
            ],
            [{"a"}, {"f", "c", "d", "e"}, {"b"}],
        ),
        (
            [
                ((1, 1), "a"),
                ((1, 2), "b"),
                ((1, 2), "c"),
                ((2, 1), "d"),
                ((2, 2), "e"),
                ((2, 2), "c"),
                ((2, 1), "f"),
                ((2, 2), "f"),
                ((2, 2), "f"),
            ],
            [{"a", "c", "d", "e", "b", "f"}],
        ),
    ],
)
def test_rectangle_cell_tiling(
    tiles_to_cells: List[Tuple[Tuple[int, int], str]], expected_rectangle_cells_list: List[Set[str]]
) -> None:
    """
    Test that from a list of cell tiling one correctly receives a list of sets of cells that must be merged
    in order to have only rectangular shaped cells
    """
    # Act
    connected_components, tile_to_cell_dict = connected_component_tiles(tiles_to_cells)
    rectangle_tiling = generate_rectangle_tiling(connected_components)
    rectangle_cells_list = rectangle_cells(rectangle_tiling, tile_to_cell_dict)

    # Assert
    for el in expected_rectangle_cells_list:
        assert el in rectangle_cells_list

    for el in rectangle_cells_list:
        assert el in expected_rectangle_cells_list


@mark.basic
@mark.parametrize(
    "table_list,cells_ann_list,number_of_rows,number_of_cols,expected_html_list",
    [
        (
            [
                (1, [(1, 1, 1, 1), (1, 2, 1, 2)]),
                (2, [(2, 1, 2, 1), (2, 2, 1, 1), (2, 3, 1, 1)]),
                (3, [(3, 2, 1, 1), (3, 3, 1, 1)]),
                (4, [(4, 1, 1, 1), (4, 2, 1, 2)]),
            ],
            [(1, ["a", "b"]), (2, ["c", "d", "e"]), (3, ["f", "g"]), (4, ["h", "i"])],
            4,
            3,
            [
                "<table>",
                "<tr>",
                "<td>",
                "a",
                "</td>",
                "<td colspan=2>",
                "b",
                "</td>",
                "</tr>",
                "<tr>",
                "<td rowspan=2>",
                "c",
                "</td>",
                "<td>",
                "d",
                "</td>",
                "<td>",
                "e",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "f",
                "</td>",
                "<td>",
                "g",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "h",
                "</td>",
                "<td colspan=2>",
                "i",
                "</td>",
                "</tr>",
                "</table>",
            ],
        ),
        (
            [(1, [(1, 1, 3, 1), (1, 2, 1, 2)]), (2, [(2, 2, 1, 1), (2, 3, 1, 1)]), (3, [(3, 2, 1, 2)])],
            [(1, ["a", "b"]), (2, ["c", "d"]), (3, ["e"])],
            3,
            3,
            [
                "<table>",
                "<tr>",
                "<td rowspan=3>",
                "a",
                "</td>",
                "<td colspan=2>",
                "b",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "c",
                "</td>",
                "<td>",
                "d",
                "</td>",
                "</tr>",
                "<tr>",
                "<td colspan=2>",
                "e",
                "</td>",
                "</tr>",
                "</table>",
            ],
        ),
        (
            [(1, [(1, 2, 1, 1), (1, 3, 1, 2)]), (2, [(2, 3, 2, 1)]), (3, [(3, 1, 2, 2)]), (4, [(4, 4, 1, 1)])],
            [(1, ["a", "b"]), (2, ["c"]), (3, ["d"]), (4, ["e"])],
            4,
            4,
            [
                "<table>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "a",
                "</td>",
                "<td colspan=2>",
                "b",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "</td>",
                "<td rowspan=2>",
                "c",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td rowspan=2 colspan=2>",
                "d",
                "</td>",
                "<td>",
                "</td>",
                "</tr>",
                "<tr>",
                "<td>",
                "</td>",
                "<td>",
                "e",
                "</td>",
                "</tr>",
                "</table>",
            ],
        ),
    ],
)
def test_generate_html_string(
    table_list: List[Tuple[int, List[Tuple[int, int, int, int]]]],
    cells_ann_list: List[Tuple[int, List[str]]],
    number_of_rows: int,
    number_of_cols: int,
    expected_html_list: List[str],
) -> None:
    """
    test _html_table
    """
    # Act
    html_list = _html_table(table_list, cells_ann_list, number_of_rows, number_of_cols)

    # Assert
    assert html_list == expected_html_list


class TestTableSegmentationRefinementService:
    """
    Test TableSegmentationRefinementService (only integrated)
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.table_segmentation_refinement_service = TableSegmentationRefinementService()

    @mark.basic
    def test_integration_pipeline_component(self, dp_image_fully_segmented_fully_tiled: Image) -> None:
        """
        Integration test for pipeline component
        """

        # Arrange
        dp = dp_image_fully_segmented_fully_tiled

        # Act
        dp = self.table_segmentation_refinement_service.pass_datapoint(dp)

        # Assert
        table = dp.get_annotation(category_names=LayoutType.table)[0]
        assert table.image is not None
        summary = table.image.summary
        summaries_table = [
            summary.get_sub_category(TableType.number_of_rows).category_id,
            summary.get_sub_category(TableType.number_of_columns).category_id,
            summary.get_sub_category(TableType.max_row_span).category_id,
            summary.get_sub_category(TableType.max_col_span).category_id,
        ]
        summary_html = table.get_sub_category(TableType.html)
        cells = dp.get_annotation(
            category_names=self.table_segmentation_refinement_service._cell_names  # pylint: disable=W0212
        )
        row_numbers = {cell.get_sub_category(CellType.row_number).category_id for cell in cells}
        col_numbers = {cell.get_sub_category(CellType.column_number).category_id for cell in cells}
        row_spans = {cell.get_sub_category(CellType.row_span).category_id for cell in cells}
        col_spans = {cell.get_sub_category(CellType.column_span).category_id for cell in cells}

        assert len(cells) == 4
        assert row_numbers == {"1", "2"}
        assert col_numbers == {"1", "2"}
        assert row_spans == {"1", "1", "1", "1"}
        assert col_spans == {"1", "1", "1", "1"}
        assert summaries_table == ["2", "2", "1", "1"]
        assert isinstance(summary_html, ContainerAnnotation)
        assert summary_html.value == [
            "<table>",
            "<tr>",
            "<td>",
            "2c1458bf-6b21-327a-9632-21373da468bb",
            "</td>",
            "<td>",
            "609243a2-0bd8-3002-939a-c8bcf7b2e3c8",
            "</td>",
            "</tr>",
            "<tr>",
            "<td>",
            "9b2b0bd1-3885-3aa6-8d45-ebebdd1abddd",
            "</td>",
            "<td>",
            "6e70cf3d-e260-3f73-81bf-77616b6d54b2",
            "</td>",
            "</tr>",
            "</table>",
        ]
