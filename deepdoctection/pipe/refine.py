# -*- coding: utf-8 -*-
# File: refine.py

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
Refining methods for table segmentation. The refining methods lead ultimately to a table structure which enables
HTML table representations.
"""
from __future__ import annotations

from collections import defaultdict
from copy import copy
from dataclasses import asdict
from itertools import chain, product
from typing import DefaultDict, Optional, Sequence, Union

import networkx as nx  # type: ignore

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import merge_boxes
from ..datapoint.image import Image, MetaAnnotation
from ..extern.base import DetectionResult
from ..mapper.maputils import MappingContextManager
from ..utils.error import ImageError
from ..utils.settings import CellType, LayoutType, ObjectTypes, Relationships, TableType, get_type
from .base import PipelineComponent
from .registry import pipeline_component_registry

__all__ = ["TableSegmentationRefinementService", "generate_html_string"]


def tiles_to_cells(dp: Image, table: ImageAnnotation) -> list[tuple[tuple[int, int], str]]:
    """
    Creates a table parquet by dividing a table into a tile parquet with the number of rows x number of columns tiles.
    Each tile is assigned a list of cell ids that are occupied by the cell. No cells but one or more cells can be
    assigned per tile.

    Args:
        dp: `Image`
        table: `ImageAnnotation`

    Returns:
        A list of tuples with tile positions and cell annotation ids.
    """

    cell_ann_ids = table.get_relationship(Relationships.CHILD)
    cells = dp.get_annotation(
        category_names=[LayoutType.CELL, CellType.HEADER, CellType.BODY], annotation_ids=cell_ann_ids
    )
    tile_to_cells = []

    for cell in cells:
        row_number = cell.get_sub_category(CellType.ROW_NUMBER).category_id
        col_number = cell.get_sub_category(CellType.COLUMN_NUMBER).category_id
        rs = cell.get_sub_category(CellType.ROW_SPAN).category_id
        cs = cell.get_sub_category(CellType.COLUMN_SPAN).category_id
        for k in range(rs):
            for l in range(cs):
                assert cell.annotation_id is not None, cell.annotation_id
                tile_to_cells.append(((row_number + k, col_number + l), cell.annotation_id))

    return tile_to_cells


def connected_component_tiles(
    tile_to_cell_list: list[tuple[tuple[int, int], str]]
) -> tuple[list[set[tuple[int, int]]], DefaultDict[tuple[int, int], list[str]]]:
    """
    Assigns bricks to their cell occupancy, inducing a graph with bricks as nodes and cell edges. Cells that lie on
    top of several bricks connect the underlying bricks. The graph generated is usually multiple connected. Determines
    the related components and the tile/cell ids assignment.

    Args:
        tile_to_cell_list: List of tuples with tile position and cell ids.

    Returns:
        A tuple containing a list of sets with tiles that belong to the same connected component and a dict with tiles
        as keys and assigned list of cell ids as values.
    """
    cell_to_tile_list = [(cell_position[1], cell_position[0]) for cell_position in tile_to_cell_list]
    cells = set(tup[0] for tup in cell_to_tile_list)

    tile_to_cell_dict = defaultdict(list)
    for key, val in tile_to_cell_list:
        tile_to_cell_dict[key].append(val)

    cell_to_tile_dict = defaultdict(list)
    for key, val in cell_to_tile_list:  # type: ignore
        cell_to_tile_dict[key].append(val)

    cell_to_edges = defaultdict(list)
    for key, _ in tile_to_cell_dict.items():
        cell_to_edges[key].extend(
            list(set(product(tile_to_cell_dict[key], tile_to_cell_dict[key])))  # pylint: disable=R1733
        )
    graph = nx.Graph()
    graph.add_nodes_from(cells)
    graph.add_edges_from(chain(*cell_to_edges.values()))
    connected_components_cell = list(nx.connected_components(graph))
    connected_components_tiles = []

    for component in connected_components_cell:
        tiles: set[tuple[int, int]] = set()
        for cell in component:
            tiles = tiles.union(set(cell_to_tile_dict[cell]))  # type: ignore
        connected_components_tiles.append(tiles)

    return connected_components_tiles, tile_to_cell_dict


def _missing_tile(inputs: set[tuple[int, int]]) -> Optional[tuple[int, int]]:
    min_x, min_y, max_x, max_y = (
        min(a[0] for a in inputs),
        min(a[1] for a in inputs),
        max(a[0] for a in inputs),
        max(a[1] for a in inputs),
    )

    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            if (x, y) not in inputs:
                return (x, y)
    return None


def _find_component(
    tile: tuple[int, int], reduced_connected_tiles: list[set[tuple[int, int]]]
) -> Optional[set[tuple[int, int]]]:
    for comp in reduced_connected_tiles:
        if tile in comp:
            return comp
    return None


def _merge_components(reduced_connected_tiles: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    new_reduced_connected_tiles = []
    for connected_tile in reduced_connected_tiles:
        out = _missing_tile(connected_tile)
        if out is not None:
            component = _find_component(out, reduced_connected_tiles)
            if component is not None:
                new_connected_tile = connected_tile.union(component)
                reduced_connected_tiles.remove(component)
                reduced_connected_tiles.remove(connected_tile)
                new_reduced_connected_tiles.append(new_connected_tile)
                if component in new_reduced_connected_tiles:
                    new_reduced_connected_tiles.remove(component)
            else:
                new_connected_tile = connected_tile.union({out})
                new_reduced_connected_tiles.append(new_connected_tile)
        else:
            new_reduced_connected_tiles.append(connected_tile)

    return new_reduced_connected_tiles


def generate_rectangle_tiling(connected_components_tiles: list[set[tuple[int, int]]]) -> list[set[tuple[int, int]]]:
    """
    Combines connected components so that all cells above them form a rectangular scheme. Ensures that all tiles are
    combined in such a way that all cells above them combine to form a rectangular tiling.

    Args:
        connected_components_tiles: List of sets with tiles that belong to the same connected component.

    Returns:
        List of sets with tiles, the cells on top of which together form a rectangular scheme.
    """
    rectangle_tiling: list[set[tuple[int, int]]] = []
    inputs = connected_components_tiles

    while rectangle_tiling != inputs:
        if rectangle_tiling:
            inputs = rectangle_tiling
        rectangle_tiling = _merge_components(copy(inputs))

    return rectangle_tiling


def rectangle_cells(
    rectangle_tiling: list[set[tuple[int, int]]], tile_to_cell_dict: DefaultDict[tuple[int, int], list[str]]
) -> list[set[str]]:
    """
    Determines all cells that are located above combined connected components and form a rectangular scheme.

    Args:
        rectangle_tiling: List of sets with tiles, the cells on top of which together form a rectangular scheme.
        tile_to_cell_dict: Dict with tiles as keys and assigned list of cell ids as values.

    Returns:
        List of sets of cell ids that form a rectangular scheme.
    """
    rectangle_tiling_cells: list[set[str]] = []
    for rect_tiling_component in rectangle_tiling:
        rect_cell_component: set[str] = set()
        for el in rect_tiling_component:
            rect_cell_component = rect_cell_component.union(set(tile_to_cell_dict[el]))
        rectangle_tiling_cells.append(rect_cell_component)
    return rectangle_tiling_cells


def _tiling_to_cell_position(inputs: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    row_number = min(a[0] for a in inputs)
    col_number = min(a[1] for a in inputs)
    row_span = max(abs(a[0] - b[0]) + 1 for a in inputs for b in inputs)
    col_span = max(abs(a[1] - b[1]) + 1 for a in inputs for b in inputs)
    return int(row_number), int(col_number), int(row_span), int(col_span)


def _html_cell(
    cell_position: Union[tuple[int, int, int, int], tuple[()]], position_filled_list: list[tuple[int, int]]
) -> list[str]:
    """
    Generates an HTML table cell string.

    Args:
        cell_position: Cell position tuple or empty tuple.
        position_filled_list: List of filled positions.

    Returns:
        List of HTML strings representing the cell.
    """
    html = ["<td"]
    if not cell_position:
        pass
    else:
        if cell_position[2] != 1:
            html.append(f" rowspan={cell_position[2]}")
        if cell_position[3] != 1:
            html.append(f" colspan={cell_position[3]}")
        if cell_position[2] != 1 or cell_position[3] != 1:
            position_filled_list.extend(
                [
                    (cell_position[0] + h_shift, cell_position[1] + v_shift)
                    for h_shift in range(cell_position[2])
                    for v_shift in range(cell_position[3])
                ]
            )
    html.append(">")
    str_html = "".join(html)
    html_list = [str_html, "</td>"]
    return html_list


def _html_row(
    row_list: list[tuple[int, int, int, int]],
    position_filled_list: list[tuple[int, int]],
    this_row: int,
    number_of_cols: int,
    row_ann_id_list: list[str],
) -> list[str]:
    """
    Generates an HTML table row string.

    Args:
        row_list: List of cell position tuples for the row.
        position_filled_list: List of filled positions.
        this_row: The current row number.
        number_of_cols: The total number of columns.
        row_ann_id_list: List of annotation ids for the row.

    Returns:
        List of HTML strings representing the row.
    """
    html = ["<tr>"]
    for idx in range(1, number_of_cols + 1):
        position_filled_this_row = list(filter(lambda x: x[0] == this_row, position_filled_list))
        column_filled_this_row = list(zip(*position_filled_this_row))
        column_filled_this_row = (
            [column_filled_this_row, column_filled_this_row]  # type:ignore
            if not column_filled_this_row
            else column_filled_this_row
        )
        if idx in column_filled_this_row[1]:
            pass
        else:
            cell_position_list = list(filter(lambda x: x[1] == idx, row_list))  # pylint:disable=W0640

            if cell_position_list:
                cell_position = cell_position_list[0]
                cell_id = row_ann_id_list.pop(0)
                ret_html = _html_cell(cell_position, position_filled_list)
                ret_html.insert(1, cell_id)
            else:
                cell_position = ()  # type: ignore
                ret_html = _html_cell(cell_position, position_filled_list)
            html.extend(ret_html)
    html.append("</tr>")
    return html


def _html_table(
    table_list: list[tuple[int, list[tuple[int, int, int, int]]]],
    cells_ann_list: list[tuple[int, list[str]]],
    number_of_rows: int,
    number_of_cols: int,
) -> list[str]:
    """
    Generates an HTML table string.

    Args:
        table_list: List of tuples with row number and list of cell position tuples.
        cells_ann_list: List of tuples with row number and list of annotation ids.
        number_of_rows: The total number of rows.
        number_of_cols: The total number of columns.

    Returns:
        List of HTML strings representing the table.
    """
    html = ["<table>"]
    position_filled: list[tuple[int, int]] = []
    for idx in range(1, number_of_rows + 1):
        row_idx = list(filter(lambda x: x[0] == idx, table_list))[0][1]  # pylint:disable=W0640
        row_ann_ids = list(filter(lambda x: x[0] == idx, cells_ann_list))[0][1]  # pylint:disable=W0640
        ret_html = _html_row(row_idx, position_filled, idx, number_of_cols, row_ann_ids)
        html.extend(ret_html)
    html.append("</table>")
    return html


def generate_html_string(table: ImageAnnotation, cell_names: Sequence[ObjectTypes]) -> list[str]:
    """
    Generates an HTML representation of a table using table segmentation by row number, column number, etc.

    Note:
        It must be ensured that all cells have a row number, column number, row span, and column span, and that the
        dissection by rows and columns is completely covered by cells.

    Args:
        table: An annotation that has a not None image and fully segmented cell annotation.
        cell_names: List of cell names that are used for the table segmentation.

    Returns:
        HTML representation of the table.

    Raises:
        `ImageError`: If `table.image` is None.
    """
    if table.image is None:
        raise ImageError("table.image cannot be None")
    table_image = table.image
    cells = table_image.get_annotation(category_names=cell_names)
    number_of_rows = table_image.summary.get_sub_category(TableType.NUMBER_OF_ROWS).category_id
    number_of_cols = table_image.summary.get_sub_category(TableType.NUMBER_OF_COLUMNS).category_id
    table_list = []
    cells_ann_list = []
    for row_number in range(1, number_of_rows + 1):
        cells_of_row = list(
            sorted(
                filter(
                    lambda cell: cell.get_sub_category(CellType.ROW_NUMBER).category_id
                    == row_number,  # pylint: disable=W0640
                    cells,
                ),
                key=lambda cell: cell.get_sub_category(CellType.COLUMN_NUMBER).category_id,
            )
        )
        row_list = [
            (
                cell.get_sub_category(CellType.ROW_NUMBER).category_id,
                cell.get_sub_category(CellType.COLUMN_NUMBER).category_id,
                cell.get_sub_category(CellType.ROW_SPAN).category_id,
                cell.get_sub_category(CellType.COLUMN_SPAN).category_id,
            )
            for cell in cells_of_row
        ]
        ann_list = [cell.annotation_id for cell in cells_of_row]
        table_list.append((row_number, row_list))
        cells_ann_list.append((row_number, ann_list))
    return _html_table(table_list, cells_ann_list, number_of_rows, number_of_cols)


@pipeline_component_registry.register("TableSegmentationRefinementService")
class TableSegmentationRefinementService(PipelineComponent):
    """
    Refinement of the cell segmentation. The aim of this component is to create a table structure so that an HTML
    structure can be created.

    Assume that the arrangement of cells, rows and in the table is as follows in the original state. There is only one
    column.

    +----------+
    | C1   C2  |
    +----------+
    | C3   C3  |
    +----------+

    The first two cells have the same column assignment via the segmentation and must therefore be merged. Note that
    the number of rows and columns does not change in the refinement process. What changes is just the number of cells.

    Furthermore, when merging, it must be ensured that the combined cells still have a rectangular shape. This is also
    guaranteed in the refining process.

    +----------+
    | C1 |  C2 |
    +          +
    | C3 |  C3 |
    +----------+

    The table consists of one row and two columns. The upper cells belong together with the lower cell. However, this
    means that all cells must be merged with one another so that the table only consists of one cell after the
    refinement process.

    Example:
        ```python
        layout = ImageLayoutService(layout_detector, to_image=True, crop_image=True)
        cell = SubImageLayoutService(cell_detector, "TABLE")
        row_col = SubImageLayoutService(row_col_detector, "TABLE")
        table_segmentation = TableSegmentationService("ioa",0.9,0.8,True,0.0001,0.0001)
        table_segmentation_refinement = TableSegmentationRefinementService()

        table_recognition_pipe = DoctectionPipe([layout,
                                                 cell,
                                                 row_col,
                                                 table_segmentation,
                                                 table_segmentation_refinement])
        df = pipe.analyze(path="path/to/document.pdf")

        for dp in df:
            ...
        ```
    """

    def __init__(self, table_names: Sequence[ObjectTypes], cell_names: Sequence[ObjectTypes]) -> None:
        """
        Initializes the `TableSegmentationRefinementService`.

        Args:
            table_names: Sequence of table object types.
            cell_names: Sequence of cell object types.
        """
        self.table_name = table_names
        self.cell_names = cell_names
        super().__init__("table_segment_refine")

    def serve(self, dp: Image) -> None:
        tables = dp.get_annotation(category_names=self.table_name)
        for table in tables:
            if table.image is None:
                raise ImageError("table.image cannot be None")
            tiles_to_cells_list = tiles_to_cells(dp, table)
            connected_components, tile_to_cell_dict = connected_component_tiles(tiles_to_cells_list)
            rectangle_tiling = generate_rectangle_tiling(connected_components)
            rectangle_cells_list = rectangle_cells(rectangle_tiling, tile_to_cell_dict)
            for tiling, cells_to_merge in zip(rectangle_tiling, rectangle_cells_list):
                no_context_error = True
                if len(cells_to_merge) != 1:
                    cells = dp.get_annotation(annotation_ids=list(cells_to_merge))
                    cell_boxes = [cell.get_bounding_box(table.image.image_id) for cell in cells]
                    merged_box = merge_boxes(*cell_boxes)
                    det_result = DetectionResult(
                        box=merged_box.to_list(mode="xyxy"),
                        score=-1.0,
                        class_id=cells[0].category_id,
                        class_name=get_type(cells[0].category_name),
                    )
                    new_cell_ann_id = self.dp_manager.set_image_annotation(det_result, table.annotation_id)
                    if new_cell_ann_id is not None:
                        row_number, col_number, row_span, col_span = _tiling_to_cell_position(tiling)
                        self.dp_manager.set_category_annotation(
                            CellType.ROW_NUMBER, row_number, CellType.ROW_NUMBER, new_cell_ann_id
                        )
                        self.dp_manager.set_category_annotation(
                            CellType.COLUMN_NUMBER, col_number, CellType.COLUMN_NUMBER, new_cell_ann_id
                        )
                        self.dp_manager.set_category_annotation(
                            CellType.ROW_SPAN, row_span, CellType.ROW_SPAN, new_cell_ann_id
                        )
                        self.dp_manager.set_category_annotation(
                            CellType.COLUMN_SPAN, col_span, CellType.COLUMN_SPAN, new_cell_ann_id
                        )
                    else:
                        # DetectionResult cannot be dumped, hence merged_box must already exist. Hence, it must
                        # contain all other boxes. Hence, we must deactivate all other boxes.
                        with MappingContextManager(
                            dp_name=dp.file_name, filter_level="annotation", detect_result=asdict(det_result)
                        ) as annotation_context:
                            box_index = cell_boxes.index(merged_box)
                            cells.pop(box_index)
                        no_context_error = not annotation_context.context_error
                    if no_context_error:
                        for cell in cells:
                            cell.deactivate()

            cells = table.image.get_annotation(category_names=self.cell_names)
            number_of_rows = max(cell.get_sub_category(CellType.ROW_NUMBER).category_id for cell in cells)
            number_of_cols = max(cell.get_sub_category(CellType.COLUMN_NUMBER).category_id for cell in cells)
            max_row_span = max(cell.get_sub_category(CellType.ROW_SPAN).category_id for cell in cells)
            max_col_span = max(cell.get_sub_category(CellType.COLUMN_SPAN).category_id for cell in cells)
            # TODO: the summaries should be sub categories of the underlying ann
            if (
                TableType.NUMBER_OF_ROWS in table.image.summary.sub_categories
                and TableType.NUMBER_OF_COLUMNS in table.image.summary.sub_categories
                and TableType.MAX_ROW_SPAN in table.image.summary.sub_categories
                and TableType.MAX_COL_SPAN in table.image.summary.sub_categories
            ):
                table.image.summary.remove_sub_category(TableType.NUMBER_OF_ROWS)
                table.image.summary.remove_sub_category(TableType.NUMBER_OF_COLUMNS)
                table.image.summary.remove_sub_category(TableType.MAX_ROW_SPAN)
                table.image.summary.remove_sub_category(TableType.MAX_COL_SPAN)

            self.dp_manager.set_summary_annotation(
                TableType.NUMBER_OF_ROWS, TableType.NUMBER_OF_ROWS, number_of_rows, annotation_id=table.annotation_id
            )
            self.dp_manager.set_summary_annotation(
                TableType.NUMBER_OF_COLUMNS,
                TableType.NUMBER_OF_COLUMNS,
                number_of_cols,
                annotation_id=table.annotation_id,
            )
            self.dp_manager.set_summary_annotation(
                TableType.MAX_ROW_SPAN, TableType.MAX_ROW_SPAN, max_row_span, annotation_id=table.annotation_id
            )
            self.dp_manager.set_summary_annotation(
                TableType.MAX_COL_SPAN, TableType.MAX_COL_SPAN, max_col_span, annotation_id=table.annotation_id
            )
            html = generate_html_string(table, self.cell_names)
            self.dp_manager.set_container_annotation(TableType.HTML, -1, TableType.HTML, table.annotation_id, html)

    def clone(self) -> TableSegmentationRefinementService:
        return self.__class__(self.table_name, self.cell_names)

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(
            image_annotations=(),
            sub_categories={
                LayoutType.CELL: {
                    CellType.ROW_NUMBER: {CellType.ROW_NUMBER},
                    CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER},
                    CellType.ROW_SPAN: {CellType.ROW_SPAN},
                    CellType.COLUMN_SPAN: {CellType.COLUMN_SPAN},
                },
                LayoutType.TABLE: {TableType.HTML: {TableType.HTML}},
            },
            relationships={},
            summaries=(),
        )

    def clear_predictor(self) -> None:
        pass
