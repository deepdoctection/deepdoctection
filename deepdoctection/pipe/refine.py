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
Module for refining methods of table segmentation. The refining methods lead ultimately to a table structure which
enables html table representations
"""

from collections import defaultdict
from copy import copy
from itertools import chain, product
from typing import DefaultDict, List, Optional, Set, Tuple, Union

import networkx as nx  # type: ignore

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import merge_boxes
from ..datapoint.image import Image
from ..extern.base import DetectionResult
from ..utils.settings import names
from .base import PipelineComponent

__all__ = ["TableSegmentationRefinementService"]


def tiles_to_cells(dp: Image, table: ImageAnnotation) -> List[Tuple[Tuple[int, int], str]]:
    """
    Creation of a table parquet: A table is divided into a tile parquet with the (number of rows) x
    (the number of columns) tiles.
    Each tile is assigned a list of cell ids that are occupied by the cell. No cells but one or more cells can be
    assigned per tile.

    :param dp: Image
    :param table: Table image annotation
    :return: Image
    """

    cell_ann_ids = table.get_relationship(names.C.CHILD)
    cells = dp.get_annotation(category_names=[names.C.CELL, names.C.HEAD, names.C.BODY], annotation_ids=cell_ann_ids)
    tile_to_cells = []

    for cell in cells:
        row_number = int(cell.get_sub_category(names.C.RN).category_id)
        col_number = int(cell.get_sub_category(names.C.CN).category_id)
        rs = int(cell.get_sub_category(names.C.RS).category_id)
        cs = int(cell.get_sub_category(names.C.CS).category_id)
        for k in range(rs):
            for l in range(cs):
                assert cell.annotation_id is not None
                tile_to_cells.append(((row_number + k, col_number + l), cell.annotation_id))

    return tile_to_cells


def connected_component_tiles(
    tile_to_cell_list: List[Tuple[Tuple[int, int], str]]
) -> Tuple[List[Set[Tuple[int, int]]], DefaultDict[Tuple[int, int], List[str]]]:
    """
    The assignment of bricks to their cell occupancy induces a graph, with bricks as corners and cell edges. Cells that
    lie on top of several bricks connect the underlying bricks. The graph generated according to this procedure is
    usually multiple connected. The related components and the tile/cell ids assignment are determined.

    :param tile_to_cell_list: List of tuples with tile position and cell ids
    :return: List of set with tiles that belong to the same connected component and a dict with tiles as keys and
             assigned list of cell ids as values.
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
        tiles: Set[Tuple[int, int]] = set()
        for cell in component:
            tiles = tiles.union(set(cell_to_tile_dict[cell]))  # type: ignore
        connected_components_tiles.append(tiles)

    return connected_components_tiles, tile_to_cell_dict


def _missing_tile(inputs: Set[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
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
    tile: Tuple[int, int], reduced_connected_tiles: List[Set[Tuple[int, int]]]
) -> Optional[Set[Tuple[int, int]]]:
    for comp in reduced_connected_tiles:
        if tile in comp:
            return comp
    return None


def _merge_components(reduced_connected_tiles: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
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


def generate_rectangle_tiling(connected_components_tiles: List[Set[Tuple[int, int]]]) -> List[Set[Tuple[int, int]]]:
    """
    The determined connected components imply that all cells have to be combined which are above a connected component.
    In addition, however, it must also be taken into account that cells must be rectangular. This means that related
    components have to be combined whose combined cells above do not create a rectangular tiling. All tiles are combined
    in such a way that all cells above them combine to form a rectangular scheme.

    :param connected_components_tiles: List of set with tiles that belong to the same connected component
    :return: List of sets with tiles, the cells on top of which together form a rectangular scheme
    """
    rectangle_tiling: List[Set[Tuple[int, int]]] = []
    inputs = connected_components_tiles

    while rectangle_tiling != inputs:
        if rectangle_tiling:
            inputs = rectangle_tiling
        rectangle_tiling = _merge_components(copy(inputs))

    return rectangle_tiling


def rectangle_cells(
    rectangle_tiling: List[Set[Tuple[int, int]]], tile_to_cell_dict: DefaultDict[Tuple[int, int], List[str]]
) -> List[Set[str]]:
    """
    All cells are determined that are located above combined connected components and form a rectangular scheme.

    :param rectangle_tiling: List of sets with tiles, the cells on top of which together form a rectangular scheme
    :param tile_to_cell_dict: Dict with tiles as keys and assigned list of cell ids as values.
    :return: List of set of cell ids that form a rectangular scheme
    """
    rectangle_tiling_cells: List[Set[str]] = []
    for rect_tiling_component in rectangle_tiling:
        rect_cell_component: Set[str] = set()
        for el in rect_tiling_component:
            rect_cell_component = rect_cell_component.union(set(tile_to_cell_dict[el]))
        rectangle_tiling_cells.append(rect_cell_component)
    return rectangle_tiling_cells


def _tiling_to_cell_position(inputs: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    row_number = min(a[0] for a in inputs)
    col_number = min(a[1] for a in inputs)
    row_span = max(abs(a[0] - b[0]) + 1 for a in inputs for b in inputs)
    col_span = max(abs(a[1] - b[1]) + 1 for a in inputs for b in inputs)
    return int(row_number), int(col_number), int(row_span), int(col_span)


def _html_cell(
    cell_position: Union[Tuple[int, int, int, int], Tuple[()]], position_filled_list: List[Tuple[int, int]]
) -> List[str]:
    """
    Html table cell string generation
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
    row_list: List[Tuple[int, int, int, int]],
    position_filled_list: List[Tuple[int, int]],
    this_row: int,
    number_of_cols: int,
    row_ann_id_list: List[str],
) -> List[str]:
    """
    Html table row string generation
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
    table_list: List[Tuple[int, List[Tuple[int, int, int, int]]]],
    cells_ann_list: List[Tuple[int, List[str]]],
    number_of_rows: int,
    number_of_cols: int,
) -> List[str]:
    """
    Html table string generation
    """
    html = ["<table>"]
    position_filled: List[Tuple[int, int]] = []
    for idx in range(1, number_of_rows + 1):
        row_idx = list(filter(lambda x: x[0] == idx, table_list))[0][1]  # pylint:disable=W0640
        row_ann_ids = list(filter(lambda x: x[0] == idx, cells_ann_list))[0][1]  # pylint:disable=W0640
        ret_html = _html_row(row_idx, position_filled, idx, number_of_cols, row_ann_ids)
        html.extend(ret_html)
    html.append("</table>")
    return html


def generate_html_string(table: ImageAnnotation) -> List[str]:
    """
    Takes the table segmentation by using table cells row number, column numbers etc. and generates a html
    representation.

    :param table: An annotation that has a not None image and fully segmented cell annotation.
    :return: HTML representation of the table
    """
    assert table.image is not None
    table_image = table.image
    cells = table_image.get_annotation(category_names=[names.C.CELL, names.C.CELL, names.C.BODY])
    number_of_rows = int(table_image.summary.get_sub_category(names.C.NR).category_id)
    number_of_cols = int(table_image.summary.get_sub_category(names.C.NC).category_id)
    table_list = []
    cells_ann_list = []
    for row_number in range(1, number_of_rows + 1):
        cells_of_row = list(
            sorted(
                filter(
                    lambda cell: cell.get_sub_category(names.C.RN).category_id
                    == str(row_number),  # pylint: disable=W0640
                    cells,
                ),
                key=lambda cell: cell.get_sub_category(names.C.CN).category_id,
            )
        )
        row_list = [
            (
                int(cell.get_sub_category(names.C.RN).category_id),
                int(cell.get_sub_category(names.C.CN).category_id),
                int(cell.get_sub_category(names.C.RS).category_id),
                int(cell.get_sub_category(names.C.CS).category_id),
            )
            for cell in cells_of_row
        ]
        ann_list = [cell.annotation_id for cell in cells_of_row]
        table_list.append((row_number, row_list))
        cells_ann_list.append((row_number, ann_list))
    return _html_table(table_list, cells_ann_list, number_of_rows, number_of_cols)


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

    The first two cells have the same column assignment via the segmentation and must therefore be merged.
    Note that the number of rows and columns does not change in the refinement process. What changes is just the number
    of cells.

    Furthermore, when merging, it must be ensured that the combined cells still have a rectangular shape.
    This is also guaranteed in the refining process.

    +----------+
    | C1 |  C2 |
    +          +
    | C3 |  C3 |
    +----------+

    The table consists of one row and two columns. The upper cells belong together with the lower cell.
    However, this means that all cells must be merged with one another so that the table only consists of one cell
    after the refinement process.
    """

    def __init__(self) -> None:
        super().__init__(None)
        self._table_name = names.C.TAB
        self._cell_names = [names.C.HEAD, names.C.BODY, names.C.CELL]

    def serve(self, dp: Image) -> None:
        tables = dp.get_annotation(category_names=self._table_name)
        for table in tables:
            assert table.image is not None
            tiles_to_cells_list = tiles_to_cells(dp, table)
            connected_components, tile_to_cell_dict = connected_component_tiles(tiles_to_cells_list)
            rectangle_tiling = generate_rectangle_tiling(connected_components)
            rectangle_cells_list = rectangle_cells(rectangle_tiling, tile_to_cell_dict)
            for tiling, cells_to_merge in zip(rectangle_tiling, rectangle_cells_list):
                if len(cells_to_merge) != 1:
                    cells = dp.get_annotation(annotation_ids=list(cells_to_merge))
                    merged_box = merge_boxes(
                        *[cell.image.get_embedding(table.image.image_id) for cell in cells if cell.image is not None]
                    )
                    det_result = DetectionResult(
                        box=merged_box.to_list(mode="xyxy"),
                        score=-1.0,
                        class_id=int(cells[0].category_id),
                        class_name=cells[0].category_name,
                    )
                    new_cell_ann_id = self.dp_manager.set_image_annotation(det_result, table.annotation_id)
                    if new_cell_ann_id is not None:
                        row_number, col_number, row_span, col_span = _tiling_to_cell_position(tiling)
                        self.dp_manager.set_category_annotation(names.C.RN, row_number, names.C.RN, new_cell_ann_id)
                        self.dp_manager.set_category_annotation(names.C.CN, col_number, names.C.CN, new_cell_ann_id)
                        self.dp_manager.set_category_annotation(names.C.RS, row_span, names.C.RS, new_cell_ann_id)
                        self.dp_manager.set_category_annotation(names.C.CS, col_span, names.C.CS, new_cell_ann_id)
                        for cell in cells:
                            cell.deactivate()

            cells = table.image.get_annotation(category_names=self._cell_names)
            number_of_rows = max([int(cell.get_sub_category(names.C.RN).category_id) for cell in cells])
            number_of_cols = max([int(cell.get_sub_category(names.C.CN).category_id) for cell in cells])
            max_row_span = max([int(cell.get_sub_category(names.C.RS).category_id) for cell in cells])
            max_col_span = max([int(cell.get_sub_category(names.C.CS).category_id) for cell in cells])
            self.dp_manager.set_summary_annotation(names.C.NR, number_of_rows, table.annotation_id)
            self.dp_manager.set_summary_annotation(names.C.NC, number_of_cols, table.annotation_id)
            self.dp_manager.set_summary_annotation(names.C.NRS, max_row_span, table.annotation_id)
            self.dp_manager.set_summary_annotation(names.C.NCS, max_col_span, table.annotation_id)
            html = generate_html_string(table)
            self.dp_manager.set_container_annotation(
                names.C.HTAB, -1, names.C.HTAB, table.annotation_id, html  # type: ignore
            )
