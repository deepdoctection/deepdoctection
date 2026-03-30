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
from typing import DefaultDict, Optional, Sequence

from lazy_imports import try_import

from dd_core.datapoint.annotation import AnnotationRef, ImageAnnotation, ReferencePayload
from dd_core.datapoint.box import merge_boxes
from dd_core.datapoint.image import Image, MetaAnnotation
from dd_core.mapper.maputils import MappingContextManager
from dd_core.utils.error import ImageError
from dd_core.utils.file_utils import networkx_available
from dd_core.utils.object_types import CellKey, LayoutLabel, ObjectTypes, RelationshipKey, TableKey, get_type

from ..extern.base import DetectionResult
from .base import PipelineComponent
from .registry import pipeline_component_registry

with try_import() as import_guard:
    import networkx as nx  # type: ignore


__all__ = ["TableSegmentationRefinementService", "generate_html_payload"]


def tiles_to_cells(
    dp: Image, table: ImageAnnotation, cell_names: Sequence[ObjectTypes]
) -> list[tuple[tuple[int, int], str]]:
    """
    Creates a table parquet by dividing a table into a tile parquet with the number of rows x number of columns tiles.
    Each tile is assigned a list of cell ids that are occupied by the cell. No cells but one or more cells can be
    assigned per tile.

    Args:
        dp: `Image`
        table: `ImageAnnotation`
        cell_names: List of cell names that are used for the table tiling

    Returns:
        A list of tuples with tile positions and cell annotation ids.
    """

    cell_ann_ids = table.get_relationship(RelationshipKey.CHILD)
    cells = dp.get_annotation(category_names=cell_names, annotation_ids=cell_ann_ids)
    tile_to_cells = []

    for cell in cells:
        if (
            CellKey.ROW_NUMBER in cell.sub_categories
            and CellKey.COLUMN_NUMBER in cell.sub_categories
            and CellKey.ROW_SPAN in cell.sub_categories
            and CellKey.COLUMN_SPAN in cell.sub_categories
        ):
            row_number = cell.get_sub_category(CellKey.ROW_NUMBER).category_id
            col_number = cell.get_sub_category(CellKey.COLUMN_NUMBER).category_id
            rs = cell.get_sub_category(CellKey.ROW_SPAN).category_id
            cs = cell.get_sub_category(CellKey.COLUMN_SPAN).category_id
            for k in range(rs):
                for l in range(cs):
                    assert cell.annotation_id is not None, cell.annotation_id
                    tile_to_cells.append(((row_number + k, col_number + l), cell.annotation_id))

    return tile_to_cells


def connected_component_tiles(
    tile_to_cell_list: list[tuple[tuple[int, int], str]],
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
    cell_position: tuple[int, int, int, int] | tuple[()],
    position_filled_list: list[tuple[int, int]],
) -> list[str | AnnotationRef]:
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
    html_list: list[str | AnnotationRef] = [str_html, "</td>"]
    return html_list


def _html_row(
    row_list: list[tuple[int, int, int, int]],
    position_filled_list: list[tuple[int, int]],
    this_row: int,
    number_of_cols: int,
    row_ann_id_list: list[str],
    image_id: str | None = None,
) -> list[str | AnnotationRef]:
    """
    Generates an HTML table row string.

    Args:
        row_list: List of cell position tuples for the row.
        position_filled_list: List of filled positions.
        this_row: The current row number.
        number_of_cols: The total number of columns.
        row_ann_id_list: List of annotation ids for the row.
        image_id: Image id of the table image.

    Returns:
        List of HTML strings and AnnotationRef objects representing the row.
    """
    html: list[str | AnnotationRef] = ["<tr>"]
    for idx in range(1, number_of_cols + 1):
        position_filled_this_row = list(filter(lambda x: x[0] == this_row, position_filled_list))
        column_filled_this_row = list(zip(*position_filled_this_row))
        column_filled_this_row = (
            [column_filled_this_row, column_filled_this_row]  # type: ignore
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
                ret_html.insert(1, AnnotationRef(image_id=image_id, annotation_id=cell_id))
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
    image_id: str | None = None,
) -> list[str | AnnotationRef]:
    """
    Generates an HTML table representation with unresolved AnnotationRef placeholders.

    Args:
        table_list: List of tuples with row number and list of cell position tuples.
        cells_ann_list: List of tuples with row number and list of annotation ids.
        number_of_rows: The total number of rows.
        number_of_cols: The total number of columns.
        image_id: Image id of the table image.

    Returns:
        List of HTML strings and AnnotationRef objects representing the table.
    """
    html: list[str | AnnotationRef] = ["<table>"]
    position_filled: list[tuple[int, int]] = []
    for idx in range(1, number_of_rows + 1):
        row_idx = list(filter(lambda x: x[0] == idx, table_list))[0][1]  # pylint:disable=W0640
        row_ann_ids = list(filter(lambda x: x[0] == idx, cells_ann_list))[0][1]  # pylint:disable=W0640
        ret_html = _html_row(row_idx, position_filled, idx, number_of_cols, row_ann_ids, image_id)
        html.extend(ret_html)
    html.append("</table>")
    return html


def generate_html_payload(table: ImageAnnotation, cell_names: Sequence[ObjectTypes]) -> ReferencePayload:
    """
    Generates an unresolved HTML representation of a table using AnnotationRef placeholders.

    Args:
        table: An annotation that has a not None image and fully segmented cell annotation.
        cell_names: List of cell names that are used for the table segmentation.

    Returns:
        ReferencePayload with HTML fragments and AnnotationRef placeholders.

    Raises:
        ImageError: If table.image is None.
    """
    if table.image is None:
        raise ImageError("table.image cannot be None")

    table_image = table.image
    cells = table_image.get_annotation(category_names=cell_names)
    number_of_rows = table_image.summary.get_sub_category(TableKey.NUMBER_OF_ROWS).category_id
    number_of_cols = table_image.summary.get_sub_category(TableKey.NUMBER_OF_COLUMNS).category_id

    table_list = []
    cells_ann_list = []

    for row_number in range(1, number_of_rows + 1):
        cells_of_row = list(
            sorted(
                filter(
                    lambda cell: cell.get_sub_category(CellKey.ROW_NUMBER).category_id
                    == row_number,  # pylint:disable=W0640
                    cells,
                ),
                key=lambda cell: cell.get_sub_category(CellKey.COLUMN_NUMBER).category_id,
            )
        )
        row_list = [
            (
                cell.get_sub_category(CellKey.ROW_NUMBER).category_id,
                cell.get_sub_category(CellKey.COLUMN_NUMBER).category_id,
                cell.get_sub_category(CellKey.ROW_SPAN).category_id,
                cell.get_sub_category(CellKey.COLUMN_SPAN).category_id,
            )
            for cell in cells_of_row
        ]
        ann_list = [cell.annotation_id for cell in cells_of_row]
        table_list.append((row_number, row_list))
        cells_ann_list.append((row_number, ann_list))

    html_fragments = _html_table(
        table_list=table_list,
        cells_ann_list=cells_ann_list,
        number_of_rows=number_of_rows,
        number_of_cols=number_of_cols,
        image_id=None,
    )

    return ReferencePayload(content=html_fragments)


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
        if not networkx_available():
            raise ModuleNotFoundError(
                "TableSegmentationRefinementService requires networkx. Please install separately."
            )
        self.table_name = table_names
        self.cell_names = cell_names
        super().__init__("table_segment_refine")

    def serve(self, dp: Image) -> None:
        tables = dp.get_annotation(category_names=self.table_name)
        for table in tables:
            if table.image is None:
                raise ImageError("table.image cannot be None")
            tiles_to_cells_list = tiles_to_cells(dp, table, self.cell_names)
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
                        class_id=cells[0].category_id,
                        class_name=get_type(cells[0].category_name),
                        absolute_coords=False,
                    )
                    new_cell_ann_id = self.dp_manager.set_image_annotation(det_result, table.annotation_id)
                    if new_cell_ann_id is not None:
                        row_number, col_number, row_span, col_span = _tiling_to_cell_position(tiling)
                        self.dp_manager.set_category_annotation(
                            CellKey.ROW_NUMBER, row_number, CellKey.ROW_NUMBER, new_cell_ann_id
                        )
                        self.dp_manager.set_category_annotation(
                            CellKey.COLUMN_NUMBER, col_number, CellKey.COLUMN_NUMBER, new_cell_ann_id
                        )
                        self.dp_manager.set_category_annotation(
                            CellKey.ROW_SPAN, row_span, CellKey.ROW_SPAN, new_cell_ann_id
                        )
                        self.dp_manager.set_category_annotation(
                            CellKey.COLUMN_SPAN, col_span, CellKey.COLUMN_SPAN, new_cell_ann_id
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
            number_of_rows = max(cell.get_sub_category(CellKey.ROW_NUMBER).category_id for cell in cells)
            number_of_cols = max(cell.get_sub_category(CellKey.COLUMN_NUMBER).category_id for cell in cells)
            max_row_span = max(cell.get_sub_category(CellKey.ROW_SPAN).category_id for cell in cells)
            max_col_span = max(cell.get_sub_category(CellKey.COLUMN_SPAN).category_id for cell in cells)
            # TODO: the summaries should be sub categories of the underlying ann
            if (
                TableKey.NUMBER_OF_ROWS in table.image.summary.sub_categories
                and TableKey.NUMBER_OF_COLUMNS in table.image.summary.sub_categories
                and TableKey.MAX_ROW_SPAN in table.image.summary.sub_categories
                and TableKey.MAX_COL_SPAN in table.image.summary.sub_categories
            ):
                table.image.summary.pop_sub_category(TableKey.NUMBER_OF_ROWS)
                table.image.summary.pop_sub_category(TableKey.NUMBER_OF_COLUMNS)
                table.image.summary.pop_sub_category(TableKey.MAX_ROW_SPAN)
                table.image.summary.pop_sub_category(TableKey.MAX_COL_SPAN)

            self.dp_manager.set_summary_annotation(
                TableKey.NUMBER_OF_ROWS, TableKey.NUMBER_OF_ROWS, number_of_rows, annotation_id=table.annotation_id
            )
            self.dp_manager.set_summary_annotation(
                TableKey.NUMBER_OF_COLUMNS,
                TableKey.NUMBER_OF_COLUMNS,
                number_of_cols,
                annotation_id=table.annotation_id,
            )
            self.dp_manager.set_summary_annotation(
                TableKey.MAX_ROW_SPAN, TableKey.MAX_ROW_SPAN, max_row_span, annotation_id=table.annotation_id
            )
            self.dp_manager.set_summary_annotation(
                TableKey.MAX_COL_SPAN, TableKey.MAX_COL_SPAN, max_col_span, annotation_id=table.annotation_id
            )
            html = generate_html_payload(table, self.cell_names)
            self.dp_manager.set_container_annotation(TableKey.HTML, -1, TableKey.HTML, table.annotation_id, html)

    def clone(self) -> TableSegmentationRefinementService:
        return self.__class__(self.table_name, self.cell_names)

    def get_meta_annotation(self) -> MetaAnnotation:
        return MetaAnnotation(
            image_annotations=(),
            sub_categories={
                LayoutLabel.CELL: {
                    CellKey.ROW_NUMBER: {CellKey.ROW_NUMBER},
                    CellKey.COLUMN_NUMBER: {CellKey.COLUMN_NUMBER},
                    CellKey.ROW_SPAN: {CellKey.ROW_SPAN},
                    CellKey.COLUMN_SPAN: {CellKey.COLUMN_SPAN},
                },
                LayoutLabel.TABLE: {TableKey.HTML: {TableKey.HTML}},
            },
            relationships={},
            summaries=(),
        )

    def clear_predictor(self) -> None:
        pass
