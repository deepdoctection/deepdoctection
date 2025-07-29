# -*- coding: utf-8 -*-
# File: segment.py

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
Module for pipeline component of table segmentation. Uses row/column detector and infers segmentations by using
`ious`/`ioas` of rows and columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import numpy as np

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import BoundingBox, global_to_local_coords, intersection_box, intersection_boxes, iou, merge_boxes
from ..datapoint.image import Image, MetaAnnotation
from ..extern.base import DetectionResult
from ..mapper.maputils import MappingContextManager
from ..mapper.match import match_anns_by_intersection
from ..utils.error import ImageError
from ..utils.settings import CellType, LayoutType, ObjectTypes, Relationships, TableType, TypeOrStr, get_type
from .base import PipelineComponent
from .refine import generate_html_string
from .registry import pipeline_component_registry

__all__ = ["TableSegmentationService", "SegmentationResult", "PubtablesSegmentationService"]


@dataclass
class SegmentationResult:
    """
    Mutable storage for segmentation results.

    Args:
        annotation_id: The annotation ID.
        row_num: The row number.
        col_num: The column number.
        rs: The row span.
        cs: The column span.
    """

    annotation_id: str
    row_num: int
    col_num: int
    rs: int
    cs: int


@dataclass
class ItemHeaderResult:
    """
    Simple mutable storage for item header results
    """

    annotation_id: str


def choose_items_by_iou(
    dp: Image,
    item_proposals: list[ImageAnnotation],
    iou_threshold: float,
    above_threshold: bool = True,
    reference_item_proposals: Optional[list[ImageAnnotation]] = None,
) -> Image:
    """
    Deactivate image annotations that have `ious` with each other above some threshold. It will deactivate an annotation
    that has `iou` above some threshold with another annotation and that has a lesser score.

    Args:
        dp: `image`.
        item_proposals: Annotations to choose from. If `reference_item_proposals` is `None` it will compare items with
                        each other.
        iou_threshold: `iou_threshold`.
        above_threshold: Whether to deactivate items above the threshold.
        reference_item_proposals: Annotations as reference. If provided, it will compare `item_proposals` with
                                 `reference_item_proposals`.

    Returns:
        The updated `Image`.
    """
    if len(item_proposals) <= 1:  # we want to ensure to have at least one element.
        return dp
    item_proposals_boxes = np.array(
        [item.get_bounding_box(dp.image_id).to_list(mode="xyxy") for item in item_proposals]
    )

    triangle_ind = None
    if reference_item_proposals is not None:
        reference_item_proposals_boxes = np.array(
            [item.get_bounding_box(dp.image_id).to_list(mode="xyxy") for item in reference_item_proposals]
        )

    else:
        reference_item_proposals_boxes = item_proposals_boxes
        triangle_ind = np.triu_indices(len(item_proposals))

    iou_matrix = iou(item_proposals_boxes, reference_item_proposals_boxes)

    if triangle_ind is not None:
        iou_matrix[triangle_ind] = 0

    indices_to_deactivate = np.where((iou_matrix > iou_threshold) & (iou_matrix != 1))[0]

    if above_threshold:
        for el in indices_to_deactivate:
            item_proposals[el].deactivate()

    else:
        unique_indices = np.unique(indices_to_deactivate)
        for idx, el in enumerate(item_proposals):
            if idx not in unique_indices:
                item_proposals[idx].deactivate()

    return dp


def stretch_item_per_table(
    dp: Image,
    table: ImageAnnotation,
    row_name: str,
    col_name: str,
    remove_iou_threshold_rows: float,
    remove_iou_threshold_cols: float,
) -> Image:
    """
    Stretch rows horizontally and stretch columns vertically. Since the predictor usually does not predict a box for
    lines across the entire width of the table, lines are stretched from the left to the right edge of the table if the
    y coordinates remain the same. Columns between the top and bottom of the table can be stretched in an analogous way.

    Args:
        dp: `Image`.
        table: Table `ImageAnnotation`.
        row_name: Item name for horizontal stretching.
        col_name: Item name for vertical stretching.
        remove_iou_threshold_rows: `iou` threshold for removing overlapping rows.
        remove_iou_threshold_cols: `iou` threshold for removing overlapping columns.

    Returns:
        The updated `Image`.
    """
    item_ann_ids = table.get_relationship(Relationships.CHILD)

    rows = dp.get_annotation(category_names=row_name, annotation_ids=item_ann_ids)
    if table.image is None:
        raise ImageError("table.image cannot be None")
    table_embedding_box = table.get_bounding_box(dp.image_id)

    for row in rows:
        if row.image is None:
            raise ImageError("row.image cannot be None")
        row_embedding_box = row.get_bounding_box(dp.image_id)
        row_embedding_box.ulx = table_embedding_box.ulx + 1.0
        row_embedding_box.lrx = table_embedding_box.lrx - 1.0

        # updating all bounding boxes for rows
        row.image.set_embedding(
            row.annotation_id,
            BoundingBox(
                ulx=0.0,
                uly=0.0,
                height=row_embedding_box.height,
                width=row_embedding_box.width,
                absolute_coords=row_embedding_box.absolute_coords,
            ),
        )
        local_row_box = global_to_local_coords(row_embedding_box, table_embedding_box)
        row.image.set_embedding(table.annotation_id, local_row_box)

    choose_items_by_iou(dp, rows, remove_iou_threshold_rows)

    cols = dp.get_annotation(category_names=col_name, annotation_ids=item_ann_ids)

    for col in cols:
        if col.image is None:
            raise ImageError("row.image cannot be None")
        col_embedding_box = col.get_bounding_box(dp.image_id)
        col_embedding_box.uly = table_embedding_box.uly + 1.0
        col_embedding_box.lry = table_embedding_box.lry - 1.0

        # updating all bounding boxes for cols
        col.image.set_embedding(
            col.annotation_id,
            BoundingBox(
                ulx=0.0,
                uly=0.0,
                height=col_embedding_box.height,
                width=col_embedding_box.width,
                absolute_coords=col_embedding_box.absolute_coords,
            ),
        )
        local_row_box = global_to_local_coords(col_embedding_box, table_embedding_box)
        col.image.set_embedding(table.annotation_id, local_row_box)

    choose_items_by_iou(dp, cols, remove_iou_threshold_cols)

    return dp


def _tile_by_stretching_rows_left_and_rightwise(
    dp: Image, items: list[ImageAnnotation], table: ImageAnnotation, item_name: str
) -> None:
    if table.image is None:
        raise ImageError("table.image cannot be None")
    table_embedding_box = table.get_bounding_box(dp.image_id)

    tmp_item_xy = table_embedding_box.uly + 1.0 if item_name == LayoutType.ROW else table_embedding_box.ulx + 1.0
    tmp_item_table_xy = 1.0
    for idx, item in enumerate(items):
        with MappingContextManager(
            dp_name=dp.file_name,
            filter_level="bounding box",
            image_annotation={"category_name": item.category_name, "annotation_id": item.annotation_id},
        ):
            if item.image is None:
                raise ImageError("item.image cannot be None")
            item_embedding_box = item.get_bounding_box(dp.image_id)
            if idx != len(items) - 1:
                next_item_embedding_box = items[idx + 1].get_bounding_box(dp.image_id)
                tmp_next_item_xy = (
                    (item_embedding_box.lry + next_item_embedding_box.uly) / 2
                    if item_name == LayoutType.ROW
                    else (item_embedding_box.lrx + next_item_embedding_box.ulx) / 2
                )
            else:
                tmp_next_item_xy = (
                    table_embedding_box.lry - 1.0 if item_name == LayoutType.ROW else table_embedding_box.lrx - 1.0
                )

            new_embedding_box = BoundingBox(
                ulx=item_embedding_box.ulx if item_name == LayoutType.ROW else tmp_item_xy,
                uly=tmp_item_xy if item_name == LayoutType.ROW else item_embedding_box.uly,
                lrx=item_embedding_box.lrx if item_name == LayoutType.ROW else tmp_next_item_xy,
                lry=tmp_next_item_xy if item_name == LayoutType.ROW else item_embedding_box.lry,
                absolute_coords=True,
            )
            item.image.set_embedding(dp.image_id, new_embedding_box)
            tmp_item_xy = tmp_next_item_xy

            item_table_embedding_box = item.get_bounding_box(table.annotation_id)
            if idx != len(items) - 1:
                next_item_table_embedding_box = items[idx + 1].get_bounding_box(table.annotation_id)
                tmp_table_next_item_xy = (
                    (item_table_embedding_box.lry + next_item_table_embedding_box.uly) / 2
                    if item_name == LayoutType.ROW
                    else (item_table_embedding_box.lrx + next_item_table_embedding_box.ulx) / 2
                )
            else:
                tmp_table_next_item_xy = (
                    table.image.height - 1.0 if item_name == LayoutType.ROW else table.image.width - 1.0
                )

            new_table_embedding_box = BoundingBox(
                ulx=item_table_embedding_box.ulx if item_name == LayoutType.ROW else tmp_item_table_xy,
                uly=tmp_item_table_xy if item_name == LayoutType.ROW else item_table_embedding_box.uly,
                lrx=item_table_embedding_box.lrx if item_name == LayoutType.ROW else tmp_table_next_item_xy,
                lry=tmp_table_next_item_xy if item_name == LayoutType.ROW else item_table_embedding_box.lry,
                absolute_coords=True,
            )
            item.image.set_embedding(table.annotation_id, new_table_embedding_box)
            tmp_item_table_xy = tmp_table_next_item_xy


def _tile_by_stretching_rows_leftwise_column_downwise(
    dp: Image, items: list[ImageAnnotation], table: ImageAnnotation, item_name: str
) -> None:
    if table.image is None:
        raise ImageError("table.image cannot be None")
    table_embedding_box = table.get_bounding_box(dp.image_id)

    tmp_item_xy = table_embedding_box.uly + 1.0 if item_name == LayoutType.ROW else table_embedding_box.ulx + 1.0
    tmp_item_table_xy = 1.0
    for item in items:
        with MappingContextManager(
            dp_name=dp.file_name,
            filter_level="bounding box",
            image_annotation={"category_name": item.category_name, "annotation_id": item.annotation_id},
        ):
            if item.image is None:
                raise ImageError("item.image cannot be None")
            item_embedding_box = item.get_bounding_box(dp.image_id)
            new_embedding_box = BoundingBox(
                ulx=item_embedding_box.ulx if item_name == LayoutType.ROW else tmp_item_xy,
                uly=tmp_item_xy if item_name == LayoutType.ROW else item_embedding_box.uly,
                lrx=item_embedding_box.lrx,
                lry=item_embedding_box.lry,
                absolute_coords=True,
            )
            item_table_embedding_box = item.get_bounding_box(table.annotation_id)
            new_table_embedding_box = BoundingBox(
                ulx=item_table_embedding_box.ulx if item_name == LayoutType.ROW else tmp_item_table_xy,
                uly=tmp_item_table_xy if item_name == LayoutType.ROW else item_table_embedding_box.uly,
                lrx=item_table_embedding_box.lrx,
                lry=item_table_embedding_box.lry,
                absolute_coords=True,
            )

            if item == items[-1]:
                new_embedding_box = BoundingBox(
                    ulx=item_embedding_box.ulx if item_name == LayoutType.ROW else tmp_item_xy,
                    uly=tmp_item_xy if item_name == LayoutType.ROW else item_embedding_box.uly,
                    lrx=item_embedding_box.lrx if item_name == LayoutType.ROW else table_embedding_box.lrx - 1.0,
                    lry=table_embedding_box.lry - 1.0 if item_name == LayoutType.ROW else item_embedding_box.lry,
                    absolute_coords=True,
                )
                new_table_embedding_box = BoundingBox(
                    ulx=item_table_embedding_box.ulx if item_name == LayoutType.ROW else tmp_item_table_xy,
                    uly=tmp_item_table_xy if item_name == LayoutType.ROW else item_table_embedding_box.uly,
                    lrx=item_table_embedding_box.lrx if item_name == LayoutType.ROW else table.image.width - 1.0,
                    lry=table.image.height - 1.0 if item_name == LayoutType.ROW else item_table_embedding_box.lry,
                    absolute_coords=True,
                )

            tmp_item_xy = item_embedding_box.lry if item_name == LayoutType.ROW else item_embedding_box.lrx
            tmp_item_table_xy = (
                item_table_embedding_box.lry if item_name == LayoutType.ROW else item_table_embedding_box.lrx
            )
            item.image.set_embedding(dp.image_id, new_embedding_box)
            item.image.set_embedding(table.annotation_id, new_table_embedding_box)


def tile_tables_with_items_per_table(
    dp: Image, table: ImageAnnotation, item_name: ObjectTypes, stretch_rule: Literal["left", "equal"] = "left"
) -> Image:
    """
    Tiling a table with items (i.e. rows or columns). To ensure that every position in a table can be assigned to a row
    or column, rows are stretched vertically and columns horizontally. The stretching takes place according to ascending
    coordinate axes. The first item is stretched to the top or right-hand edge of the table. The next item down or to
    the right is stretched to the lower or right edge of the previous item.

    Args:
        dp: `Image`.
        table: Table.
        item_name: `names.C.ROW` or `names.C.COL`.
        stretch_rule: Tiling can be achieved by two different stretching rules for rows and columns.
            - `left`: The upper horizontal edge of a row will be shifted up to the lower horizontal edge of the upper
                      neighboring row. Similarly, the left sided vertical edge of a column will be shifted towards the
                      right sided vertical edge of the left sided neighboring column.
            - `equal`: Upper and lower horizontal edge of rows will be shifted to the middle of the gap of two
                       neighboring rows. Similarly, left and right sided vertical edge of a column will be shifted to
                       the middle of the gap of two neighboring columns.

    Returns:
        The updated `Image`.
    """

    item_ann_ids = table.get_relationship(Relationships.CHILD)
    items = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)

    items.sort(
        key=lambda x: (
            x.get_bounding_box(dp.image_id).cx if item_name == LayoutType.COLUMN else x.get_bounding_box(dp.image_id).cy
        )
    )

    if stretch_rule == "left":
        _tile_by_stretching_rows_leftwise_column_downwise(dp, items, table, item_name)
    else:
        _tile_by_stretching_rows_left_and_rightwise(dp, items, table, item_name)

    return dp


def stretch_items(
    dp: Image,
    table_name: ObjectTypes,
    row_name: ObjectTypes,
    col_name: ObjectTypes,
    remove_iou_threshold_rows: float,
    remove_iou_threshold_cols: float,
) -> Image:
    """
    Stretch rows and columns from item detector to full table length and width. See `stretch_item_per_table`.

    Args:
        dp: `Image`.
        table_name: Category name for a table category annotation.
        row_name: Category name for row category annotation.
        col_name: Category name for column category annotation.
        remove_iou_threshold_rows: `iou` threshold for removing overlapping rows.
        remove_iou_threshold_cols: `iou` threshold for removing overlapping columns.

    Returns:
        An `Image`.
    """
    table_anns = dp.get_annotation(category_names=table_name)

    for table in table_anns:
        dp = stretch_item_per_table(dp, table, row_name, col_name, remove_iou_threshold_rows, remove_iou_threshold_cols)

    return dp


def _default_segment_table(cells: list[ImageAnnotation]) -> list[SegmentationResult]:
    """
    Error segmentation handling when segmentation goes wrong. It will generate a default segmentation, e.g. no real
    segmentation.

    Args:
        cells: List of all cells of one table.

    Returns:
        List of `SegmentationResult`.
    """
    raw_table_segments = []
    for cell in cells:
        raw_table_segments.append(
            SegmentationResult(annotation_id=cell.annotation_id, row_num=0, col_num=0, rs=0, cs=0)
        )
    return raw_table_segments


def segment_table(
    dp: Image,
    table: ImageAnnotation,
    item_names: Union[ObjectTypes, Sequence[ObjectTypes]],
    cell_names: Union[ObjectTypes, Sequence[ObjectTypes]],
    segment_rule: Literal["iou", "ioa"],
    threshold_rows: float,
    threshold_cols: float,
) -> list[SegmentationResult]:
    """
    Segments a table, i.e. produces for each cell a `SegmentationResult`. It uses numbered rows and columns that have
    to be predicted by an appropriate detector. For calculating row and row spans it first infers the `iou` of a cell
    with all rows. All `ious` with rows above `iou_threshold_rows` will induce the cell to have that row number. As
    there might be several rows, the row number of the cell will be the smallest of the number of all intersected rows.
    The row span will be equal to the number of all rows with `iou` above the `iou` threshold.

    Args:
        dp: A datapoint.
        table: The table as `ImageAnnotation`.
        item_names: A list of item names (e.g. `row` and `column`).
        cell_names: A list of cell names (e.g. `cell`).
        segment_rule: `iou` or `ioa`.
        threshold_rows: The `iou`/`ioa` threshold of a cell with a row in order to conclude that the cell belongs to
                        the row.
        threshold_cols: The `iou`/`ioa` threshold of a cell with a column in order to conclude that the cell belongs to
                        the column.

    Returns:
        A list of `SegmentationResult` for each cell.
    """

    child_ann_ids = table.get_relationship(Relationships.CHILD)
    cell_index_rows, row_index, _, _ = match_anns_by_intersection(
        dp,
        parent_ann_category_names=item_names[0],
        child_ann_category_names=cell_names,
        matching_rule=segment_rule,
        threshold=threshold_rows,
        use_weighted_intersections=True,
        # Rows and columns are child annotations of the table.
        parent_ann_ids=child_ann_ids,
        child_ann_ids=child_ann_ids,
    )

    cell_index_cols, col_index, _, _ = match_anns_by_intersection(
        dp,
        parent_ann_category_names=item_names[1],
        child_ann_category_names=cell_names,
        matching_rule=segment_rule,
        threshold=threshold_cols,
        use_weighted_intersections=True,
        # Rows and columns are child annotations of the table.
        parent_ann_ids=child_ann_ids,
        child_ann_ids=child_ann_ids,
    )

    cells = dp.get_annotation(annotation_ids=child_ann_ids, category_names=cell_names)

    rows = dp.get_annotation(annotation_ids=child_ann_ids, category_names=item_names[0])
    columns = dp.get_annotation(annotation_ids=child_ann_ids, category_names=item_names[1])

    raw_table_segments = []
    with MappingContextManager(dp_name=dp.file_name) as segment_mapping_context:
        for idx, cell in enumerate(cells):
            cell_positions_rows = cell_index_rows == idx
            rows_of_cell = [rows[k] for k in row_index[cell_positions_rows]]
            rs = np.count_nonzero(cell_index_rows == idx)
            if len(rows_of_cell):
                row_number = min(row.get_sub_category(CellType.ROW_NUMBER).category_id for row in rows_of_cell)
            else:
                row_number = 0

            cell_positions_cols = cell_index_cols == idx
            cols_of_cell = [columns[k] for k in col_index[cell_positions_cols]]
            cs = np.count_nonzero(cell_index_cols == idx)
            if len(cols_of_cell):
                col_number = min(col.get_sub_category(CellType.COLUMN_NUMBER).category_id for col in cols_of_cell)
            else:
                col_number = 0

            raw_table_segments.append(
                SegmentationResult(
                    annotation_id=cell.annotation_id,
                    row_num=row_number,
                    col_num=col_number,
                    rs=rs,
                    cs=cs,
                )
            )

    if segment_mapping_context.context_error:
        return _default_segment_table(cells)
    return raw_table_segments


def create_intersection_cells(
    rows: Sequence[ImageAnnotation],
    cols: Sequence[ImageAnnotation],
    table_annotation_id: str,
    sub_item_names: Sequence[ObjectTypes],
) -> tuple[Sequence[DetectionResult], Sequence[SegmentationResult]]:
    """
    Given rows and columns with row- and column number sub categories, create a list of `DetectionResult` and
    `SegmentationResult` as intersection of all their intersection rectangles.

    Args:
        rows: List of rows.
        cols: List of columns.
        table_annotation_id: Annotation ID of underlying table `ImageAnnotation`.
        sub_item_names: `ObjectTypes` for row-/column number.

    Returns:
        Pair of lists of `DetectionResult` and `SegmentationResult`.
    """
    boxes_rows = [row.get_bounding_box(table_annotation_id) for row in rows]
    boxes_cols = [col.get_bounding_box(table_annotation_id) for col in cols]

    boxes_cells = intersection_boxes(boxes_rows, boxes_cols)
    detect_result_cells = []
    segment_result_cells = []
    idx = 0
    break_outer_loop = False
    for row in rows:
        for col in cols:
            detect_result_cells.append(
                DetectionResult(
                    box=boxes_cells[idx].to_list(mode="xyxy"),
                    absolute_coords=boxes_cells[idx].absolute_coords,
                    class_name=LayoutType.CELL,
                )
            )
            segment_result_cells.append(
                SegmentationResult(
                    annotation_id="",
                    row_num=row.get_sub_category(sub_item_names[0]).category_id,
                    col_num=col.get_sub_category(sub_item_names[1]).category_id,
                    rs=1,
                    cs=1,
                )
            )
            idx += 1
            # it is possible to have less intersection boxes, e.g. if one cell has height/width 0. We need to break both
            # loops.
            if idx >= len(boxes_cells):
                break_outer_loop = True
                break
        if break_outer_loop:
            break
    return detect_result_cells, segment_result_cells


def header_cell_to_item_detect_result(
    dp: Image,
    table: ImageAnnotation,
    item_name: ObjectTypes,
    item_header_name: ObjectTypes,
    segment_rule: Literal["iou", "ioa"],
    threshold: float,
) -> list[ItemHeaderResult]:
    """
    Match header cells to items (rows or columns) based on intersection-over-union (`iou`) or
    intersection-over-area (`ioa`) and return a list of `ItemHeaderResult`.

    Args:
        dp: The image containing the table and items.
        table: The table `ImageAnnotation`.
        item_name: The type of items (e.g., rows or columns) to match with header cells.
        item_header_name: The type of header cells to match with items.
        segment_rule: The rule to use for matching, either `iou` or `ioa`.
        threshold: The `iou`/`ioa` threshold for matching header cells with items.

    Returns:
        A list of `ItemHeaderResult` containing the matched header cells.
    """
    child_ann_ids = table.get_relationship(Relationships.CHILD)
    item_index, _, items, _ = match_anns_by_intersection(
        dp,
        parent_ann_category_names=item_header_name,
        child_ann_category_names=item_name,
        matching_rule=segment_rule,
        threshold=threshold,
        use_weighted_intersections=True,
        parent_ann_ids=child_ann_ids,
        child_ann_ids=child_ann_ids,
    )
    item_headers = []
    for idx, item in enumerate(items):
        if idx in item_index:
            item_headers.append(ItemHeaderResult(annotation_id=item.annotation_id))
    return item_headers


def segment_pubtables(
    dp: Image,
    table: ImageAnnotation,
    item_names: Sequence[ObjectTypes],
    spanning_cell_names: Sequence[ObjectTypes],
    segment_rule: Literal["iou", "ioa"],
    threshold_rows: float,
    threshold_cols: float,
) -> list[SegmentationResult]:
    """
    Segment a table based on the results of `table-transformer-structure-recognition`. The processing assumes that cells
    have already been generated from the intersection of columns and rows and that column and row numbers have been
    inferred for rows and columns.

    Row and column positions as well as row and column lengths are determined for all types of spanning cells.
    All simple cells that are covered by a spanning cell as well in the table position (double allocation) are then
    replaced by the spanning cell and deactivated.

    Args:
        dp: `Image`.
        table: Table `ImageAnnotation`.
        item_names: A list of item names (e.g. `row` and `column`).
        spanning_cell_names: A list of spanning cell names (e.g. `projected_row_header` and `spanning`).
        segment_rule: `iou` or `ioa`.
        threshold_rows: The `iou`/`ioa` threshold of a cell with a row in order to conclude that the cell belongs to
                        the row.
        threshold_cols: The `iou`/`ioa` threshold of a cell with a column in order to conclude that the cell belongs to
                        the column.

    Returns:
        A list of `SegmentationResult` for spanning cells.
    """

    child_ann_ids = table.get_relationship(Relationships.CHILD)
    cell_index_rows, row_index, _, _ = match_anns_by_intersection(
        dp,
        parent_ann_category_names=item_names[0],
        child_ann_category_names=spanning_cell_names,
        matching_rule=segment_rule,
        threshold=threshold_rows,
        use_weighted_intersections=True,
        # Rows and columns are child annotations of the table.
        parent_ann_ids=child_ann_ids,
        child_ann_ids=child_ann_ids,
    )

    cell_index_cols, col_index, _, _ = match_anns_by_intersection(
        dp,
        parent_ann_category_names=item_names[1],
        child_ann_category_names=spanning_cell_names,
        matching_rule=segment_rule,
        threshold=threshold_cols,
        use_weighted_intersections=True,
        # Rows and columns are child annotations of the table.
        parent_ann_ids=child_ann_ids,
        child_ann_ids=child_ann_ids,
    )

    spanning_cells = dp.get_annotation(annotation_ids=child_ann_ids, category_names=spanning_cell_names)

    rows = dp.get_annotation(annotation_ids=child_ann_ids, category_names=item_names[0])
    columns = dp.get_annotation(annotation_ids=child_ann_ids, category_names=item_names[1])

    raw_table_segments = []

    with MappingContextManager(dp_name=dp.file_name) as segment_mapping_context:
        for idx, cell in enumerate(spanning_cells):
            cell_positions_rows = cell_index_rows == idx
            rows_of_cell = [rows[k] for k in row_index[cell_positions_rows]]
            if rows_of_cell:
                min_row_cell = min(rows_of_cell, key=lambda row: row.get_sub_category(CellType.ROW_NUMBER).category_id)
                max_row_cell = max(rows_of_cell, key=lambda row: row.get_sub_category(CellType.ROW_NUMBER).category_id)
                max_row = max_row_cell.get_sub_category(CellType.ROW_NUMBER).category_id
                min_row = min_row_cell.get_sub_category(CellType.ROW_NUMBER).category_id
                rs = max_row - min_row + 1
                row_number = min_row
            else:
                rs = 0
                row_number = 0

            cell_positions_cols = cell_index_cols == idx
            cols_of_cell = [columns[k] for k in col_index[cell_positions_cols]]

            if cols_of_cell:
                min_col_cell = min(
                    cols_of_cell, key=lambda col: col.get_sub_category(CellType.COLUMN_NUMBER).category_id
                )
                max_col_cell = max(
                    cols_of_cell, key=lambda col: col.get_sub_category(CellType.COLUMN_NUMBER).category_id
                )
                max_col = max_col_cell.get_sub_category(CellType.COLUMN_NUMBER).category_id
                min_col = min_col_cell.get_sub_category(CellType.COLUMN_NUMBER).category_id
                cs = max_col - min_col + 1
                col_number = min_col
            else:
                cs = 0
                col_number = 0

            if rows_of_cell and cols_of_cell:
                # We resize all bounding boxes of spanning cells so that they match with the grid structure, determined
                # by the rows ans columns.
                merge_box_image_row = merge_boxes(
                    *[min_row_cell.get_bounding_box(dp.image_id), max_row_cell.get_bounding_box(dp.image_id)]
                )
                merge_box_image_column = merge_boxes(
                    *[min_col_cell.get_bounding_box(dp.image_id), max_col_cell.get_bounding_box(dp.image_id)]
                )
                merge_box_image = intersection_box(merge_box_image_row, merge_box_image_column)
                merge_box_table_row = merge_boxes(
                    *[
                        min_row_cell.get_bounding_box(table.annotation_id),
                        max_row_cell.get_bounding_box(table.annotation_id),
                    ]
                )
                merge_box_table_column = merge_boxes(
                    *[
                        min_col_cell.get_bounding_box(table.annotation_id),
                        max_col_cell.get_bounding_box(table.annotation_id),
                    ]
                )
                merge_box_table = intersection_box(merge_box_table_row, merge_box_table_column)
                merge_box_spanning_cell_row = merge_boxes(
                    *[
                        min_row_cell.get_bounding_box(min_row_cell.annotation_id),
                        max_row_cell.get_bounding_box(max_row_cell.annotation_id),
                    ]
                )
                merge_box_spanning_cell_column = merge_boxes(
                    *[
                        min_col_cell.get_bounding_box(min_col_cell.annotation_id),
                        max_col_cell.get_bounding_box(max_col_cell.annotation_id),
                    ]
                )
                merge_box_spanning_cell = intersection_box(merge_box_spanning_cell_row, merge_box_spanning_cell_column)
                if cell.image is None:
                    raise ImageError("cell.image cannot be None")
                cell.image.set_embedding(dp.image_id, merge_box_image)
                cell.image.set_embedding(table.annotation_id, merge_box_table)
                cell.image.set_embedding(cell.annotation_id, merge_box_spanning_cell)

            raw_table_segments.append(
                SegmentationResult(
                    annotation_id=cell.annotation_id,
                    row_num=row_number,
                    col_num=col_number,
                    rs=rs,
                    cs=cs,
                )
            )

    if segment_mapping_context.context_error:
        return _default_segment_table(spanning_cells)
    return raw_table_segments


@pipeline_component_registry.register("TableSegmentationService")
class TableSegmentationService(PipelineComponent):
    """
    Table segmentation after successful cell detection. In addition, row and column detection must have been carried
    out.

    After cell recognition, these must be given a semantically correct position within the table. The row number,
    column number, row span and column span of the cell are determined. The determination takes place via an assignment
    via intersection.

    - Predicted rows are stretched horizontally to the edges of the table. Columns are stretched vertically. There is
      also the option of stretching rows and columns so that they completely pave the table (set
      `tile_table_with_items=True`).

    - Next, rows and columns are given a row or column number by sorting them vertically or horizontally
      according to the box center.

    - The averages are then determined in pairs separately for rows and columns (more precisely: `Iou` /
      intersection-over-union or `ioa` / intersection-over-area of rows and cells or columns and cells. A cell is
      assigned a row position if the `iou` / `ioa` is above a defined threshold.

    - The minimum row or column with which the cell was matched is used as the row and column of the cell. Row span /
      col span result from the number of matched rows and columns.

    Note:
        It should be noted that this method means that cell positions can be assigned multiple times by different cells.
        If this should be excluded, `TableSegmentationRefinementService` can be used to merge cells.


    """

    def __init__(
        self,
        segment_rule: Literal["iou", "ioa"],
        threshold_rows: float,
        threshold_cols: float,
        tile_table_with_items: bool,
        remove_iou_threshold_rows: float,
        remove_iou_threshold_cols: float,
        table_name: TypeOrStr,
        cell_names: Sequence[TypeOrStr],
        item_names: Sequence[TypeOrStr],
        sub_item_names: Sequence[TypeOrStr],
        stretch_rule: Literal["left", "equal"] = "left",
    ):
        """
        Args:
            segment_rule: Rule to assign cell to row, columns resp. must be either `iou` or `ioa`.
            threshold_rows: `iou`/`ioa` threshold for rows.
            threshold_cols: `iou`/`ioa` threshold for columns.
            tile_table_with_items: Will shift the left edge of rows vertically to coincide with the right edge of the
                                   adjacent row. Will do a similar shifting with columns.
            remove_iou_threshold_rows: `iou` threshold for removing overlapping rows.
            remove_iou_threshold_cols: `iou` threshold for removing overlapping columns.
            table_name: Layout type table.
            cell_names: Layout type of cells.
            item_names: Layout type of items (e.g. row and column).
            sub_item_names: Cell types of sub items (e.g. row number and column number).
            stretch_rule: Check the description in `tile_tables_with_items_per_table`.
        """
        if segment_rule not in ("iou", "ioa"):
            raise ValueError("segment_rule must be either iou or ioa")
        if stretch_rule not in ("left", "equal"):
            raise ValueError("stretch rule must be either 'left' or 'equal'")

        self.segment_rule = segment_rule
        self.threshold_rows = threshold_rows
        self.threshold_cols = threshold_cols
        self.tile_table = tile_table_with_items
        self.remove_iou_threshold_rows = remove_iou_threshold_rows
        self.remove_iou_threshold_cols = remove_iou_threshold_cols
        self.table_name = get_type(table_name)
        self.cell_names = [get_type(cell_name) for cell_name in cell_names]
        self.item_names = [get_type(item_name) for item_name in item_names]  # row names must be before column name
        self.sub_item_names = [get_type(sub_item_name) for sub_item_name in sub_item_names]
        self.stretch_rule = stretch_rule
        self.item_iou_threshold = 0.0001
        super().__init__("table_segment")

    def serve(self, dp: Image) -> None:
        dp = stretch_items(
            dp,
            self.table_name,
            self.item_names[0],
            self.item_names[1],
            self.remove_iou_threshold_rows,
            self.remove_iou_threshold_cols,
        )
        table_anns = dp.get_annotation(category_names=self.table_name)
        for table in table_anns:
            item_ann_ids = table.get_relationship(Relationships.CHILD)
            for item_sub_item_name in zip(self.item_names, self.sub_item_names):  # one pass for rows and one for cols
                item_name, sub_item_name = item_sub_item_name[0], item_sub_item_name[1]
                if self.tile_table:
                    dp = tile_tables_with_items_per_table(dp, table, item_name, self.stretch_rule)
                items_proposals = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)
                reference_items_proposals = dp.get_annotation(
                    category_names=self.cell_names, annotation_ids=item_ann_ids
                )
                dp = choose_items_by_iou(dp, items_proposals, self.item_iou_threshold, False, reference_items_proposals)

                # new query of items so that we only get active annotations
                items = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)

                # we will assume that either all or no image attribute has been generated
                items.sort(
                    key=lambda x: (
                        x.get_bounding_box(dp.image_id).cx  # pylint: disable=W0640
                        if item_name == LayoutType.COLUMN  # pylint: disable=W0640
                        else x.get_bounding_box(dp.image_id).cy  # pylint: disable=W0640
                    )
                )

                for item_number, item in enumerate(items, 1):
                    self.dp_manager.set_category_annotation(
                        sub_item_name, item_number, sub_item_name, item.annotation_id
                    )
            raw_table_segments = segment_table(
                dp,
                table,
                self.item_names,
                self.cell_names,
                self.segment_rule,
                self.threshold_rows,
                self.threshold_cols,
            )
            for segment_result in raw_table_segments:
                self.dp_manager.set_category_annotation(
                    CellType.ROW_NUMBER, segment_result.row_num, CellType.ROW_NUMBER, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.COLUMN_NUMBER, segment_result.col_num, CellType.COLUMN_NUMBER, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.ROW_SPAN, segment_result.rs, CellType.ROW_SPAN, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.COLUMN_SPAN, segment_result.cs, CellType.COLUMN_SPAN, segment_result.annotation_id
                )

            if table.image:
                cells = table.image.get_annotation(category_names=self.cell_names)
                number_of_rows = max(
                    cell.get_sub_category(CellType.ROW_NUMBER).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                number_of_cols = max(
                    cell.get_sub_category(CellType.COLUMN_NUMBER).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                max_row_span = max(
                    cell.get_sub_category(CellType.ROW_SPAN).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                max_col_span = max(
                    cell.get_sub_category(CellType.COLUMN_SPAN).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                # TODO: the summaries should be sub categories of the underlying ann
                self.dp_manager.set_summary_annotation(
                    TableType.NUMBER_OF_ROWS,
                    TableType.NUMBER_OF_ROWS,
                    number_of_rows,
                    annotation_id=table.annotation_id,
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

    def clone(self) -> TableSegmentationService:
        return self.__class__(
            self.segment_rule,
            self.threshold_rows,
            self.threshold_cols,
            self.tile_table,
            self.remove_iou_threshold_rows,
            self.remove_iou_threshold_cols,
            self.table_name,
            self.cell_names,
            self.item_names,
            self.sub_item_names,
            self.stretch_rule,
        )

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
                LayoutType.ROW: {CellType.ROW_NUMBER: {CellType.ROW_NUMBER}},
                LayoutType.COLUMN: {CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER}},
            },
            relationships={},
            summaries=(),
        )

    def clear_predictor(self) -> None:
        """clear predictor. Will do nothing"""


class PubtablesSegmentationService(PipelineComponent):
    """
    Table segmentation for table recognition detectors trained on Pubtables1M dataset. It will require `ImageAnnotation`
    of type `LayoutType.row`, `LayoutType.column` and cells of at least one type `CellType.spanning`,
    `CellType.ROW_HEADER`, `CellType.COLUMN_HEADER`, `CellType.PROJECTED_ROW_HEADER`. For table recognition using
    this service build a pipeline as follows:

    Example:
        ```python
        layout = ImageLayoutService(layout_detector, to_image=True, crop_image=True)
        recognition = SubImageLayoutService(table_recognition_detector, LayoutType.TABLE, {1: 6, 2:7, 3:8, 4:9}, True)
        segment = PubtablesSegmentationService('ioa', 0.4, 0.4, True, 0.8, 0.8, 7)
        pipe = DoctectionPipe([layout, recognition, segment])
        ```

    Under the hood this service performs the following tasks:

    - Stretching of rows and columns horizontally and vertically, so that the underlying table is fully tiled by rows
      and columns.
    - Enumerating rows and columns.
    - For intersecting rows and columns it will create an `ImageAnnotation` of category `LayoutType.cell`.
    - Using spanning cells from the detector to determine their `row_number` and `column_number` position.
    - Using cells and spanning cells, it will generate a tiling of the table with cells. When some cells have a position
      with some spanning cells, it will deactivate those simple cells and prioritize the spanning cells.
    - Determining the HTML representation of table.

    Info:
        Different from the `TableSegmentationService` this service does not require a refinement service: the advantage
        of this method is, that the segmentation can already be 'HTMLized'.


    """

    def __init__(
        self,
        segment_rule: Literal["iou", "ioa"],
        threshold_rows: float,
        threshold_cols: float,
        tile_table_with_items: bool,
        remove_iou_threshold_rows: float,
        remove_iou_threshold_cols: float,
        table_name: TypeOrStr,
        cell_names: Sequence[TypeOrStr],
        spanning_cell_names: Sequence[TypeOrStr],
        item_names: Sequence[TypeOrStr],
        sub_item_names: Sequence[TypeOrStr],
        item_header_cell_names: Sequence[TypeOrStr],
        item_header_thresholds: Sequence[float],
        cell_to_image: bool = True,
        crop_cell_image: bool = False,
        stretch_rule: Literal["left", "equal"] = "left",
    ) -> None:
        """
        Args:
            segment_rule: Rule to assign spanning cells to row, columns resp. must be either `iou` or `ioa`.
            threshold_rows: `iou`/`ioa` threshold for rows.
            threshold_cols: `iou`/`ioa` threshold for columns.
            tile_table_with_items: Will shift the left edge of rows vertically to coincide with the right edge of the
                                   adjacent row. Will do a similar shifting with columns.
            remove_iou_threshold_rows: `iou` threshold for removing overlapping rows.
            remove_iou_threshold_cols: `iou` threshold for removing overlapping columns.
            table_name: Layout type table.
            cell_names: Layout type of cells.
            spanning_cell_names: Layout type of spanning cells.
            item_names: Layout type of items (e.g. row and column).
            sub_item_names: Layout type of sub items (e.g. row number and column number).
            item_header_cell_names: Layout type of item header cells (e.g. `CellType.COLUMN_HEADER`,
                                    `CellType.ROW_HEADER`).
                                    Note that column header, resp. row header will be first assigned to rows, resp.
                                    columns and then transferred to cells.
            item_header_thresholds: `iou`/`ioa` threshold for matching header cells with items. The first threshold
                                    corresponds to matching the first entry of `item_names`.
            cell_to_image: If set to `True` it will create an `Image` for `LayoutType.cell`.
            crop_cell_image: If set to `True` it will crop a numpy array image for `LayoutType.cell`. Requires
                            `cell_to_image=True`.
            stretch_rule: Check the description in `tile_tables_with_items_per_table`.
        """
        self.segment_rule = segment_rule
        self.threshold_rows = threshold_rows
        self.threshold_cols = threshold_cols
        self.tile_table = tile_table_with_items
        self.table_name = get_type(table_name)
        self.cell_names = [get_type(cell_name) for cell_name in cell_names]
        self.spanning_cell_names = [get_type(cell_name) for cell_name in spanning_cell_names]
        self.remove_iou_threshold_rows = remove_iou_threshold_rows
        self.remove_iou_threshold_cols = remove_iou_threshold_cols
        self.cell_to_image = cell_to_image
        self.crop_cell_image = crop_cell_image
        self.item_names = [get_type(item_name) for item_name in item_names]  # row names must be before column name
        self.sub_item_names = [get_type(item_name) for item_name in sub_item_names]
        self.stretch_rule = stretch_rule
        self.item_header_cell_names = [get_type(item_name) for item_name in item_header_cell_names]
        self.item_header_thresholds = item_header_thresholds

        super().__init__("table_transformer_segment")

    def serve(self, dp: Image) -> None:
        dp = stretch_items(
            dp,
            self.table_name,
            self.item_names[0],
            self.item_names[1],
            self.remove_iou_threshold_rows,
            self.remove_iou_threshold_cols,
        )
        table_anns = dp.get_annotation(category_names=self.table_name)
        has_item_headers = {item: False for item in self.item_names}
        for table in table_anns:
            item_ann_ids = table.get_relationship(Relationships.CHILD)
            for item_sub_item_name in zip(
                self.item_names, self.sub_item_names, self.item_header_cell_names, self.item_header_thresholds
            ):  # one pass for rows and one for cols
                item_name, sub_item_name, item_header_cell_name, item_header_threshold = (
                    item_sub_item_name[0],
                    item_sub_item_name[1],
                    item_sub_item_name[2],
                    item_sub_item_name[3],
                )
                if self.tile_table:
                    dp = tile_tables_with_items_per_table(dp, table, item_name, self.stretch_rule)
                items = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)

                # we will assume that either all or no image attribute has been generated
                items.sort(
                    key=lambda x: (
                        x.get_bounding_box(dp.image_id).cx
                        if item_name == LayoutType.COLUMN  # pylint: disable=W0640
                        else x.get_bounding_box(dp.image_id).cy
                    )
                )

                item_headers_detect_results = header_cell_to_item_detect_result(
                    dp, table, item_name, item_header_cell_name, self.segment_rule, item_header_threshold
                )
                if item_headers_detect_results:
                    has_item_headers[item_name] = True

                for item_number, item in enumerate(items, 1):
                    self.dp_manager.set_category_annotation(
                        sub_item_name, item_number, sub_item_name, item.annotation_id
                    )
                for item_header_detect_result in item_headers_detect_results:
                    self.dp_manager.set_category_annotation(
                        category_name=item_header_cell_name,
                        category_id=None,
                        sub_cat_key=item_header_cell_name,
                        annotation_id=item_header_detect_result.annotation_id,
                    )

            rows = dp.get_annotation(category_names=self.item_names[0], annotation_ids=item_ann_ids)
            columns = dp.get_annotation(category_names=self.item_names[1], annotation_ids=item_ann_ids)
            detect_result_cells, segment_result_cells = create_intersection_cells(
                rows, columns, table.annotation_id, self.sub_item_names
            )
            cell_rn_cn_to_ann_id = {}
            for detect_result, segment_result in zip(detect_result_cells, segment_result_cells):
                segment_result.annotation_id = self.dp_manager.set_image_annotation(  # type: ignore
                    detect_result,
                    to_annotation_id=table.annotation_id,
                    to_image=self.cell_to_image,
                    crop_image=self.crop_cell_image,
                )
                self.dp_manager.set_category_annotation(
                    CellType.ROW_NUMBER, segment_result.row_num, CellType.ROW_NUMBER, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.COLUMN_NUMBER, segment_result.col_num, CellType.COLUMN_NUMBER, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.ROW_SPAN, segment_result.rs, CellType.ROW_SPAN, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.COLUMN_SPAN, segment_result.cs, CellType.COLUMN_SPAN, segment_result.annotation_id
                )
                cell_rn_cn_to_ann_id[(segment_result.row_num, segment_result.col_num)] = segment_result.annotation_id

            spanning_cell_raw_segments = segment_pubtables(
                dp,
                table,
                self.item_names,
                self.spanning_cell_names,
                self.segment_rule,
                self.threshold_rows,
                self.threshold_cols,
            )

            for segment_result in spanning_cell_raw_segments:
                if (
                    (segment_result.rs == 1 and segment_result.cs == 1)
                    or segment_result.rs == 0
                    or segment_result.cs == 0
                ):
                    self.dp_manager.deactivate_annotation(segment_result.annotation_id)
                    continue
                self.dp_manager.set_category_annotation(
                    CellType.ROW_NUMBER, segment_result.row_num, CellType.ROW_NUMBER, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.COLUMN_NUMBER, segment_result.col_num, CellType.COLUMN_NUMBER, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.ROW_SPAN, segment_result.rs, CellType.ROW_SPAN, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.COLUMN_SPAN, segment_result.cs, CellType.COLUMN_SPAN, segment_result.annotation_id
                )
                cells_to_deactivate = []
                for rs in range(segment_result.rs):
                    for cs in range(segment_result.cs):
                        cells_to_deactivate.append((segment_result.row_num + rs, segment_result.col_num + cs))
                for cell_position in cells_to_deactivate:
                    cell_ann_id = cell_rn_cn_to_ann_id[cell_position]
                    self.dp_manager.deactivate_annotation(cell_ann_id)

            for segment_result in spanning_cell_raw_segments:
                if (
                    (segment_result.rs == 1 and segment_result.cs == 1)
                    or segment_result.rs == 0
                    or segment_result.cs == 0
                ):
                    continue
                for rs in range(segment_result.rs):
                    for cs in range(segment_result.cs):
                        cell_rn_cn_to_ann_id[
                            (segment_result.row_num + rs, segment_result.col_num + cs)
                        ] = segment_result.annotation_id

            cells = []
            if table.image:
                cells = table.image.get_annotation(category_names=self.cell_names)
            if cells:
                number_of_rows = max(
                    cell.get_sub_category(CellType.ROW_NUMBER).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                number_of_cols = max(
                    cell.get_sub_category(CellType.COLUMN_NUMBER).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                max_row_span = max(
                    cell.get_sub_category(CellType.ROW_SPAN).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
                max_col_span = max(
                    cell.get_sub_category(CellType.COLUMN_SPAN).category_id
                    for cell in cells
                    if CellType.ROW_NUMBER in cell.sub_categories
                )
            else:
                number_of_rows = 0
                number_of_cols = 0
                max_row_span = 0
                max_col_span = 0

            for idx, item_vals in enumerate(zip(self.item_names, self.item_header_cell_names, self.sub_item_names)):
                item_obj_type, item_header_cell_name, sub_item_name = item_vals[0], item_vals[1], item_vals[2]

                if has_item_headers[item_obj_type]:
                    items = dp.get_annotation(category_names=item_obj_type)

                    for item_ann in items:
                        if item_header_cell_name in item_ann.sub_categories:
                            item_number = item_ann.get_sub_category(sub_item_name).category_id
                            for key, value in cell_rn_cn_to_ann_id.items():
                                if key[idx] == item_number:
                                    cell_ann = dp.get_annotation(annotation_ids=value)[0]
                                    if item_header_cell_name not in cell_ann.sub_categories:
                                        self.dp_manager.set_category_annotation(
                                            item_header_cell_name, None, item_header_cell_name, cell_ann.annotation_id
                                        )
                                else:
                                    cell_ann = dp.get_annotation(annotation_ids=value)[0]
                                    if CellType.BODY not in cell_ann.sub_categories:
                                        self.dp_manager.set_category_annotation(
                                            item_header_cell_name, None, CellType.BODY, cell_ann.annotation_id
                                        )

            # TODO: the summaries should be sub categories of the underlying ann
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
            html = generate_html_string(table, self.cell_names + self.spanning_cell_names)
            self.dp_manager.set_container_annotation(TableType.HTML, -1, TableType.HTML, table.annotation_id, html)

    def clone(self) -> PubtablesSegmentationService:
        return self.__class__(
            self.segment_rule,
            self.threshold_rows,
            self.threshold_cols,
            self.tile_table,
            self.remove_iou_threshold_rows,
            self.remove_iou_threshold_cols,
            self.table_name,
            self.cell_names,
            self.spanning_cell_names,
            self.item_names,
            self.sub_item_names,
            self.item_header_cell_names,
            self.item_header_thresholds,
            self.cell_to_image,
            self.crop_cell_image,
            self.stretch_rule,
        )

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
                CellType.SPANNING: {
                    CellType.ROW_NUMBER: {CellType.ROW_NUMBER},
                    CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER},
                    CellType.ROW_SPAN: {CellType.ROW_SPAN},
                    CellType.COLUMN_SPAN: {CellType.COLUMN_SPAN},
                },
                CellType.ROW_HEADER: {
                    CellType.ROW_NUMBER: {CellType.ROW_NUMBER},
                    CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER},
                    CellType.ROW_SPAN: {CellType.ROW_SPAN},
                    CellType.COLUMN_SPAN: {CellType.COLUMN_SPAN},
                },
                CellType.COLUMN_HEADER: {
                    CellType.ROW_NUMBER: {CellType.ROW_NUMBER},
                    CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER},
                    CellType.ROW_SPAN: {CellType.ROW_SPAN},
                    CellType.COLUMN_SPAN: {CellType.COLUMN_SPAN},
                },
                CellType.PROJECTED_ROW_HEADER: {
                    CellType.ROW_NUMBER: {CellType.ROW_NUMBER},
                    CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER},
                    CellType.ROW_SPAN: {CellType.ROW_SPAN},
                    CellType.COLUMN_SPAN: {CellType.COLUMN_SPAN},
                },
                LayoutType.ROW: {CellType.ROW_NUMBER: {CellType.ROW_NUMBER}},
                LayoutType.COLUMN: {CellType.COLUMN_NUMBER: {CellType.COLUMN_NUMBER}},
            },
            relationships={},
            summaries=(),
        )
