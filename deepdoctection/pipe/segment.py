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
ious/ioas of rows and columns.
"""


from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Union

import numpy as np

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import BoundingBox, iou
from ..datapoint.image import Image
from ..mapper.maputils import MappingContextManager
from ..mapper.match import match_anns_by_intersection
from ..utils.detection_types import JsonDict
from ..utils.settings import CellType, LayoutType, ObjectTypes, Relationships
from .base import PipelineComponent
from .registry import pipeline_component_registry

__all__ = ["TableSegmentationService", "SegmentationResult"]


@dataclass
class SegmentationResult:
    """
    Simple mutable storage for segmentation results
    """

    annotation_id: str
    row_num: int
    col_num: int
    rs: int
    cs: int


def choose_items_by_iou(
    dp: Image,
    item_proposals: List[ImageAnnotation],
    iou_threshold: float,
    above_threshold: bool = True,
    reference_item_proposals: Optional[List[ImageAnnotation]] = None,
) -> Image:
    """
    Deactivate image annotations that have ious with each other above some threshold. It will deactivate an annotation
    that has iou above some threshold with another annotation and that has a lesser score.

    :param dp: image
    :param item_proposals:
    :param iou_threshold:
    :param above_threshold:
    :param reference_item_proposals:
    """
    item_proposals_boxes = np.array(
        [
            item.image.get_embedding(dp.image_id).to_list(mode="xyxy")
            for item in item_proposals
            if item.image is not None
        ]
    )

    triangle_ind = None
    if reference_item_proposals is not None:
        reference_item_proposals_boxes = np.array(
            [
                item.image.get_embedding(dp.image_id).to_list(mode="xyxy")
                for item in reference_item_proposals
                if item.image is not None
            ]
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

    :param dp: Image
    :param table: table image annotation
    :param row_name: item name for horizontal stretching
    :param col_name:  item name for vertical stretching
    :param remove_iou_threshold_rows: iou threshold for removing overlapping rows
    :param remove_iou_threshold_cols: iou threshold for removing overlapping columns
    :return: Image
    """

    item_ann_ids = table.get_relationship(Relationships.child)

    rows = dp.get_annotation(category_names=row_name, annotation_ids=item_ann_ids)
    if table.image is None:
        raise ValueError("table.image cannot be None")
    table_embedding_box = table.image.get_embedding(dp.image_id)

    for row in rows:
        if row.image is None:
            raise ValueError("row.image cannot be None")
        row_embedding_box = row.image.get_embedding(dp.image_id)
        row_embedding_box.ulx = table_embedding_box.ulx + 1.0
        row_embedding_box.lrx = table_embedding_box.lrx - 1.0

    choose_items_by_iou(dp, rows, remove_iou_threshold_rows)

    cols = dp.get_annotation(category_names=col_name, annotation_ids=item_ann_ids)

    for col in cols:
        if col.image is None:
            raise ValueError("row.image cannot be None")
        col_embedding_box = col.image.get_embedding(dp.image_id)
        col_embedding_box.uly = table_embedding_box.uly + 1.0
        col_embedding_box.lry = table_embedding_box.lry - 1.0

    choose_items_by_iou(dp, cols, remove_iou_threshold_cols)

    return dp


def _tile_by_stretching_rows_left_and_rightwise(
    dp: Image, items: List[ImageAnnotation], table: ImageAnnotation, item_name: str
) -> None:
    if table.image is None:
        raise ValueError("table.image cannot be None")
    table_embedding_box = table.image.get_embedding(dp.image_id)

    tmp_item_xy = table_embedding_box.uly + 1.0 if item_name == LayoutType.row else table_embedding_box.ulx + 1.0
    for idx, item in enumerate(items):
        with MappingContextManager(dp_name=dp.file_name):
            if item.image is None:
                raise ValueError("item.image cannot be None")
            item_embedding_box = item.image.get_embedding(dp.image_id)
            if idx != len(items) - 1:
                next_item_embedding_box = items[idx + 1].image.get_embedding(dp.image_id)  # type: ignore
                tmp_next_item_xy = (
                    (item_embedding_box.lry + next_item_embedding_box.uly) / 2
                    if item_name == LayoutType.row
                    else (item_embedding_box.lrx + next_item_embedding_box.ulx) / 2
                )
            else:
                tmp_next_item_xy = (
                    table_embedding_box.lry - 1.0 if item_name == LayoutType.row else table_embedding_box.lrx - 1.0
                )

            new_embedding_box = BoundingBox(
                ulx=item_embedding_box.ulx if item_name == LayoutType.row else tmp_item_xy,
                uly=tmp_item_xy if item_name == LayoutType.row else item_embedding_box.uly,
                lrx=item_embedding_box.lrx if item_name == LayoutType.row else tmp_next_item_xy,
                lry=tmp_next_item_xy if item_name == LayoutType.row else item_embedding_box.lry,
                absolute_coords=True,
            )
            item.image.set_embedding(dp.image_id, new_embedding_box)
            tmp_item_xy = tmp_next_item_xy


def _tile_by_stretching_rows_leftwise_column_downwise(
    dp: Image, items: List[ImageAnnotation], table: ImageAnnotation, item_name: str
) -> None:
    if table.image is None:
        raise ValueError("table.image cannot be None")
    table_embedding_box = table.image.get_embedding(dp.image_id)

    tmp_item_xy = table_embedding_box.uly + 1.0 if item_name == LayoutType.row else table_embedding_box.ulx + 1.0
    for item in items:
        with MappingContextManager(dp_name=dp.file_name):
            if item.image is None:
                raise ValueError("item.image cannot be None")
            item_embedding_box = item.image.get_embedding(dp.image_id)
            new_embedding_box = BoundingBox(
                ulx=item_embedding_box.ulx if item_name == LayoutType.row else tmp_item_xy,
                uly=tmp_item_xy if item_name == LayoutType.row else item_embedding_box.uly,
                lrx=item_embedding_box.lrx,
                lry=item_embedding_box.lry,
                absolute_coords=True,
            )

            if item == items[-1]:
                new_embedding_box = BoundingBox(
                    ulx=item_embedding_box.ulx if item_name == LayoutType.row else tmp_item_xy,
                    uly=tmp_item_xy if item_name == LayoutType.row else item_embedding_box.uly,
                    lrx=item_embedding_box.lrx if item_name == LayoutType.row else table_embedding_box.lrx - 1.0,
                    lry=table_embedding_box.lry - 1.0 if item_name == LayoutType.row else item_embedding_box.lry,
                    absolute_coords=True,
                )

            tmp_item_xy = item_embedding_box.lry if item_name == LayoutType.row else item_embedding_box.lrx
            item.image.set_embedding(dp.image_id, new_embedding_box)


def tile_tables_with_items_per_table(
    dp: Image, table: ImageAnnotation, item_name: str, stretch_rule: Literal["left", "equal"] = "left"
) -> Image:
    """
    Tiling a table with items (i.e. rows or columns). To ensure that every position in a table can be assigned to a row
    or column, rows are stretched vertically and columns horizontally. The stretching takes place according to ascending
    coordinate axes. The first item is stretched to the top or right-hand edge of the table. The next item down or to
    the right is stretched to the lower or right edge of the previous item.

    :param dp: Image
    :param table: table
    :param item_name: names.C.ROW or names.C.COL
    :param stretch_rule: Tiling can be achieved by two different stretching rules for rows and columns.
                         - 'left': The upper horizontal edge of a row will be shifted up to the lower horizontal edge
                                   of the upper neighboring row. Similarly, the left sided vertical edge of a column
                                   will be shifted towards the right sided vertical edge of the left sided neighboring
                                   column.
                         - 'equal': Upper and lower horizontal edge of rows will be shifted to the middle of the gap
                                    of two neighboring rows. Similarly, left and right sided vertical edge of a column
                                    will be shifted to the middle of the gap of two neighboring columns.
    :return: Image
    """

    item_ann_ids = table.get_relationship(Relationships.child)
    items = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)
    items.sort(key=lambda x: x.bounding_box.cx if item_name == LayoutType.column else x.bounding_box.cy)  # type: ignore

    if stretch_rule == "left":
        _tile_by_stretching_rows_leftwise_column_downwise(dp, items, table, item_name)
    else:
        _tile_by_stretching_rows_left_and_rightwise(dp, items, table, item_name)

    return dp


def stretch_items(
    dp: Image,
    table_name: str,
    row_name: str,
    col_name: str,
    remove_iou_threshold_rows: float,
    remove_iou_threshold_cols: float,
) -> Image:
    """
    Stretch rows and columns from item detector to full table length and width. See :func:`stretch_item_per_table`

    :param dp: Image
    :param table_name: category name for a table category ann.
    :param row_name: category name for row category ann
    :param col_name: category name for column category ann
    :param remove_iou_threshold_rows: iou threshold for removing overlapping rows
    :param remove_iou_threshold_cols: iou threshold for removing overlapping columns
    :return: An Image
    """
    table_anns = dp.get_annotation_iter(category_names=table_name)

    for table in table_anns:
        dp = stretch_item_per_table(dp, table, row_name, col_name, remove_iou_threshold_rows, remove_iou_threshold_cols)

    return dp


def _default_segment_table(cells: List[ImageAnnotation]) -> List[SegmentationResult]:
    """
    Error segmentation handling when segmentation goes wrong. It will generate a default segmentation, e.g. no real
    segmentation.

    :param cells: list of all cells of one table
    :return: list of segmentation results
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
) -> List[SegmentationResult]:
    """
    Segments a table,i.e. produces for each cell a SegmentationResult. It uses numbered rows and columns that have to
    be predicted by an appropriate detector. E.g. for calculating row and rwo spans it first infers the iou of a cell
    with all rows. All ious with rows above iou_threshold_rows will induce the cell to have that row number. As there
    might be several rows, the row number of the cell will be the smallest of the number of all intersected rows. The
    row span will be equal the number of all rows with iou above the iou threshold.

    :param dp: A datapoint
    :param table: the table as image annotation.
    :param item_names: A list of item names (e.g. "ROW" and "COLUMN")
    :param cell_names: A list of cell names (e.g. "CELL")
    :param segment_rule: 'iou' or 'ioa'
    :param threshold_rows: the iou/ioa threshold of a cell with a row in order to conclude that the cell belongs
                               to the row.
    :param threshold_cols: the iou/ioa threshold of a cell with a column in order to conclude that the cell belongs
                               to the column.
    :return: A list of len(number of cells) of SegmentationResult.
    """

    child_ann_ids = table.get_relationship(Relationships.child)
    cell_index_rows, row_index, _, _ = match_anns_by_intersection(
        dp,
        item_names[0],
        cell_names,
        segment_rule,
        threshold_rows,
        True,
        child_ann_ids,
        child_ann_ids,
    )

    cell_index_cols, col_index, _, _ = match_anns_by_intersection(
        dp,
        item_names[1],
        cell_names,
        segment_rule,
        threshold_cols,
        True,
        child_ann_ids,
        child_ann_ids,
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
                row_number = min([row.get_sub_category(CellType.row_number).category_id for row in rows_of_cell])
            else:
                row_number = 0

            cell_positions_cols = cell_index_cols == idx
            cols_of_cell = [columns[k] for k in col_index[cell_positions_cols]]
            cs = np.count_nonzero(cell_index_cols == idx)
            if len(cols_of_cell):
                col_number = min([col.get_sub_category(CellType.column_number).category_id for col in cols_of_cell])
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


@pipeline_component_registry.register("TableSegmentationService")
class TableSegmentationService(PipelineComponent):
    """
    Table segmentation after successful cell detection. In addition, row and column detection must have been carried
    out.

    After cell recognition, these must be given a semantically correct position within the table. The row number,
    column number, row span and column span of the cell are determined. The determination takes place via an assignment
    via intersection.

    - Predicted rows are stretched horizontally to the edges of the table. Columns are stretched vertically. There is
      also the option of stretching rows and columns so that they completely pave the table (set tile_table_with_items
      =True).

    - Next, rows and columns are given a row or column number by sorting them vertically or horizontally
      according to the box center.

    - The averages are then determined in pairs separately for rows and columns (more precisely: Iou /
      intersection-over-union or ioa / intersection-over-area of rows and cells or columns and cells. A cell is
      assigned a row position if the iou / ioa is above a defined threshold.

    - The minimum row or column with which the cell was matched is used as the row and column of the cell. Row span /
      col span result from the number of matched rows and columns.

    It should be noted that this method means that cell positions can be assigned multiple times by different cells.
    If this should be excluded, class:`TableSegmentationRefinementService` can be used to merge cells.
    """

    def __init__(
        self,
        segment_rule: Literal["iou", "ioa"],
        threshold_rows: float,
        threshold_cols: float,
        tile_table_with_items: bool,
        remove_iou_threshold_rows: float,
        remove_iou_threshold_cols: float,
        stretch_rule: Literal["left", "equal"] = "left",
    ):
        """
        :param segment_rule: rule to assign cell to row, columns resp. must be either iou or ioa
        :param threshold_rows: iou/ioa threshold for rows
        :param threshold_cols: iou/ioa threshold for columns
        :param tile_table_with_items: Will shift the left edge of rows vertically to coincide with the right edge of
                                      the adjacent row. Will do a similar shifting with columns.
        :param remove_iou_threshold_rows: iou threshold for removing overlapping rows
        :param remove_iou_threshold_cols: iou threshold for removing overlapping columns
        :param stretch_rule: Check the description in :func:`tile_tables_with_items_per_table`
        """
        assert segment_rule in ("iou", "ioa"), "segment_rule must be either iou or ioa"
        assert stretch_rule in ("left", "equal"), "stretch rule must be either 'left' or 'equal'"

        self.segment_rule = segment_rule
        self.threshold_rows = threshold_rows
        self.threshold_cols = threshold_cols
        self.tile_table = tile_table_with_items
        self.remove_iou_threshold_rows = remove_iou_threshold_rows
        self.remove_iou_threshold_cols = remove_iou_threshold_cols
        self.stretch_rule = stretch_rule
        self._table_name = LayoutType.table
        self._cell_names = [CellType.header, CellType.body, LayoutType.cell]
        self._item_names = [LayoutType.row, LayoutType.column]  # row names must be before column name
        self._sub_item_names = [CellType.row_number, CellType.column_number]
        super().__init__("table_segment")

    def serve(self, dp: Image) -> None:
        dp = stretch_items(
            dp,
            self._table_name,
            self._item_names[0],
            self._item_names[1],
            self.remove_iou_threshold_rows,
            self.remove_iou_threshold_cols,
        )
        table_anns = dp.get_annotation(category_names=self._table_name)
        for table in table_anns:
            item_ann_ids = table.get_relationship(Relationships.child)
            for item_sub_item_name in zip(self._item_names, self._sub_item_names):  # one pass for rows and one for cols
                item_name, sub_item_name = item_sub_item_name[0], item_sub_item_name[1]
                if self.tile_table:
                    dp = tile_tables_with_items_per_table(dp, table, item_name, self.stretch_rule)
                items_proposals = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)
                reference_items_proposals = dp.get_annotation(
                    category_names=self._cell_names, annotation_ids=item_ann_ids
                )
                dp = choose_items_by_iou(dp, items_proposals, 0.0001, False, reference_items_proposals)

                # new query of items so that we only get active annotations
                items = dp.get_annotation(category_names=item_name, annotation_ids=item_ann_ids)
                items.sort(
                    key=lambda x: x.bounding_box.cx  # type: ignore
                    if item_name == LayoutType.column  # pylint: disable=W0640
                    else x.bounding_box.cy  # type: ignore
                )
                for item_number, item in enumerate(items, 1):
                    self.dp_manager.set_category_annotation(
                        sub_item_name, item_number, sub_item_name, item.annotation_id
                    )
            raw_table_segments = segment_table(
                dp,
                table,
                self._item_names,
                self._cell_names,
                self.segment_rule,
                self.threshold_rows,
                self.threshold_cols,
            )
            for segment_result in raw_table_segments:
                self.dp_manager.set_category_annotation(
                    CellType.row_number, segment_result.row_num, CellType.row_number, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.column_number, segment_result.col_num, CellType.column_number, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.row_span, segment_result.rs, CellType.row_span, segment_result.annotation_id
                )
                self.dp_manager.set_category_annotation(
                    CellType.column_span, segment_result.cs, CellType.column_span, segment_result.annotation_id
                )

    def clone(self) -> PipelineComponent:
        return self.__class__(
            self.segment_rule,
            self.threshold_rows,
            self.threshold_cols,
            self.tile_table,
            self.remove_iou_threshold_rows,
            self.remove_iou_threshold_cols,
        )

    def get_meta_annotation(self) -> JsonDict:
        return dict(
            [
                ("image_annotations", []),
                (
                    "sub_categories",
                    {
                        LayoutType.cell: {
                            CellType.row_number,
                            CellType.column_number,
                            CellType.row_span,
                            CellType.column_span,
                        },
                        LayoutType.row: {CellType.row_number},
                        LayoutType.column: {CellType.column_number},
                    },
                ),
                ("relationships", {}),
                ("summaries", []),
            ]
        )
