# -*- coding: utf-8 -*-
# File: pagestruct.py

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
Module for mapping Images or exported dictionaries into page formats
"""

from typing import List, Optional, Tuple

from numpy import float32

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.convert import convert_np_array_to_b64
from ..datapoint.doc import Cell, LayoutSegment, Page, Table, TableSegment
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.settings import names

__all__ = ["to_page", "page_dict_to_page"]


def _to_float(value: Optional[float32]) -> float:
    return float(value) if value is not None else 0.0


def _to_table_segment(dp: Image, annotation: ImageAnnotation) -> TableSegment:
    if annotation.image is not None:
        bounding_box = annotation.image.get_embedding(dp.image_id)
    else:
        bounding_box = annotation.bounding_box

    return TableSegment(
        annotation.annotation_id,
        bounding_box.to_list(mode="xyxy"),
        annotation.category_name,
        _to_float(annotation.score),  # type: ignore
    )


def _to_cell(dp: Image, annotation: ImageAnnotation) -> Tuple[Cell, str]:
    text_ids = annotation.get_relationship(names.C.CHILD)
    text_anns = dp.get_annotation(annotation_ids=text_ids)
    text_anns.sort(key=lambda x: int(x.get_sub_category(names.C.RO).category_id))
    text = " ".join([text.get_sub_category(names.C.CHARS).value for text in text_anns])  # type: ignore

    if annotation.image is not None:
        bounding_box = annotation.image.get_embedding(dp.image_id)
    else:
        bounding_box = annotation.bounding_box

    return (
        Cell(
            annotation.annotation_id,
            bounding_box.to_list(mode="xyxy"),
            text,
            int(annotation.get_sub_category(names.C.RN).category_id),
            int(annotation.get_sub_category(names.C.CN).category_id),
            int(annotation.get_sub_category(names.C.RS).category_id),
            int(annotation.get_sub_category(names.C.CS).category_id),
            _to_float(annotation.score),  # type: ignore
        ),
        text,
    )


def _to_table(dp: Image, annotation: ImageAnnotation) -> Table:
    cell_ids = annotation.get_relationship(names.C.CHILD)
    cell_anns = dp.get_annotation(annotation_ids=cell_ids, category_names=[names.C.CELL, names.C.HEAD, names.C.BODY])
    cells = []
    number_rows = -1
    number_cols = -1
    html_list: List[str] = []
    if names.C.HTAB in annotation.sub_categories:
        html_list = annotation.get_sub_category(names.C.HTAB).value  # type: ignore
    if annotation.image is not None:
        if annotation.image.summary is not None:
            if (
                names.C.NR in annotation.image.summary.sub_categories
                and names.C.NC in annotation.image.summary.sub_categories
            ):
                number_rows = int(annotation.image.summary.get_sub_category(names.C.NR).category_id)
                number_cols = int(annotation.image.summary.get_sub_category(names.C.NC).category_id)

    # cell
    for cell_ann in cell_anns:
        cell, text = _to_cell(dp, cell_ann)
        cells.append(cell)
        try:
            html_index = html_list.index(cell_ann.annotation_id)
            html_list.pop(html_index)
            html_list.insert(html_index, text)
        except ValueError:
            pass
    html = "".join(html_list)

    # table segments
    table_segm_ids = annotation.get_relationship(names.C.CHILD)
    table_segm_anns = dp.get_annotation(annotation_ids=table_segm_ids, category_names=[names.C.ROW, names.C.COL])
    table_segm = []
    for table_segm_ann in table_segm_anns:

        table_segm.append(_to_table_segment(dp, table_segm_ann))

    if annotation.image is not None:
        bounding_box = annotation.image.get_embedding(dp.image_id)
    else:
        bounding_box = annotation.bounding_box

    return Table(
        annotation.annotation_id,
        bounding_box.to_list(mode="xyxy"),
        cells,
        table_segm,
        number_rows,
        number_cols,
        html,
        _to_float(annotation.score),  # type: ignore
    )


def _to_layout_segment(dp: Image, annotation: ImageAnnotation) -> LayoutSegment:
    text_ids = annotation.get_relationship(names.C.CHILD)
    text_anns = dp.get_annotation(annotation_ids=text_ids)
    text_anns.sort(key=lambda x: int(x.get_sub_category(names.C.RO).category_id))
    text = " ".join([text.get_sub_category(names.C.CHARS).value for text in text_anns])  # type: ignore
    if names.C.RO in annotation.sub_categories:
        reading_order = int(annotation.get_sub_category(names.C.RO).category_id)
    else:
        reading_order = -1
    if annotation.image is not None:
        bounding_box = annotation.image.get_embedding(dp.image_id)
    else:
        bounding_box = annotation.bounding_box

    return LayoutSegment(
        annotation.annotation_id,
        bounding_box.to_list(mode="xyxy"),
        annotation.category_name,
        reading_order,
        text,
        _to_float(annotation.score),  # type: ignore
    )


def to_page(dp: Image) -> Page:
    """
    Converts an Image to the lightweight data format Page, where all detected objects are parsed into an easy consumable
    format.

    :param dp: Image
    :return: Page
    """

    # page
    image: Optional[str] = None
    if dp.image is not None:
        image = convert_np_array_to_b64(dp.image)
    page = Page(dp.image_id, dp.file_name, dp.width, dp.height, image)

    # all types of items
    for ann in dp.get_annotation(category_names=[names.C.TEXT, names.C.TITLE, names.C.LIST, names.C.TAB]):
        if ann.category_name in [names.C.TEXT, names.C.TITLE, names.C.LIST]:
            page.items.append(_to_layout_segment(dp, ann))

        # table item
        elif ann.category_name in [names.C.TAB]:
            page.tables.append(_to_table(dp, ann))
        else:
            pass

    return page


def page_dict_to_page(page_dict: JsonDict) -> Page:
    """
    Converts a dictionary (from a page export) to the page data format.

    :param page_dict: A dictionary from a page export
    :return: Page
    """

    page = Page(
        uuid=page_dict["uuid"], file_name=page_dict["file_name"], width=page_dict["width"], height=page_dict["height"]
    )

    page.image = page_dict["image"]
    tables = page_dict["tables"]
    tables_target = []
    for table_dict in tables:
        cells = table_dict["cells"]
        cells_target = []

        for cell_dict in cells:
            cell = Cell(
                uuid=cell_dict["uuid"],
                bounding_box=cell_dict["bounding_box"],
                text=cell_dict["text"],
                row_number=cell_dict["row_number"],
                col_number=cell_dict["col_number"],
                row_span=cell_dict["row_span"],
                col_span=cell_dict["col_span"],
                score=cell_dict["score"],
            )
            cells_target.append(cell)
        segment_items = page_dict["items"]
        segment_items_target = []

        for segm_item_dict in segment_items:
            segm_item = TableSegment(
                uuid=segm_item_dict["uuid"],
                bounding_box=segm_item_dict["bounding_box"],
                category=segm_item_dict["category"],
                score=segm_item_dict["score"],
            )
            segment_items_target.append(segm_item)
        table = Table(
            uuid=table_dict["uuid"],
            bounding_box=table_dict["bounding_box"],
            number_rows=table_dict["number_rows"],
            number_cols=table_dict["number_cols"],
            cells=cells_target,
            items=segment_items_target,
            score=table_dict["score"],
            html=table_dict["html"],
        )
        tables_target.append(table)
    page.tables = tables_target

    items = page_dict["items"]
    items_target = []
    for item_dict in items:
        item = LayoutSegment(
            uuid=item_dict["uuid"],
            bounding_box=item_dict["bounding_box"],
            category=item_dict["category"],
            order=item_dict["order"],
            text=item_dict["text"],
            score=item_dict["score"],
        )
        items_target.append(item)
    page.items = items_target

    return page
