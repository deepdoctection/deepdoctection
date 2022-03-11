# -*- coding: utf-8 -*-
# File: pubstruct.py

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
Module for mapping annotations in pubtabnet style structure
"""
import itertools
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..datapoint import BoundingBox, CategoryAnnotation, ImageAnnotation
from ..datapoint.annotation import SummaryAnnotation
from ..datapoint.convert import convert_pdf_bytes_to_np_array_v2
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from ..utils.fs import is_file_extension, load_bytes_from_pdf_file, load_image_from_file
from ..utils.settings import names
from .maputils import MappingContextManager, cur, maybe_get_fake_score

__all__ = ["pub_to_image"]


def _convert_boxes(dp: JsonDict, height: int) -> JsonDict:

    if "bbox" in dp:
        table_box_3 = height - dp["bbox"][1]
        dp["bbox"][1] = height - dp["bbox"][3]
        dp["bbox"][3] = table_box_3

    for cell in dp["html"]["cells"]:
        if "bbox" in cell:
            cell_box_3 = height - cell["bbox"][1]
            cell["bbox"][1] = height - cell["bbox"][3]
            cell["bbox"][3] = cell_box_3

    return dp


def _get_table_annotation(dp: JsonDict, category_id: str) -> ImageAnnotation:

    ulx, uly, lrx, lry = list(map(float, dp["bbox"]))
    bbox = BoundingBox(absolute_coords=True, ulx=ulx, uly=uly, lrx=lrx, lry=lry)
    annotation = ImageAnnotation(category_name=names.C.TAB, bounding_box=bbox, category_id=category_id)
    return annotation


def _cell_token(html: List[str]) -> List[List[int]]:

    index_rows = [i for i, tag in enumerate(html) if tag == "<tr>"]
    index_cells = [i for i, tag in enumerate(html) if tag in ("<td>", ">")]
    index_rows_tmp = [(index_rows[i], index_rows[i + 1]) for i in range(len(index_rows) - 1)]
    index_rows_tmp.append((index_rows[-1], index_cells[-1]))
    index_cells_tmp = [
        [index_cell for index_cell in index_cells if index_rows_tmp[i][0] < index_cell < index_rows_tmp[i][1]]
        for i in range(len(index_rows_tmp))
    ]
    index_cells_tmp[-1].append(index_cells[-1])
    return index_cells_tmp


def _item_spans(html: List[str], index_cells: List[List[int]], item: str) -> List[List[int]]:

    item_spans = [
        [
            int(html[index_cell - 1].replace(item + "=", "").replace('"', ""))
            if (item in html[index_cell - 1] and html[index_cell] == ">")
            else (
                int(html[index_cell - 2].replace(item + "=", "").replace('"', ""))
                if (item in html[index_cell - 2] and html[index_cell] == ">")
                else 1
            )
            for index_cell in index_cell_per_row
        ]
        for index_cell_per_row in index_cells
    ]
    return item_spans


def _end_of_header(html: List[str]) -> int:
    index_cells = [i for i, tag in enumerate(html) if tag in ("<td>", ">")]
    header_in_html = [i for i, tag in enumerate(html) if tag == "</thead>"]
    index_header_end = max(header_in_html)
    index_header = [i for i in index_cells if i < index_header_end]
    if len(index_header):
        last_index_header_cell = max(index_header)
        return index_cells.index(last_index_header_cell) + 1
    return 0


def tile_table(row_spans: List[List[int]], col_spans: List[List[int]]) -> List[List[int]]:
    """
    Tiles a table according the row and column span scheme. A table can be represented as a list of list, where each
    inner list has the same length. Each cell with a cell id can be located according to their row and column spans in
    that scheme

    :param row_spans: A list of list of row spans
    :param col_spans: A list of list of column spans
    :return: A list of list of the tiling of the table, indicating the precise place of each cell.
    """
    number_of_cols = sum(col_spans[0])
    number_of_rows = len(col_spans)
    cell_ids = []
    i = 1
    for row in col_spans:
        cell_id_per_row = []
        for idx, k in enumerate(itertools.count(i)):

            if idx < len(row):
                i += 1
                cell_id_per_row.append(k)
            else:
                break
        cell_ids.append(cell_id_per_row)

    tiling = [[-1] * number_of_cols for _ in range(number_of_rows)]  # initialize placeholders
    table = zip(cell_ids, row_spans, col_spans)

    for row_id, row in enumerate(table):  # type: ignore # pylint: disable=R1702
        for cell in zip(row[0], row[1], row[2]):  # type: ignore
            cell_id = cell[0]
            row_span = cell[1]
            col_span = cell[2]
            if 0 in tiling[row_id]:  # calculate actual position of the cell
                col = tiling[row_id].index(0)
            else:
                col = len(tiling[row_id]) - tiling[row_id].count(-1)
            # tile the cell
            for rs in range(row_span):
                for cs in range(col_span):
                    if rs >= 1 and cs == 0:  # if rowSpan>=2 every row below needs to be filled with trailing 0
                        fill = 0
                        while fill < col:
                            if tiling[row_id + rs][fill] == -1:
                                tiling[row_id + rs][fill] = 0
                            fill += 1
                    tiling[row_id + rs][col + cs] = cell_id

    np.array(tiling, dtype=np.int32)
    return tiling


def _add_items(image: Image, item_type: str, categories_name_as_key: Dict[str, str]) -> Image:

    item_number = names.C.RN if item_type == names.C.ROW else names.C.CN
    item_span = names.C.RS if item_type == names.C.ROW else names.C.CS

    summary_key = names.C.NR if item_type == names.C.ROW else names.C.NC

    number_of_items = 0

    if image.summary is not None:
        category_item = image.summary.get_sub_category(summary_key)
        number_of_items = int(category_item.category_id)

    cells = image.get_annotation(category_names=names.C.CELL)

    for item in range(1, number_of_items + 1):
        cell_item = list(
            filter(lambda x: x.get_sub_category(item_number).category_id == str(item), cells)  # pylint: disable=W0640
        )
        cell_item = list(filter(lambda x: x.get_sub_category(item_span).category_id == "1", cell_item))
        if cell_item:
            ulx = float(
                min([cell.bounding_box.ulx for cell in cell_item if isinstance(cell, ImageAnnotation)])  # type: ignore
            )
            uly = float(
                min([cell.bounding_box.uly for cell in cell_item if isinstance(cell, ImageAnnotation)])  # type: ignore
            )
            lrx = float(
                max([cell.bounding_box.lrx for cell in cell_item if isinstance(cell, ImageAnnotation)])  # type: ignore
            )
            lry = float(
                max([cell.bounding_box.lry for cell in cell_item if isinstance(cell, ImageAnnotation)])  # type: ignore
            )
            item_ann = ImageAnnotation(
                category_id=categories_name_as_key[names.C.ITEM],
                category_name=names.C.ITEM,
                bounding_box=BoundingBox(absolute_coords=True, ulx=ulx, uly=uly, lrx=lrx, lry=lry),
            )
            item_sub_ann = CategoryAnnotation(category_name=item_type)
            item_ann.dump_sub_category("row_col", item_sub_ann, image.image_id)
            image.dump(item_ann)
    return image


def row_col_cell_ids(tiling: List[List[int]]) -> List[Tuple[int, int, int]]:
    """
    Infers absolute rows and columns for every cell from the tiling of a table.

    :param tiling: A list of list of tiling of a table as returned from the :meth:`_tile_table`
    :return: A list of 3-tuples with row number, column number and cell id.
    """
    indices = sorted(
        [(i + 1, j + 1, cell_id) for i, row in enumerate(tiling) for j, cell_id in enumerate(row)], key=lambda x: x[2]
    )
    seen = set()
    rows_col_cell_ids = [(a, b, c) for a, b, c in indices if not (c in seen or seen.add(c))]  # type: ignore

    return rows_col_cell_ids


def pub_to_image_uncur(  # pylint: disable=R0914, R0915
    dp: JsonDict,
    categories_name_as_key: Dict[str, str],
    load_image: bool,
    fake_score: bool,
    rows_and_cols: bool,
) -> Optional[Image]:
    """
    Map a datapoint of annotation structure as given in the Pubtabnet dataset to an Image structure.
    https://github.com/ibm-aur-nlp/PubTabNet

    :param dp: A datapoint in serialized coco format.
    :param categories_name_as_key: A dict of categories, e.g. DatasetCategories.get_categories(name_as_key=True)
    :param load_image: If 'True' it will load image to :attr:`Image.image`
    :param fake_score: If dp does not contain a score, a fake score with uniform random variables in (0,1)
                       will be added.
    :param rows_and_cols: If set to True, synthetic "ITEM" ImageAnnotations will be added.  Each item has a
                          sub-category "row_col" that is equal to "ROW" or "COL".
    :return: Image
    """

    idx = dp.get("imgid")
    if not idx:
        idx = dp.get("table_id", "")

    with MappingContextManager(str(idx) + " is malformed") as transforming_context:
        html = dp["html"]["structure"]["tokens"]
        index_cells = _cell_token(html)
        col_spans = _item_spans(html, index_cells, "colspan")
        row_spans = _item_spans(html, index_cells, "rowspan")
        number_of_rows = len(col_spans)
        number_of_cols = sum(col_spans[0])
        _has_header = html[0] == "<thead>"

        if _has_header:
            end_of_header = _end_of_header(html)

        tiling = tile_table(row_spans, col_spans)

        rows_cols_cell_ids = row_col_cell_ids(tiling)
        col_spans = list(itertools.chain(*col_spans))  # type: ignore
        row_spans = list(itertools.chain(*row_spans))  # type: ignore

    if transforming_context.context_error:
        return None

    with MappingContextManager(idx) as mapping_context:

        max_rs, max_cs = 0, 0
        if idx is None:
            raise ValueError("No valid datapoint external id")

        image = Image(file_name=os.path.split(dp["filename"])[1], location=dp["filename"], external_id=idx)

        if is_file_extension(dp["filename"], ".png"):
            np_image = load_image_from_file(dp["filename"])
        if is_file_extension(dp["filename"], ".pdf"):
            pdf_bytes = load_bytes_from_pdf_file(dp["filename"])
            np_image = convert_pdf_bytes_to_np_array_v2(pdf_bytes)
            dp = _convert_boxes(dp, np_image.shape[0])

        if load_image and np_image is not None:
            image.image = np_image  # type: ignore
        elif np_image is not None:
            image.set_width_height(np_image.shape[1], np_image.shape[0])  # type: ignore

        table_ann: Optional[ImageAnnotation] = None
        if categories_name_as_key.get(names.C.TAB):
            table_ann = _get_table_annotation(dp, categories_name_as_key[names.C.TAB])
            image.dump(table_ann)

        for idx, value in enumerate(
            zip(rows_cols_cell_ids[::-1], dp["html"]["cells"][::-1], row_spans[::-1], col_spans[::-1])
        ):
            row_col_cell_id = value[0]
            row_number, col_number, cell_id = row_col_cell_id[0], row_col_cell_id[1], row_col_cell_id[2]
            cell = value[1]
            row_span = value[2]
            col_span = value[3]

            if "bbox" in cell:  # empty cells have no box
                ulx, uly, lrx, lry = list(map(float, cell["bbox"]))
                cell_ann = ImageAnnotation(
                    category_name=names.C.CELL,
                    bounding_box=BoundingBox(absolute_coords=True, ulx=ulx, uly=uly, lrx=lrx, lry=lry),
                    category_id=categories_name_as_key[names.C.CELL],
                    score=maybe_get_fake_score(fake_score),
                )
                cell_ann.dump_sub_category(
                    names.C.RN,
                    CategoryAnnotation(category_name=names.C.RN, category_id=row_number),
                    image.image_id,
                )
                cell_ann.dump_sub_category(
                    names.C.CN,
                    CategoryAnnotation(category_name=names.C.CN, category_id=col_number),
                    image.image_id,
                )
                cell_ann.dump_sub_category(
                    names.C.RS,
                    CategoryAnnotation(category_name=names.C.RS, category_id=row_span),  # type: ignore
                    image.image_id,
                )
                cell_ann.dump_sub_category(
                    names.C.CS,
                    CategoryAnnotation(category_name=names.C.CS, category_id=col_span),  # type: ignore
                    image.image_id,
                )

                max_rs = max(max_rs, row_span)  # type: ignore
                max_cs = max(max_cs, col_span)  # type: ignore

                if _has_header:
                    category_name = names.C.HEAD if cell_id <= end_of_header else names.C.BODY
                    cell_ann.dump_sub_category(
                        names.C.HEAD, CategoryAnnotation(category_name=category_name), image.image_id
                    )
                image.dump(cell_ann)
                if table_ann is not None:
                    table_ann.dump_relationship(names.C.CHILD, cell_ann.annotation_id)

        summary_ann = SummaryAnnotation(external_id=image.image_id + "SUMMARY")
        summary_ann.dump_sub_category(
            names.C.NR, CategoryAnnotation(category_name=names.C.NR, category_id=number_of_rows), image.image_id
        )
        summary_ann.dump_sub_category(
            names.C.NC, CategoryAnnotation(category_name=names.C.NC, category_id=number_of_cols), image.image_id
        )
        summary_ann.dump_sub_category(
            names.C.NRS, CategoryAnnotation(category_name=names.C.NRS, category_id=max_rs), image.image_id
        )
        summary_ann.dump_sub_category(
            names.C.NCS, CategoryAnnotation(category_name=names.C.NCS, category_id=max_cs), image.image_id
        )
        image.summary = summary_ann

        if rows_and_cols:
            image = _add_items(image, names.C.ROW, categories_name_as_key)
            image = _add_items(image, names.C.COL, categories_name_as_key)

    if mapping_context.context_error:
        return None
    return image


pub_to_image = cur(pub_to_image_uncur)  # type: ignore  # using cur as decorator is not possible as picking will
# fail in multiprocessing
