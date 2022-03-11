# -*- coding: utf-8 -*-
# File: doc.py

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
Dataclasses LayoutSegment, derived classes, Page and parsing methods.

These data classes are intended for the consumer. They have no overhead and only contain minimal functionalities for
evaluating the text extractions.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from ..utils.detection_types import ImageType, JsonDict
from ..utils.settings import names
from ..utils.viz import draw_boxes, interactive_imshow
from .convert import convert_b64_to_np_array


@dataclass
class LayoutSegment:
    """
    Dataclass for storing simple layout items that contain text but dot not have any more structure.

    :attr:`uuid`: Unique identifier over all items

    :attr:`bounding_box`: bounding box in coord terms of the image

    :attr:`category`: The category/label name

    :attr:`order`: Index of reading order.

    :attr:`text`: Text

    :attr:`score`: The confidence score, in case the item comes from a prediction
    """

    uuid: str
    bounding_box: List[float]
    category: str
    order: int
    text: str
    score: Optional[float] = field(default_factory=float)


@dataclass
class Cell:
    """
    Dataclass that appear in combination with  :class:`Table`.

    :attr:`uuid`: Unique identifier over all items

    :attr:`bounding_box`: bounding box in coord terms of the image

    :attr:`text`: Text

    :attr:`row_number`: Row position of the cell

    :attr:`col_number`: Column position of the cell

    :attr:`row_span`: Row span position of the cell

    :attr:`col_span`: Column span position of the cell

    :attr:`score`: The confidence score, in case the item comes from a prediction
    """

    uuid: str
    bounding_box: List[float]
    text: str
    row_number: int
    col_number: int
    row_span: int
    col_span: int
    score: Optional[float] = field(default=-1.0)

    def __str__(self) -> str:
        """
        A string output for a cell
        """
        return (
            f"row: {self.row_number}, "
            f"col: {self.col_number}, "
            f"rs: {self.row_span}, "
            f"cs: {self.col_span}, "
            f"text: {self.text} \n"
        )


@dataclass
class TableSegment:
    """
    Dataclass that appear in combination with  :class:`Table`. Either row or column

    :attr:`uuid`: Unique identifier over all items

    :attr:`bounding_box`: bounding box in coord terms of the image

    :attr:`score`: The confidence score, in case the item comes from a prediction
    """

    uuid: str
    bounding_box: List[float]
    category: str
    score: Optional[float] = field(default=-1.0)


@dataclass
class Table:
    """
    Dataclass for tables. Tables have cells along rows and columns that, in turn, might contain text.

    :attr:`uuid`: Unique identifier over all items

    :attr:`bounding_box`: bounding box in coord terms of the image

    :attr:`cells`: List of cells

    :attr:`items`: List of items (i.e. rows or columns)

    :attr:`number_rows`: Total number of rows

    :attr:`number_cols`:  Total number of columns

    :attr:`html`: HTML string representation of the table

    :attr:`score`: The confidence score, in case the item comes from a prediction
    """

    uuid: str
    bounding_box: List[float]
    cells: List[Cell]
    items: List[TableSegment]
    number_rows: int
    number_cols: int
    html: str
    score: Optional[float] = field(default=-1.0)

    def __str__(self) -> str:
        """
        A string output for a table.
        """
        output = ""
        for row in range(1, self.number_rows):  # pylint: disable=W0640
            output += f"______________ row: {row} ______________\n"
            cells_row = sorted(
                list(filter(lambda x: x.row_number == row, self.cells)),  # pylint: disable=W0640
                key=lambda x: x.col_number,
            )

            for cell in cells_row:
                output += str(cell)
        return output


@dataclass
class Page:
    """
    Dataclass for a Page. It contains all DocItems and TableItems as well as the document image

    :attr:`uuid:`: Unique identifier

    :attr:`file_name`: File name of the page. If it's origin is a document it will be the file name of the original
    document concatenated with its page number

    :attr:`doc_items`: List of all DocItems

    :attr:`table_items`: List of all TableItems

    :attr:`image`: image as b64 encoding
    """

    uuid: str
    file_name: str
    width: float
    height: float
    items: List[LayoutSegment] = field(default_factory=list, init=False)
    tables: List[Table] = field(default_factory=list, init=False)
    image: Optional[str] = None

    def get_text(self) -> str:
        """
        Get text of all DocItems.

        :return: Text string
        """
        text: str = ""
        self.items.sort(key=lambda x: x.order)
        for item in self.items:
            text += "\n" + item.text

        return text

    def viz(self, show_tables: bool = True, show_items: bool = True, interactive: bool = False) -> Optional[ImageType]:
        """
        Display a page detected bounding boxes. One can select bounding boxes of tables or other layout components.

        :param show_tables: Will display all tables boxes as well as cells, rows and columns
        :param show_items: Will display all other layout components.
        :param interactive: If set to True will open an interactive image, otherwise it will return a numpy array that
                            can be displayed differently.
        :return: If interactive will return nothing else a numpy array.
        """
        category_names_list = []
        box_stack = []

        if show_items:
            for item in self.items:
                box_stack.append(item.bounding_box)
                category_names_list.append(item.category)

        if show_tables:
            for table in self.tables:
                box_stack.append(table.bounding_box)
                category_names_list.append(names.C.TAB)
                for cell in table.cells:
                    box_stack.append(cell.bounding_box)
                    category_names_list.append(f"({cell.row_number},{cell.col_number})")
                for segment_item in table.items:
                    box_stack.append(segment_item.bounding_box)
                    category_names_list.append(segment_item.category)

        if self.image is not None:
            img = convert_b64_to_np_array(self.image)
            if box_stack:
                boxes = np.vstack(box_stack)
                img = draw_boxes(img, boxes, category_names_list)
            img = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

            if interactive:
                interactive_imshow(img)
                return None
            return img
        return None

    def as_dict(self) -> JsonDict:
        """
        Converts a Page object to a dictionary that can be saved as json object.

        :return: Dictionary with json serializable values
        """
        page_dict = asdict(self)
        if self.image is not None:
            page_dict["image"] = self.image
        return page_dict

    def save(self, path: str) -> None:
        """
        Saves the page object to a json file.

        :param path: Path to save to.
        """

        page_dict = self.as_dict()
        with open(  # pylint: disable=W1514
            os.path.join(path, os.path.splitext(page_dict["file_name"])[0] + ".json"), "w"
        ) as file:
            json.dump(page_dict, file)
