# -*- coding: utf-8 -*-
# File: page.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Dataclasses Word, Layout, Tables etc. and Page as well as parsing methods.

These data classes are intended for the consumer.
"""

import json
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union, no_type_check

import cv2
import numpy as np

from ..datapoint.annotation import ContainerAnnotation, ImageAnnotation
from ..datapoint.image import Image
from ..utils.detection_types import ImageType, JsonDict, Pathlike
from ..utils.settings import CellType, LayoutType, PageType, Relationships, TableType, WordType
from ..utils.viz import draw_boxes, interactive_imshow
from .convert import convert_b64_to_np_array, convert_np_array_to_b64


def _bounding_box_in_abs_coords(
    annotation: ImageAnnotation, image_id: str, image_width: float, image_height: float
) -> List[float]:
    if annotation.image:
        bounding_box = annotation.image.get_embedding(image_id)
    else:
        bounding_box = annotation.bounding_box
    if not bounding_box.absolute_coords:
        bounding_box = bounding_box.transform(image_width, image_height, absolute_coords=True)
    return bounding_box.to_list(mode="xyxy")


@dataclass
class Word:
    """
    Dataclass for storing word annotations

    :attr:`uuid`: Unique identifier over all items

    :attr:`text`: Text

    :attr:`reading_order`: Index of reading order.

    :attr:`token_class`: token classification

    :attr:`tag`: tag
    """

    uuid: str
    bounding_box: List[float]
    text: str
    reading_order: Optional[int]
    token_class: Optional[str]
    tag: Optional[str]

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> "Word":
        """Generating instance from dict"""
        return cls(**kwargs)

    @classmethod
    def from_annotation(
        cls, annotation: ImageAnnotation, image_id: str, image_width: float, image_height: float
    ) -> "Word":
        """
        Generating an instance from an image annotation

        :param annotation: Image annotation to generate an instance from
        :param image_id: the top level image uuid
        :param image_width: the top level image width
        :param image_height: the top level image height
        :return: A word instance
        """

        text = ""
        reading_order = None
        token = None
        tag = None

        # generating bounding box in terms of full image size and with absolute coords
        bounding_box = _bounding_box_in_abs_coords(annotation, image_id, image_width, image_height)

        if WordType.characters in annotation.sub_categories:
            ann = annotation.get_sub_category(WordType.characters)
            if isinstance(ann, ContainerAnnotation):
                text = str(ann.value)
        if Relationships.reading_order in annotation.sub_categories:
            reading_order = int(annotation.get_sub_category(Relationships.reading_order).category_id)
        if WordType.token_class in annotation.sub_categories:
            token = annotation.get_sub_category(WordType.token_class).category_name
        if WordType.tag in annotation.sub_categories:
            tag = annotation.get_sub_category(WordType.tag).category_name

        return cls(annotation.annotation_id, bounding_box, text, reading_order, token, tag)


def _word_list(annotation: ImageAnnotation, image: Image, text_container: str) -> List[Word]:
    words = []
    if annotation.category_name != text_container:
        text_ids = annotation.get_relationship(Relationships.child)
        word_anns = image.get_annotation(annotation_ids=text_ids, category_names=text_container)
    else:
        word_anns = [annotation]
    for word_ann in word_anns:
        words.append(Word.from_annotation(word_ann, image.image_id, image.width, image.height))
    return words


@dataclass
class Layout:
    """
    Dataclass for storing simple layout items that contain text but dot not have any more structure.

    :attr:`uuid`: Unique identifier over all items

    :attr:`bounding_box`: bounding box in coord terms of the image

    :attr:`layout_type`: The category/label name

    :attr:` reading_order`: Index of reading order.

    :attr:`words`: List of Words

    :attr:`score`: The confidence score, if the layout from a prediction
    """

    uuid: str
    layout_type: str
    reading_order: Optional[int]
    score: Optional[float]
    bounding_box: List[float]
    words: List[Word]

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> "Layout":
        """Generating instance from dict"""
        word_list = []
        words = kwargs.pop("words")
        for word_dict in words:
            word_list.append(Word.from_dict(**word_dict))
        kwargs["words"] = word_list
        return cls(**kwargs)

    @classmethod
    def from_annotation(cls, annotation: ImageAnnotation, dp: Image, text_container: str) -> "Layout":
        """
        Generating an instance from an image annotation

        :param annotation: Image annotation to generate an instance from
        :param dp: The top level image
        :param text_container: Text container to indicate in what image annotation the text can be found
        :return: A layout instance
        """

        reading_order = None

        # generating a list of words
        words = _word_list(annotation, dp, text_container)

        # generating bounding box in terms of full image size and with absolute coords
        bounding_box = _bounding_box_in_abs_coords(annotation, dp.image_id, dp.width, dp.height)

        if Relationships.reading_order in annotation.sub_categories:
            reading_order = int(annotation.get_sub_category(Relationships.reading_order).category_id)

        return cls(
            annotation.annotation_id, annotation.category_name, reading_order, annotation.score, bounding_box, words
        )

    @property
    def text(self) -> str:
        """Text contained in layout instance"""
        words_with_reading_order = [word for word in self.words if word.reading_order is not None]
        words_with_reading_order.sort(key=lambda x: x.reading_order)  # type: ignore
        return " ".join([word.text for word in words_with_reading_order])

    @classmethod
    def from_image(cls, dp: Image, text_container: str) -> "Layout":
        """Generating a Layout object from :class:`Image`. The purpose is to create an object that can store words
        if no Layout information are available."""

        word_anns = dp.get_annotation(category_names=text_container)
        words = []
        for word in word_anns:
            words.append(Word.from_annotation(word, dp.image_id, dp.width, dp.height))

        return cls(
            uuid=dp.image_id,
            layout_type=LayoutType.page,
            reading_order=0,
            score=None,
            bounding_box=[0, 0, dp.width, dp.height],
            words=words,
        )


@dataclass
class Cell(Layout):
    """
    Dataclass that appear in combination with  :class:`Table`.

    :attr:`row_number`: Row position of the cell

    :attr:`col_number`: Column position of the cell

    :attr:`row_span`: Row span position of the cell

    :attr:`col_span`: Column span position of the cell
    """

    row_number: Optional[int]
    col_number: Optional[int]
    row_span: Optional[int]
    col_span: Optional[int]

    @classmethod
    def from_annotation(cls, annotation: ImageAnnotation, dp: Image, text_container: str) -> "Cell":
        """
        Generating an instance from an image annotation

        :param annotation: Image annotation to generate an instance from
        :param dp: The top level image
        :param text_container: Text container to indicate in what image annotation the text can be found
        :return: A cell instance
        """
        reading_order = None
        row_number = None
        col_number = None
        row_span = None
        col_span = None

        # generating a list of words
        words = _word_list(annotation, dp, text_container)

        # generating bounding box in terms of full image size and with absolute coords
        bounding_box = _bounding_box_in_abs_coords(annotation, dp.image_id, dp.width, dp.height)

        if Relationships.reading_order in annotation.sub_categories:
            reading_order = int(annotation.get_sub_category(Relationships.reading_order).category_id)

        if CellType.row_number in annotation.sub_categories:
            row_number = int(annotation.get_sub_category(CellType.row_number).category_id)
        if CellType.column_number in annotation.sub_categories:
            col_number = int(annotation.get_sub_category(CellType.column_number).category_id)
        if CellType.row_span in annotation.sub_categories:
            row_span = int(annotation.get_sub_category(CellType.row_span).category_id)
        if CellType.column_span in annotation.sub_categories:
            col_span = int(annotation.get_sub_category(CellType.column_span).category_id)

        return cls(
            annotation.annotation_id,
            annotation.category_name,
            reading_order,
            annotation.score,
            bounding_box,
            words,
            row_number,
            col_number,
            row_span,
            col_span,
        )


def _get_table_str(cells: List[Cell], number_rows: int, plain: bool = False) -> str:
    output = ""
    for row in range(1, number_rows + 1):
        if not plain:
            output += f"______________ row: {row} ______________\n"
        cells_row = sorted(
            list(filter(lambda x: x.row_number == row, cells)),  # pylint: disable=W0640
            key=lambda x: x.col_number,  # type: ignore
        )

        for cell in cells_row:
            if not plain:
                output += str(cell)
            else:
                output += " " + cell.text
        if plain:
            output += "\n"
    return output


@dataclass
class Table(Layout):
    """
    Dataclass for tables. Tables have cells along rows and columns that, in turn, might contain text.

    :attr:`cells`: List of cells

    :attr:`table_segments`: List of items (i.e. rows or columns)

    :attr:`number_rows`: Total number of rows

    :attr:`number_cols`:  Total number of columns

    :attr:`html`: HTML string representation of the table
    """

    cells: List[Cell]
    table_segments: List[Layout]
    number_rows: Optional[int]
    number_cols: Optional[int]
    html: str

    @classmethod
    def from_annotation(cls, annotation: ImageAnnotation, dp: Image, text_container: str) -> "Table":
        """
        Generating an instance from an image annotation

        :param annotation: Image annotation to generate an instance from
        :param dp: The top level image
        :param text_container: Text container to indicate in what image annotation the text can be found
        :return: A table instance
        """
        cells = []
        html_list: List[str] = []
        reading_order = None
        table_segments = []
        number_rows = None
        number_cols = None

        # generating a list of words
        words = _word_list(annotation, dp, text_container)

        # generating bounding box in terms of full image size and with absolute coords
        bounding_box = _bounding_box_in_abs_coords(annotation, dp.image_id, dp.width, dp.height)

        if Relationships.reading_order in annotation.sub_categories:
            reading_order = int(annotation.get_sub_category(Relationships.reading_order).category_id)

        if TableType.html in annotation.sub_categories:
            ann = annotation.get_sub_category(TableType.html)
            if isinstance(ann, ContainerAnnotation):
                if isinstance(ann.value, list):
                    html_list = ann.value

        # generating cells and html representation
        all_relation_ids = annotation.get_relationship(Relationships.child)
        cell_anns = dp.get_annotation(
            annotation_ids=all_relation_ids, category_names=[LayoutType.cell, CellType.header, CellType.body]
        )
        for cell_ann in cell_anns:
            cell = Cell.from_annotation(cell_ann, dp, text_container)
            cells.append(cell)
            try:
                html_index = html_list.index(cell_ann.annotation_id)
                html_list.pop(html_index)
                html_list.insert(html_index, cell.text)
            except ValueError:
                pass

        html_str = "".join(html_list)

        # generating table segments (i.e. rows and columns)
        table_segm_anns = dp.get_annotation(
            annotation_ids=all_relation_ids, category_names=[LayoutType.row, LayoutType.column]
        )

        for table_segm_ann in table_segm_anns:
            table_segments.append(Layout.from_annotation(table_segm_ann, dp, text_container))

        if annotation.image is not None:
            if annotation.image.summary is not None:
                if (
                    TableType.number_of_rows in annotation.image.summary.sub_categories
                    and TableType.number_of_columns in annotation.image.summary.sub_categories
                ):
                    number_rows = int(annotation.image.summary.get_sub_category(TableType.number_of_rows).category_id)
                    number_cols = int(
                        annotation.image.summary.get_sub_category(TableType.number_of_columns).category_id
                    )
            else:
                if cell_anns:
                    number_rows = max(
                        [int(cell.get_sub_category(CellType.row_number).category_id) for cell in cell_anns]
                    )
                    number_cols = max(
                        [int(cell.get_sub_category(CellType.column_number).category_id) for cell in cell_anns]
                    )

        return cls(
            annotation.annotation_id,
            annotation.category_name,
            reading_order,
            annotation.score,
            bounding_box,
            words,
            cells,
            table_segments,
            number_rows,
            number_cols,
            html_str,
        )

    @property
    def text(self) -> str:
        """Text contained in table instance"""
        if self.number_rows:
            return _get_table_str(self.cells, self.number_rows, True)
        raise ValueError(
            "Table text cannot be printed because not all information for table structure recognition are" " available"
        )

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> "Table":
        """Generating instance from dict"""
        cell_list = []
        table_segment_list = []
        cells = kwargs.pop("cells")
        for cell_dict in cells:
            cell_list.append(Cell.from_dict(**cell_dict))
        table_segments = kwargs.pop("table_segments")
        for table_segment_dict in table_segments:
            table_segment_list.append(Layout.from_dict(**table_segment_dict))
        kwargs["cells"] = cell_list
        kwargs["table_segments"] = table_segment_list
        return cls(**kwargs)

    @property
    def items(self) -> List[Layout]:
        """table segments"""
        return self.table_segments


@dataclass
class Page:
    """
    Dataclass for top level document page. Can be used to convert an image into a page

    :attr:`uuid`: Unique identifier over all items

    :attr:`file_name`: file name. For document with several pages the file name is a concatenation of the document file
                       name and a suffix page_xxx

    :attr:`location`: full path to location of the document

    :attr:`width`: pixel width of page

    :attr:`height`: pixel height of page

    :attr:`language`: language of text content. This can be derived from the :class:`LanguageDetectionService`

    :attr:`document_type`: document type of the page. This can be derived from  :class:`LMSequenceClassifierService`

    :attr:`layouts`: layout information of the page. Layouts can be determined from :class:`ImageLayoutService`

    :attr:`image`: image as b64 encoded string
    """

    uuid: str
    file_name: str
    location: str
    width: float
    height: float
    language: Optional[str]
    document_type: Optional[str]
    layouts: List[Layout]
    image: Optional[str]

    def as_dict(self) -> JsonDict:
        """
        Converts a Page object to a dictionary that can be saved as json object.

        :return: Dictionary with json serializable values
        """
        return asdict(self)

    def get_export(self, save_image: bool = False) -> JsonDict:
        """
        Exporting image as dictionary. Will optionally remove base64 encoded image from export

        :return: Dict that e.g. can be saved to a file.
        """
        export_dict = self.as_dict()
        if not save_image:
            export_dict["image"] = None
        return export_dict

    def save(self, path: Optional[Pathlike] = None, save_image: bool = False) -> None:
        """
        Save a page instance as .json

        :param path: Path to save the .json file to
        :param save_image: If True it will save the image as b64 encoded string
        """
        if isinstance(path, str):
            path = Path(path)
        elif path is None:
            path = Path(self.location)
        suffix = path.suffix
        path_json = path.as_posix().replace(suffix, ".json")
        with open(path_json, "w", encoding="UTF-8") as file:
            json.dump(self.get_export(save_image), file)

    @classmethod
    def from_image(
        cls,
        image: Image,
        text_container: str,
        floating_text_block_names: Optional[List[str]] = None,
        text_block_names: Optional[List[str]] = None,
        text_container_to_text_block: bool = False,
    ) -> "Page":
        """
        Generating an instance from an image

        :param image: The top level image
        :param text_container: Text container to indicate in what image annotation the text can be found
        :param floating_text_block_names: name of image annotation that belong to floating text. These annotations form
                                          the highest hierarchy of text blocks that will ordered to generate a sensible
                                          output of text
        :param text_block_names: name of image annotation that have a relation with text containers (or which might be
                                 text containers themselves).
        :param text_container_to_text_block: Text containers are in general no text blocks and belong to a lower
                                             hierarchy. However, if a text container is not assigned to a text block
                                             you can add it to the text block ordering to ensure that the full text is
                                             part of the subsequent sub process.
        :return: A page instance
        """

        image_str: Optional[str] = None
        layouts: List[Layout] = []
        language = None
        doc_class = None

        if floating_text_block_names is None:
            floating_text_block_names = []
        if text_block_names is None:
            text_block_names = []
        assert isinstance(floating_text_block_names, list)
        assert isinstance(text_block_names, list)

        # page

        if image.image is not None:
            image_str = convert_np_array_to_b64(image.image)

        # all types of layout items and text containers that are not mapped to a layout block
        text_block_anns = image.get_annotation(category_names=text_block_names)
        if text_container_to_text_block:
            floating_text_block_names.append(text_container)
            mapped_text_container = list(
                chain(*[text_block.get_relationship(Relationships.child) for text_block in text_block_anns])
            )
            text_container_anns = image.get_annotation(category_names=text_container)
            text_container_anns = [ann for ann in text_container_anns if ann.annotation_id not in mapped_text_container]
            text_block_anns.extend(text_container_anns)

        for ann in text_block_anns:
            if ann.category_name in {LayoutType.table}:
                layouts.append(Table.from_annotation(ann, image, text_container))
            else:
                layouts.append(Layout.from_annotation(ann, image, text_container))

        if image.summary:
            if PageType.language in image.summary.sub_categories:
                cat_ann = image.summary.get_sub_category(PageType.language)
                if isinstance(cat_ann, ContainerAnnotation):
                    language = str(cat_ann.value)
            if PageType.document_type in image.summary.sub_categories:
                doc_class = image.summary.get_sub_category(PageType.document_type).category_name

        if not text_block_anns and not text_container_to_text_block:
            layouts.append(Layout.from_image(image, text_container))

        return cls(
            image.image_id,
            image.file_name,
            image.location,
            image.width,
            image.height,
            language,
            doc_class,
            layouts,
            image_str,
        )

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> "Page":
        """Generating instance from dict"""
        layout_list = []
        layouts = kwargs.pop("layouts")
        for layout_dict in layouts:
            if layout_dict["layout_type"] == LayoutType.table:
                layout_list.append(Table.from_dict(**layout_dict))
            else:
                layout_list.append(Layout.from_dict(**layout_dict))
        kwargs["layouts"] = layout_list
        return cls(**kwargs)

    def get_text(self, no_line_break: bool = False) -> str:
        """
        Get text of all layouts.
        :param no_line_break: Text will receive a line break for every layout block. Setting argument to `True` will
                              supress this behaviour
        :return: Text string
        """
        text: str = ""
        layouts_with_order = [layout for layout in self.layouts if layout.reading_order is not None]
        layouts_with_order.sort(key=lambda x: x.reading_order)  # type: ignore
        for layout in layouts_with_order:
            if no_line_break:
                text += " " + layout.text
            else:
                text += "\n" + layout.text

        return text

    @property
    def tables(self) -> List[Table]:
        """table from all layouts"""
        return list(filter(lambda x: isinstance(x, Table), self.layouts))  # type:ignore

    @property
    def items(self) -> List[Layout]:
        """All layouts with tables excluded"""
        return list(filter(lambda x: x.layout_type not in [LayoutType.table], self.layouts))

    def viz(
        self,
        show_tables: bool = True,
        show_layouts: bool = True,
        show_cells: bool = True,
        show_table_structure: bool = True,
        show_words: bool = False,
        interactive: bool = False,
    ) -> Optional[ImageType]:
        """
        Display a page detected bounding boxes. One can select bounding boxes of tables or other layout components.

        **Example:**

            .. code-block:: python

                from matplotlib import pyplot as plt

                img = page.viz()
                plt.imshow(img)

        :param show_tables: Will display all tables boxes as well as cells, rows and columns
        :param show_layouts: Will display all other layout components.
        :param show_cells: Will display cells within tables. (Only available if `show_tables=True`)
        :param show_table_structure: Will display rows and columns
        :param show_words: Will display bounding boxes around words labeled with token class and bio tag (experimental)
        :param interactive: If set to True will open an interactive image, otherwise it will return a numpy array that
                            can be displayed differently.
        :return: If interactive will return nothing else a numpy array.
        """

        category_names_list: List[Union[str, None]] = []
        box_stack = []

        if show_layouts:
            for item in self.items:
                box_stack.append(item.bounding_box)
                category_names_list.append(item.layout_type)

        if show_tables:
            for table in self.tables:
                box_stack.append(table.bounding_box)
                category_names_list.append(LayoutType.table)
                if show_cells:
                    for cell in table.cells:
                        box_stack.append(cell.bounding_box)
                        category_names_list.append(None)
                if show_table_structure:
                    for segment_item in table.table_segments:
                        box_stack.append(segment_item.bounding_box)
                        category_names_list.append(None)

        if show_words:
            all_words = []
            for layout in self.layouts:
                all_words.extend(layout.words)
            for word in all_words:
                box_stack.append(word.bounding_box)
                category_names_list.append(str(word.tag) + "-" + str(word.token_class))

        if self.image is not None:
            img = convert_b64_to_np_array(self.image)
            if box_stack:
                boxes = np.vstack(box_stack)
                if show_words:
                    img = draw_boxes(img, boxes, category_names_list, font_scale=0.4, rectangle_thickness=1)
                else:
                    img = draw_boxes(img, boxes, category_names_list)
            img = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

            if interactive:
                interactive_imshow(img)
                return None
            return img
        return None
