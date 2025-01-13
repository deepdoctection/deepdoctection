# -*- coding: utf-8 -*-
# File: view.py

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
Subclasses for ImageAnnotation and Image objects with various properties. These classes
simplify consumption
"""
from __future__ import annotations

from copy import copy
from typing import Any, Mapping, Optional, Sequence, Type, TypedDict, Union, no_type_check

import numpy as np
from typing_extensions import LiteralString

from ..utils.error import AnnotationError, ImageError
from ..utils.logger import LoggingRecord, log_once, logger
from ..utils.settings import (
    CellType,
    LayoutType,
    ObjectTypes,
    PageType,
    Relationships,
    SummaryType,
    TableType,
    TokenClasses,
    WordType,
    get_type,
)
from ..utils.transform import ResizeTransform
from ..utils.types import HTML, AnnotationDict, Chunks, ImageDict, PathLikeOrStr, PixelValues, Text_, csv
from ..utils.viz import draw_boxes, interactive_imshow, viz_handler
from .annotation import CategoryAnnotation, ContainerAnnotation, ImageAnnotation, ann_from_dict
from .box import BoundingBox, crop_box_from_image
from .convert import box_to_point4, point4_to_box
from .image import Image


class ImageAnnotationBaseView(ImageAnnotation):
    """
    Consumption class for having easier access to categories added to an ImageAnnotation.

    ImageAnnotation is a generic class in the sense that different categories might have different
    sub categories collected while running through a pipeline. In order to get properties for a specific
    category one has to understand the internal data structure.

    To circumvent this obstacle `ImageAnnotationBaseView` provides the `__getattr__` so that
    to gather values defined by `ObjectTypes`. To be more precise: A sub class will have attributes either
    defined explicitly by a `@property` or by the set of `get_attribute_names()` . Do not define any attribute
    setter method and regard this class as a view to the super class.

    The class does contain its base page, which mean, that it is possible to retrieve all annotations that have a
    relation.

    base_page: `Page` class instantiated by the lowest hierarchy `Image`
    """

    base_page: Page

    @property
    def bbox(self) -> list[float]:
        """
        Get the bounding box as list and in absolute coordinates of the base page.
        """

        bounding_box = self.get_bounding_box(self.base_page.image_id)

        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(self.base_page.width, self.base_page.height, absolute_coords=True)
        return bounding_box.to_list(mode="xyxy")

    def viz(self, interactive: bool = False) -> Optional[PixelValues]:
        """
        Display the annotation (without any sub-layout elements).

        :param interactive: If set to True will open an interactive image, otherwise it will return a numpy array that
                            can be displayed with e.g. matplotlib
        :return:
        """

        bounding_box = self.get_bounding_box(self.base_page.image_id)
        if self.base_page.image is not None:
            np_image = crop_box_from_image(
                self.base_page.image, bounding_box, self.base_page.width, self.base_page.height
            )

            if interactive:
                interactive_imshow(np_image)
                return None
            return np_image
        raise AnnotationError(f"base_page.image is None for {self.annotation_id}")

    def __getattr__(self, item: str) -> Optional[Union[str, int, list[str], list[ImageAnnotationBaseView]]]:
        """
        Get attributes defined by registered `self.get_attribute_names()` in a multi step process:

        - Unregistered attributes will raise an `AttributeError`.
        - Registered attribute will look for a corresponding sub category. If the sub category does not exist `Null`
          will be returned.
        - If the sub category exists it will return `category_name` provided that the attribute is not equal to the
          `category_name` otherwise
        - Check if the sub category is a `ContainerAnnotation` in which case the `value` will be returned otherwise
          `category_id` will be returned.
        - If nothing works, look at `self.image.summary` if the item exist. Follow the same logic as for ordinary sub
          categories.
        :param item: attribute name
        :return: value according to the logic described above
        """
        if item not in self.get_attribute_names():
            raise AnnotationError(f"Attribute {item} is not supported for {type(self)}")
        if item in self.sub_categories:
            sub_cat = self.get_sub_category(get_type(item))
            if item != sub_cat.category_name:
                return sub_cat.category_name
            if isinstance(sub_cat, ContainerAnnotation):
                return sub_cat.value
            return sub_cat.category_id
        if item in self.relationships:
            relationship_ids = self.get_relationship(get_type(item))
            return self.base_page.get_annotation(annotation_ids=relationship_ids)
        if self.image is not None:
            if item in self.image.summary.sub_categories:
                sub_cat = self.get_summary(get_type(item))
                if item != sub_cat.category_name:
                    return sub_cat.category_name
                if isinstance(sub_cat, ContainerAnnotation):
                    return sub_cat.value
                return sub_cat.category_id
        return None

    def get_attribute_names(self) -> set[str]:
        """
        :return: A set of registered attributes. When sub classing modify this method accordingly.
        """

        # sub categories and summary sub categories are valid attribute names
        attribute_names = {"bbox", "np_image"}.union({cat.value for cat in self.sub_categories})
        if self.image:
            attribute_names = attribute_names.union({cat.value for cat in self.image.summary.sub_categories.keys()})
        return attribute_names

    @classmethod
    def from_dict(cls, **kwargs: AnnotationDict) -> ImageAnnotationBaseView:
        """
        Identical to its base class method for having correct return types. If the base class changes, please
        change this method as well.
        """
        image_ann = ann_from_dict(cls, **kwargs)
        if box_kwargs := kwargs.get("bounding_box"):
            image_ann.bounding_box = BoundingBox.from_dict(**box_kwargs)
        return image_ann


class Word(ImageAnnotationBaseView):
    """
    Word specific subclass of `ImageAnnotationBaseView` modelled by `WordType`.
    """

    def get_attribute_names(self) -> set[str]:
        return (
            set(WordType)
            .union(super().get_attribute_names())
            .union({Relationships.READING_ORDER, Relationships.LAYOUT_LINK})
        )


class Layout(ImageAnnotationBaseView):
    """
    Layout specific subclass of `ImageAnnotationBaseView`. In order check what ImageAnnotation will be wrapped
    into `Layout`, please consult `IMAGE_ANNOTATION_TO_LAYOUTS`.

    text_container: Pass the `LayoutObject` that is supposed to be used for `words`. It is possible that the
                    text_container is equal to `self.category_name`, in which case `words` returns `self`.
    """

    text_container: Optional[ObjectTypes] = None

    @property
    def words(self) -> list[ImageAnnotationBaseView]:
        """
        Get a list of `ImageAnnotationBaseView` objects with `LayoutType` defined by `text_container`.
        It will only select those among all annotations that have an entry in `Relationships.child` .
        """
        if self.category_name != self.text_container:
            text_ids = self.get_relationship(Relationships.CHILD)
            return self.base_page.get_annotation(annotation_ids=text_ids, category_names=self.text_container)
        return [self]

    @property
    def text(self) -> str:
        """
        Text captured within the instance respecting the reading order of each word.
        """
        words = self.get_ordered_words()
        return " ".join([word.characters for word in words])  # type: ignore

    def get_ordered_words(self) -> list[ImageAnnotationBaseView]:
        """Returns a list of words order by reading order. Words with no reading order will not be returned"""
        words_with_reading_order = [word for word in self.words if word.reading_order is not None]
        words_with_reading_order.sort(key=lambda x: x.reading_order)  # type: ignore
        return words_with_reading_order

    @property
    def text_(self) -> Text_:
        """Returns a dict

        `{"text": text string,
          "text_list": list of single words,
          "ann_ids": word annotation ids`,
          "token_classes": token classes,
          "token_tags": token tags,
          "token_class_ids": token class ids,
          "token_tag_ids": token tag ids}`

        """
        words = self.get_ordered_words()
        characters, ann_ids, token_classes, token_tags, token_classes_ids, token_tag_ids = zip(
            *[
                (
                    word.characters,
                    word.annotation_id,
                    word.token_class,
                    word.token_tag,
                    word.get_sub_category(WordType.TOKEN_CLASS).category_id
                    if WordType.TOKEN_CLASS in word.sub_categories
                    else None,
                    word.get_sub_category(WordType.TOKEN_TAG).category_id
                    if WordType.TOKEN_TAG in word.sub_categories
                    else None,
                )
                for word in words
            ]
        )
        return {
            "text": " ".join(characters),
            "words": characters,
            "ann_ids": ann_ids,
            "token_classes": token_classes,
            "token_tags": token_tags,
            "token_class_ids": token_classes_ids,
            "token_tag_ids": token_tag_ids,
        }

    def get_attribute_names(self) -> set[str]:
        return (
            {"words", "text"}
            .union(super().get_attribute_names())
            .union({Relationships.READING_ORDER, Relationships.LAYOUT_LINK})
        )

    def __len__(self) -> int:
        """len of text counted by number of characters"""
        return len(self.text)


class Cell(Layout):
    """
    Cell specific subclass of `ImageAnnotationBaseView` modelled by `CellType`.
    """

    def get_attribute_names(self) -> set[str]:
        return set(CellType).union(super().get_attribute_names())


class Table(Layout):
    """
    Table specific sub class of `ImageAnnotationBaseView` modelled by `TableType`.
    """

    @property
    def cells(self) -> list[Cell]:
        """
        A list of a table cells.
        """
        all_relation_ids = self.get_relationship(Relationships.CHILD)
        cell_anns: list[Cell] = self.base_page.get_annotation(  # type: ignore
            annotation_ids=all_relation_ids,
            category_names=[
                LayoutType.CELL,
                CellType.HEADER,
                CellType.BODY,
                CellType.SPANNING,
            ],
        )
        return cell_anns

    @property
    def column_header_cells(self) -> list[Cell]:
        """
        Retrieve a list of cells that are column headers in the table.

        This property filters and sorts the cells in the table to return only those that are column headers.
        The cells are sorted by their column number.

        :return: A list of `Cell` objects that are column headers.
        """
        all_relation_ids = self.get_relationship(Relationships.CHILD)
        all_cells: list[Cell] = self.base_page.get_annotation(  # type: ignore
            category_names=[LayoutType.CELL, CellType.SPANNING], annotation_ids=all_relation_ids
        )
        headers = list(filter(lambda cell: CellType.COLUMN_HEADER in cell.sub_categories, all_cells))
        headers.sort(key=lambda x: x.column_number)  # type: ignore
        return headers

    @property
    def row_header_cells(self) -> list[Cell]:
        """
        Retrieve a list of cells that are row headers in the table.

        This property filters and sorts the cells in the table to return only those that are row headers.
        The cells are sorted by their column number.

        :return: A list of `Cell` objects that are row headers.
        """
        all_relation_ids = self.get_relationship(Relationships.CHILD)
        all_cells: list[Cell] = self.base_page.get_annotation( # type: ignore
            category_names=[LayoutType.CELL, CellType.SPANNING], annotation_ids=all_relation_ids
        )
        row_header_cells = list(filter(lambda cell: CellType.ROW_HEADER in cell.sub_categories, all_cells))
        row_header_cells.sort(key=lambda x: x.column_number)  # type: ignore
        return row_header_cells

    def kv_header_rows(self, row_number: int) -> Mapping[str, str]:
        """
        For a given row number, returns a dictionary mapping column headers to cell values in that row.

        This method retrieves all cells in the specified row and matches them with their corresponding column headers.
        It then creates a key-value pair where the key is a tuple containing the column number and header text,
        and the value is the cell text.

        :param row_number: The row number for which to retrieve the key-value pairs.
        :return: A dictionary where keys are tuples of (column number, header text) and values are cell texts.

        Example:
        If the table has the following structure:
        | Header1 | Header2 |
        |---------|---------|
        | Value1  | Value2  |
        | Value3  | Value4  |

        Calling kv_header_rows(1) would return:
        {
            (1, 'Header1'): 'Value1',
            (2, 'Header2'): 'Value2'
        }
        """
        all_relation_ids = self.get_relationship(Relationships.CHILD)
        all_cells = self.base_page.get_annotation(
            category_names=[LayoutType.CELL, CellType.SPANNING], annotation_ids=all_relation_ids
        )
        row_cells = list(
            filter(
                lambda c: row_number in (c.row_number, c.row_number + c.row_span), all_cells  # type: ignore
            )
        )
        row_cells.sort(key=lambda c: c.column_number) # type: ignore
        column_header_cells = self.column_header_cells

        kv_dict: Mapping[str, str] = {}
        for cell in row_cells:
            for header in column_header_cells:
                if (cell.column_number == header.column_number and  # type: ignore
                        cell.annotation_id != header.annotation_id):  # type: ignore
                    kv_dict[(header.column_number, header.text)] = cell.text  # type: ignore
                    break
        return kv_dict

    @property
    def rows(self) -> list[ImageAnnotationBaseView]:
        """
        A list of a table rows.
        """
        all_relation_ids = self.get_relationship(Relationships.CHILD)
        row_anns = self.base_page.get_annotation(annotation_ids=all_relation_ids, category_names=[LayoutType.ROW])
        return row_anns

    @property
    def columns(self) -> list[ImageAnnotationBaseView]:
        """
        A list of a table columns.
        """
        all_relation_ids = self.get_relationship(Relationships.CHILD)
        col_anns = self.base_page.get_annotation(annotation_ids=all_relation_ids, category_names=[LayoutType.COLUMN])
        return col_anns

    @property
    def html(self) -> HTML:
        """
        The html representation of the table
        """

        html_list = []
        if TableType.HTML in self.sub_categories:
            ann = self.get_sub_category(TableType.HTML)
            if isinstance(ann, ContainerAnnotation):
                if isinstance(ann.value, list):
                    html_list = copy(ann.value)
        for cell in self.cells:
            try:
                html_index = html_list.index(cell.annotation_id)
                html_list.pop(html_index)
                html_list.insert(html_index, cell.text)
            except ValueError:
                logger.warning(LoggingRecord("html construction not possible", {"annotation_id": cell.annotation_id}))

        return "".join(html_list)

    def get_attribute_names(self) -> set[str]:
        return (
            set(TableType)
            .union(super().get_attribute_names())
            .union({"cells", "rows", "columns", "html", "csv", "text"})
        )

    @property
    def csv(self) -> csv:
        """Returns a csv-style representation of a table as list of lists of string. Cell content of cell with higher
        row or column spans will be shown at the upper left cell tile. All other tiles covered by the cell will be left
        as blank
        """
        cells = self.cells
        table_list = [["" for _ in range(self.number_of_columns)] for _ in range(self.number_of_rows)]  # type: ignore
        for cell in cells:
            if cell.category_name == CellType.SPANNING:
                log_once(
                    "Table has spanning cells. This implies, that the .csv output will not be correct."
                    "To prevent spanning cell table creation set PT.ITEM.FILTER=['table','spanning'] ",
                    "error",
                )
            table_list[cell.row_number - 1][cell.column_number - 1] = (  # type: ignore
                table_list[cell.row_number - 1][cell.column_number - 1] + cell.text + " "  # type: ignore
            )
        return table_list

    def __str__(self) -> str:
        out = " ".join([" ".join(row + ["\n"]) for row in self.csv])
        return out

    @property
    def text(self) -> str:
        try:
            return str(self)
        except (TypeError, AnnotationError):
            return super().text

    @property
    def text_(self) -> Text_:
        cells = self.cells
        if not cells:
            return super().text_
        text: list[str] = []
        words: list[str] = []
        ann_ids: list[str] = []
        token_classes: list[str] = []
        token_tags: list[str] = []
        token_class_ids: list[str] = []
        token_tag_ids: list[str] = []
        for cell in cells:
            text.extend(cell.text_["text"])
            words.extend(cell.text_["words"])
            ann_ids.extend(cell.text_["ann_ids"])
            token_classes.extend(cell.text_["token_classes"])
            token_tags.extend(cell.text_["token_tags"])
            token_class_ids.extend(cell.text_["token_class_ids"])
            token_tag_ids.extend(cell.text_["token_tag_ids"])
        return {
            "text": " ".join(text),
            "words": words,
            "ann_ids": ann_ids,
            "token_classes": token_classes,
            "token_tags": token_tags,
            "token_class_ids": token_class_ids,
            "token_tag_ids": token_tag_ids,
        }

    @property
    def words(self) -> list[ImageAnnotationBaseView]:
        """
        Get a list of `ImageAnnotationBaseView` objects with `LayoutType` defined by `text_container`.
        It will only select those among all annotations that have an entry in `Relationships.child` .
        """
        all_words: list[ImageAnnotationBaseView] = []
        cells = self.cells
        if not cells:
            return super().words
        for cell in cells:
            all_words.extend(cell.words)
        return all_words

    def get_ordered_words(self) -> list[ImageAnnotationBaseView]:
        """Returns a list of words order by reading order. Words with no reading order will not be returned"""
        try:
            cells = self.cells
            all_words = []
            cells.sort(key=lambda x: (x.ROW_NUMBER, x.COLUMN_NUMBER))
            for cell in cells:
                all_words.extend(cell.get_ordered_words())
            return all_words
        except (TypeError, AnnotationError):
            return super().get_ordered_words()


IMAGE_ANNOTATION_TO_LAYOUTS: dict[ObjectTypes, Type[Union[Layout, Table, Word]]] = {
    **{i: Layout for i in LayoutType if (i not in {LayoutType.TABLE, LayoutType.WORD, LayoutType.CELL})},
    LayoutType.TABLE: Table,
    LayoutType.TABLE_ROTATED: Table,
    LayoutType.WORD: Word,
    LayoutType.CELL: Cell,
    CellType.SPANNING: Cell,
    CellType.ROW_HEADER: Cell,
    CellType.COLUMN_HEADER: Cell,
    CellType.PROJECTED_ROW_HEADER: Cell,
}


class ImageDefaults(TypedDict):
    """ImageDefaults"""

    text_container: LayoutType
    floating_text_block_categories: tuple[Union[LayoutType, CellType], ...]
    text_block_categories: tuple[Union[LayoutType, CellType], ...]


IMAGE_DEFAULTS: ImageDefaults = {
    "text_container": LayoutType.WORD,
    "floating_text_block_categories": (
        LayoutType.TEXT,
        LayoutType.TITLE,
        LayoutType.FIGURE,
        LayoutType.LIST,
    ),
    "text_block_categories": (
        LayoutType.TEXT,
        LayoutType.TITLE,
        LayoutType.LIST,
        LayoutType.CELL,
        LayoutType.FIGURE,
        CellType.SPANNING,
    ),
}


@no_type_check
def ann_obj_view_factory(annotation: ImageAnnotation, text_container: ObjectTypes) -> ImageAnnotationBaseView:
    """
    Create an `ImageAnnotationBaseView` sub class given the mapping `IMAGE_ANNOTATION_TO_LAYOUTS` .

    :param annotation: The annotation to transform. Note, that we do not use the input annotation as base class
                       but create a whole new instance.
    :param text_container: `LayoutType` to create a list of `words` and eventually generate `text`
    :return: Transformed annotation
    """

    # We need to handle annotations that are text containers like words
    if annotation.category_name == text_container:
        layout_class = IMAGE_ANNOTATION_TO_LAYOUTS[LayoutType.WORD]
    else:
        layout_class = IMAGE_ANNOTATION_TO_LAYOUTS[annotation.category_name]
    ann_dict = annotation.as_dict()
    layout = layout_class.from_dict(**ann_dict)
    if image_dict := ann_dict.get("image"):
        layout.image = Page.from_dict(**image_dict)
    layout.text_container = text_container
    return layout


class Page(Image):
    """
    Consumer class for its super `Image` class. It comes with some handy `@property` as well as
    custom `__getattr__` to give easier access to various information that are stored in the base class
    as `ImageAnnotation` or `CategoryAnnotation`.

    Its factory function `Page().from_image(image, text_container, text_block_names)` creates for every
    `ImageAnnotation` a corresponding subclass of `ImageAnnotationBaseView` which drives the object towards
    less generic classes with custom attributes that are controlled some `ObjectTypes`.

    top_level_text_block_names: Top level layout objects, e.g. `LayoutType.text` or `LayoutType.table`.

    image_orig: Base image

    text_container: LayoutType to take the text from
    """

    text_container: ObjectTypes
    floating_text_block_categories: list[ObjectTypes]
    image_orig: Image
    _attribute_names: set[str] = {
        "text",
        "chunks",
        "tables",
        "layouts",
        "words",
        "file_name",
        "location",
        "document_id",
        "page_number",
        "angle",
        "figures",
        "residual_layouts",
    }
    include_residual_text_container: bool = True

    def get_annotation(  # type: ignore
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        service_id: Optional[Union[str, Sequence[str]]] = None,
        model_id: Optional[Union[str, Sequence[str]]] = None,
        session_ids: Optional[Union[str, Sequence[str]]] = None,
        ignore_inactive: bool = True,
    ) -> list[ImageAnnotationBaseView]:
        """
        Selection of annotations from the annotation container. Filter conditions can be defined by specifying
        the annotation_id or the category name. (Since only image annotations are currently allowed in the container,
        annotation_type is a redundant filter condition.) Only annotations that have  active = 'True' are
        returned. If more than one condition is provided, only annotations will be returned that satisfy all conditions.
        If no condition is provided, it will return all active annotations.

        Identical to its base class method for having correct return types. If the base class changes, please
        change this method as well.

        :param category_names: A single name or list of names
        :param annotation_ids: A single id or list of ids
        :param service_id: A single service name or list of service names
        :param model_id: A single model name or list of model names
        :param session_ids: A single session id or list of session ids
        :param ignore_inactive: If set to `True` only active annotations are returned.

        :return: A (possibly empty) list of Annotations
        """

        if category_names is not None:
            category_names = (
                (get_type(category_names),)
                if isinstance(category_names, str)
                else tuple(get_type(cat_name) for cat_name in category_names)
            )
        ann_ids = [annotation_ids] if isinstance(annotation_ids, str) else annotation_ids
        service_id = [service_id] if isinstance(service_id, str) else service_id
        model_id = [model_id] if isinstance(model_id, str) else model_id
        session_id = [session_ids] if isinstance(session_ids, str) else session_ids

        if ignore_inactive:
            anns: Union[list[ImageAnnotation], filter[ImageAnnotation]] = filter(lambda x: x.active, self.annotations)
        else:
            anns = self.annotations

        if category_names is not None:
            anns = filter(lambda x: x.category_name in category_names, anns)

        if ann_ids is not None:
            anns = filter(lambda x: x.annotation_id in ann_ids, anns)

        if service_id is not None:
            anns = filter(lambda x: x.generating_service in service_id, anns)

        if model_id is not None:
            anns = filter(lambda x: x.generating_model in model_id, anns)

        if session_id is not None:
            anns = filter(lambda x: x.session_id in session_id, anns)

        return list(anns)  # type: ignore

    def __getattr__(self, item: str) -> Any:
        if item not in self.get_attribute_names():
            raise ImageError(f"Attribute {item} is not supported for {type(self)}")
        if item in self.summary.sub_categories:
            sub_cat = self.summary.get_sub_category(get_type(item))
            if item != sub_cat.category_name:
                return sub_cat.category_name
            if isinstance(sub_cat, ContainerAnnotation):
                return sub_cat.value
            return sub_cat.category_id
        return None

    @property
    def layouts(self) -> list[ImageAnnotationBaseView]:
        """
        A list of a layouts. Layouts are all exactly all floating text block categories
        """
        return self.get_annotation(category_names=self.floating_text_block_categories)

    @property
    def words(self) -> list[ImageAnnotationBaseView]:
        """
        A list of a words. Word are all text containers
        """
        return self.get_annotation(category_names=self.text_container)

    @property
    def tables(self) -> list[ImageAnnotationBaseView]:
        """
        A list of a tables.
        """
        return self.get_annotation(category_names=LayoutType.TABLE)

    @property
    def figures(self) -> list[ImageAnnotationBaseView]:
        """
        A list of a figures.
        """
        return self.get_annotation(category_names=LayoutType.FIGURE)

    @property
    def residual_layouts(self) -> list[ImageAnnotationBaseView]:
        """
        A list of all residual layouts. Residual layouts are all layouts that are
           - not floating text blocks,
           - not text containers,
           - not tables,
           - not figures
           - not cells
           - not rows
           - not columns
        """
        return self.get_annotation(category_names=self._get_residual_layout())

    def _get_residual_layout(self) -> list[LiteralString]:
        layouts = copy(list(self.floating_text_block_categories))
        layouts.extend(
            [
                LayoutType.TABLE,
                LayoutType.FIGURE,
                self.text_container,
                LayoutType.CELL,
                LayoutType.ROW,
                LayoutType.COLUMN,
            ]
        )
        return [layout for layout in LayoutType if layout not in layouts]

    @classmethod
    def from_image(
        cls,
        image_orig: Image,
        text_container: Optional[ObjectTypes] = None,
        floating_text_block_categories: Optional[Sequence[ObjectTypes]] = None,
        include_residual_text_container: bool = True,
        base_page: Optional[Page] = None,
    ) -> Page:
        """
        Factory function for generating a `Page` instance from `image_orig` .

        :param image_orig: `Image` instance to convert
        :param text_container: A LayoutType to get the text from. It will steer the output of `Layout.words`.
        :param floating_text_block_categories: A list of top level layout objects
        :param include_residual_text_container: This will regard synthetic text line annotations as floating text
                                                blocks and therefore incorporate all image annotations of category
                                                `word` when building text strings.
        :param base_page: For top level objects that are images themselves, pass the page that encloses all objects.
                          In doubt, do not populate this value.
        :return:
        """

        if text_container is None:
            text_container = IMAGE_DEFAULTS["text_container"]

        if not floating_text_block_categories:
            floating_text_block_categories = IMAGE_DEFAULTS["floating_text_block_categories"]

        if include_residual_text_container and LayoutType.LINE not in floating_text_block_categories:
            floating_text_block_categories = tuple(floating_text_block_categories) + (LayoutType.LINE,)

        img_kwargs = image_orig.as_dict()
        page = cls(
            img_kwargs.get("file_name"), img_kwargs.get("location"), img_kwargs.get("external_id")  # type: ignore
        )
        page.image_orig = image_orig
        page.page_number = image_orig.page_number
        page.document_id = image_orig.document_id
        if image_orig.image is not None:
            page.image = image_orig.image  # pass image explicitly so
        page._image_id = img_kwargs.get("_image_id")
        if page.image is None:
            if b64_image := img_kwargs.get("_image"):
                page.image = b64_image
        if box_kwargs := img_kwargs.get("_bbox"):
            page._bbox = BoundingBox.from_dict(**box_kwargs)
        if embeddings := img_kwargs.get("embeddings"):
            for image_id, box_dict in embeddings.items():
                page.set_embedding(image_id, BoundingBox.from_dict(**box_dict))
        for ann_dict in img_kwargs.get("annotations", []):
            image_ann = ImageAnnotation.from_dict(**ann_dict)
            layout_ann = ann_obj_view_factory(image_ann, text_container)
            if "image" in ann_dict:
                image_dict = ann_dict["image"]
                if image_dict:
                    image = Image.from_dict(**image_dict)
                    layout_ann.image = cls.from_image(
                        image_orig=image,
                        text_container=text_container,
                        floating_text_block_categories=floating_text_block_categories,
                        include_residual_text_container=include_residual_text_container,
                        base_page=page,
                    )
            layout_ann.base_page = base_page if base_page is not None else page
            page.dump(layout_ann)
        if summary_dict := img_kwargs.get("_summary"):
            page.summary = CategoryAnnotation.from_dict(**summary_dict)
            page.summary.category_name = SummaryType.SUMMARY
        page.floating_text_block_categories = floating_text_block_categories  # type: ignore
        page.text_container = text_container
        page.include_residual_text_container = include_residual_text_container
        return page

    def _order(self, block: str) -> list[ImageAnnotationBaseView]:
        blocks_with_order = [layout for layout in getattr(self, block) if layout.reading_order is not None]
        blocks_with_order.sort(key=lambda x: x.reading_order)
        return blocks_with_order

    def _make_text(self, line_break: bool = True) -> str:
        text: str = ""
        block_with_order = self._order("layouts")
        break_str = "\n" if line_break else " "
        for block in block_with_order:
            text += f"{block.text}{break_str}"
        return text[:-1]

    @property
    def text(self) -> str:
        """
        Get text of all layouts.
        """
        return self._make_text()

    @property
    def text_(self) -> Text_:
        """Returns a dict `{"text": text string,
        "text_list": list of single words,
        "annotation_ids": word annotation ids`"""
        block_with_order = self._order("layouts")
        text: list[str] = []
        words: list[str] = []
        ann_ids: list[str] = []
        token_classes: list[str] = []
        token_tags: list[str] = []
        token_class_ids: list[str] = []
        token_tag_ids: list[str] = []
        for block in block_with_order:
            text.append(block.text_["text"])  # type: ignore
            words.extend(block.text_["words"])  # type: ignore
            ann_ids.extend(block.text_["ann_ids"])  # type: ignore
            token_classes.extend(block.text_["token_classes"])  # type: ignore
            token_tags.extend(block.text_["token_tags"])  # type: ignore
            token_class_ids.extend(block.text_["token_class_ids"])  # type: ignore
            token_tag_ids.extend(block.text_["token_tag_ids"])  # type: ignore
        return {
            "text": " ".join(text),
            "words": words,
            "ann_ids": ann_ids,
            "token_classes": token_classes,
            "token_tags": token_tags,
            "token_class_ids": token_class_ids,
            "token_tag_ids": token_tag_ids,
        }

    def get_layout_context(self, annotation_id: str, context_size: int = 3) -> list[ImageAnnotationBaseView]:
        """For a given `annotation_id` get a list of `ImageAnnotation` that are nearby in terms of reading order.
        For a given context_size it will return all layouts with reading_order between
        reading_order(annoation_id)-context_size and  reading_order(annoation_id)-context_size.

        :param annotation_id: id of central layout element
        :param context_size: number of elements to the left and right of the central element
        :return: list of `ImageAnnotationBaseView` objects
        """
        ann = self.get_annotation(annotation_ids=annotation_id)[0]
        if ann.category_name not in self.floating_text_block_categories:
            raise ImageError(
                f"Cannot get context. Make sure to parametrize this category to a floating text: "
                f"annotation_id: {annotation_id},"
                f"category_name: {ann.category_name}"
            )
        block_with_order = self._order("layouts")
        position = block_with_order.index(ann)
        return block_with_order[
            max(0, position - context_size) : min(position + context_size + 1, len(block_with_order))
        ]

    @property
    def chunks(self) -> Chunks:
        """
        :return: Returns a "chunk" of a layout element or a table as 6-tuple containing

                    - document id
                    - image id
                    - page number
                    - annotation_id
                    - reading order
                    - category name
                    - text string

        """
        block_with_order = self._order("layouts")
        for table in self.tables:
            if table.reading_order:
                block_with_order.append(table)
        all_chunks = []
        for chunk in block_with_order:
            all_chunks.append(
                (
                    self.document_id,
                    self.image_id,
                    self.page_number,
                    chunk.annotation_id,
                    chunk.reading_order,
                    chunk.category_name,
                    chunk.text,
                )
            )
        return all_chunks  # type: ignore

    @property
    def text_no_line_break(self) -> str:
        """
        Get text of all layouts. While `text` will do a line break for each layout block this here will return the
        string in one single line.
        """
        return self._make_text(False)

    def _ann_viz_bbox(self, ann: ImageAnnotationBaseView) -> list[float]:
        """
        Get the bounding box as list and in absolute coordinates of the base page.
        """
        bounding_box = ann.get_bounding_box(self.image_id)

        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(self.width, self.height, absolute_coords=True)
        return bounding_box.to_list(mode="xyxy")

    @no_type_check
    def viz(
        self,
        show_tables: bool = True,
        show_layouts: bool = True,
        show_figures: bool = False,
        show_residual_layouts: bool = False,
        show_cells: bool = True,
        show_table_structure: bool = True,
        show_words: bool = False,
        show_token_class: bool = True,
        ignore_default_token_class: bool = False,
        interactive: bool = False,
        scaled_width: int = 600,
        **debug_kwargs: str,
    ) -> Optional[PixelValues]:
        """
        Display a page with detected bounding boxes of various types.

        **Example:**

                from matplotlib import pyplot as plt

                img = page.viz()
                plt.imshow(img)

        In interactive mode it will display the image in a separate window.

                **Example:**

                page.viz(interactive='True') # will open a new window with the image. Can be closed by pressing 'q'

        :param show_tables: Will display all tables boxes as well as cells, rows and columns
        :param show_layouts: Will display all other layout components.
        :param show_figures: Will display all figures
        :param show_residual_layouts: Will display all residual layouts
        :param show_cells: Will display cells within tables. (Only available if `show_tables=True`)
        :param show_table_structure: Will display rows and columns
        :param show_words: Will display bounding boxes around words labeled with token class and bio tag (experimental)
        :param show_token_class: Will display token class instead of token tags (i.e. token classes with tags)
        :param interactive: If set to True will open an interactive image, otherwise it will return a numpy array that
                            can be displayed differently.
        :param scaled_width: Width of the image to display
        :param ignore_default_token_class: Will ignore displaying word bounding boxes with default or None token class
                                           label
        :return: If `interactive=False` will return a numpy array.
        """

        category_names_list: list[Union[str, None]] = []
        box_stack = []
        cells_found = False

        if self.image is None and interactive:
            logger.warning(
                LoggingRecord("No image provided. Cannot display image in interactive mode", {"page_id": self.image_id})
            )

        if debug_kwargs:
            anns = self.get_annotation(category_names=list(debug_kwargs.keys()))
            for ann in anns:
                box_stack.append(self._ann_viz_bbox(ann))
                category_names_list.append(str(getattr(ann, debug_kwargs[ann.category_name])))

        if show_layouts and not debug_kwargs:
            for item in self.layouts:
                box_stack.append(self._ann_viz_bbox(item))
                category_names_list.append(item.category_name.value)

        if show_figures and not debug_kwargs:
            for item in self.figures:
                box_stack.append(self._ann_viz_bbox(item))
                category_names_list.append(item.category_name.value)

        if show_tables and not debug_kwargs:
            for table in self.tables:
                box_stack.append(self._ann_viz_bbox(table))
                category_names_list.append(LayoutType.TABLE.value)
                if show_cells:
                    for cell in table.cells:
                        if cell.category_name in {
                            LayoutType.CELL,
                            CellType.SPANNING,
                        }:
                            cells_found = True
                            box_stack.append(self._ann_viz_bbox(cell))
                            category_names_list.append(None)
                if show_table_structure:
                    rows = table.rows
                    cols = table.columns
                    for row in rows:
                        box_stack.append(self._ann_viz_bbox(row))
                        category_names_list.append(None)
                    for col in cols:
                        box_stack.append(self._ann_viz_bbox(col))
                        category_names_list.append(None)

        if show_cells and not cells_found and not debug_kwargs:
            for ann in self.get_annotation(category_names=[LayoutType.CELL, CellType.SPANNING]):
                box_stack.append(self._ann_viz_bbox(ann))
                category_names_list.append(None)

        if show_words and not debug_kwargs:
            all_words = []
            for layout in self.layouts:
                all_words.extend(layout.words)
            for table in self.tables:
                all_words.extend(table.words)
            if not all_words:
                all_words = self.get_annotation(category_names=LayoutType.WORD)
            if not ignore_default_token_class:
                for word in all_words:
                    box_stack.append(self._ann_viz_bbox(word))
                    if show_token_class:
                        category_names_list.append(word.token_class.value if word.token_class is not None else None)
                    else:
                        category_names_list.append(word.token_tag.value if word.token_tag is not None else None)
            else:
                for word in all_words:
                    if word.token_class is not None and word.token_class != TokenClasses.OTHER:
                        box_stack.append(self._ann_viz_bbox(word))
                        if show_token_class:
                            category_names_list.append(word.token_class.value if word.token_class is not None else None)
                        else:
                            category_names_list.append(word.token_tag.value if word.token_tag is not None else None)

        if show_residual_layouts and not debug_kwargs:
            for item in self.residual_layouts:
                box_stack.append(item.bbox)
                category_names_list.append(item.category_name.value)

        if self.image is not None:
            scale_fx = scaled_width / self.width
            scaled_height = int(self.height * scale_fx)
            img = viz_handler.resize(self.image, scaled_width, scaled_height, "VIZ")

            if box_stack:
                boxes = np.vstack(box_stack)
                boxes = box_to_point4(boxes)
                resizer = ResizeTransform(self.height, self.width, scaled_height, scaled_width, "VIZ")
                boxes = resizer.apply_coords(boxes)
                boxes = point4_to_box(boxes)
                if show_words:
                    img = draw_boxes(
                        np_image=img,
                        boxes=boxes,
                        category_names_list=category_names_list,
                        font_scale=1.0,
                        rectangle_thickness=4,
                    )
                else:
                    img = draw_boxes(
                        np_image=img, boxes=boxes, category_names_list=category_names_list, show_palette=False
                    )

            if interactive:
                interactive_imshow(img)
                return None
            return img
        return None

    @classmethod
    def get_attribute_names(cls) -> set[str]:
        """
        :return: A set of registered attributes.
        """
        return set(PageType).union(cls._attribute_names)

    @classmethod
    def add_attribute_name(cls, attribute_name: Union[str, ObjectTypes]) -> None:
        """
        Adding a custom attribute name to a Page class.

                **Example:**

                Page.add_attribute_name("foo")

                page = Page.from_image(...)
                print(page.foo)

        Note, that the attribute must be registered as a valid `ObjectTypes`

        :param attribute_name: attribute name to add
        """

        attribute_name = get_type(attribute_name)
        cls._attribute_names.add(attribute_name.value)

    def save(
        self,
        image_to_json: bool = True,
        highest_hierarchy_only: bool = False,
        path: Optional[PathLikeOrStr] = None,
        dry: bool = False,
    ) -> Optional[Union[ImageDict, str]]:
        """
        Export image as dictionary. As numpy array cannot be serialized `image` values will be converted into
        base64 encodings.
        :param image_to_json: If `True` will save the image as b64 encoded string in output
        :param highest_hierarchy_only: If True it will remove all image attributes of ImageAnnotations
        :param path: Path to save the .json file to. If `None` results will be saved in the folder of the original
                     document.
        :param dry: Will run dry, i.e. without saving anything but returning the dict

        :return: optional dict
        """
        return self.image_orig.save(image_to_json, highest_hierarchy_only, path, dry)

    @classmethod
    @no_type_check
    def from_file(
        cls,
        file_path: str,
        text_container: Optional[ObjectTypes] = None,
        floating_text_block_categories: Optional[list[ObjectTypes]] = None,
        include_residual_text_container: bool = True,
    ) -> Page:
        """Reading JSON file and building a `Page` object with given config.
        :param file_path: Path to file
        :param text_container: A LayoutType to get the text from. It will steer the output of `Layout.words`.
        :param floating_text_block_categories: A list of top level layout objects
        :param include_residual_text_container: This will regard synthetic text line annotations as floating text
                                                blocks and therefore incorporate all image annotations of category
                                                `word` when building text strings.
        """
        image = Image.from_file(file_path)
        return cls.from_image(image, text_container, floating_text_block_categories, include_residual_text_container)

    def get_token(self) -> list[Mapping[str, str]]:
        """Return a list of tuples with word and non default token tags"""
        block_with_order = self._order("layouts")
        all_words = []
        for block in block_with_order:
            all_words.extend(block.get_ordered_words())  # type: ignore
        return [
            {"word": word.CHARACTERS, "entity": word.TOKEN_TAG}
            for word in all_words
            if word.TOKEN_TAG not in (TokenClasses.OTHER, None)
        ]

    def __copy__(self) -> Page:
        return self.__class__.from_image(
            self.image_orig,
            self.text_container,
            self.floating_text_block_categories,
            self.include_residual_text_container,
        )
