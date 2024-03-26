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

from copy import copy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union, no_type_check

import numpy as np

from ..utils.detection_types import ImageType, JsonDict, Pathlike
from ..utils.error import AnnotationError, ImageError
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import (
    CellType,
    LayoutType,
    ObjectTypes,
    PageType,
    Relationships,
    TableType,
    TokenClasses,
    WordType,
    get_type,
)
from ..utils.viz import draw_boxes, interactive_imshow, viz_handler
from .annotation import ContainerAnnotation, ImageAnnotation, SummaryAnnotation, ann_from_dict
from .box import BoundingBox, crop_box_from_image
from .image import Image
import PyPDF2


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

    base_page: "Page"

    @property
    def bbox(self) -> List[float]:
        """
        Get the bounding box as list and in absolute coordinates of the base page.
        """

        bounding_box = self.get_bounding_box(self.base_page.image_id)

        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(self.base_page.width, self.base_page.height, absolute_coords=True)
        return bounding_box.to_list(mode="xyxy")

    def viz(self, interactive: bool = False) -> Optional[ImageType]:
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

    def __getattr__(self, item: str) -> Optional[Union[str, int, List[str]]]:
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
            return int(sub_cat.category_id)
        if self.image is not None:
            if self.image.summary is not None:
                if item in self.image.summary.sub_categories:
                    sub_cat = self.get_summary(get_type(item))
                    if item != sub_cat.category_name:
                        return sub_cat.category_name
                    if isinstance(sub_cat, ContainerAnnotation):
                        return sub_cat.value
                    return int(sub_cat.category_id)
        return None

    def get_attribute_names(self) -> Set[str]:
        """
        :return: A set of registered attributes. When sub classing modify this method accordingly.
        """

        # sub categories and summary sub categories are valid attribute names
        attribute_names = {"bbox", "np_image"}.union({cat.value for cat in self.sub_categories})
        if self.image:
            if self.image.summary:
                attribute_names = attribute_names.union({cat.value for cat in self.image.summary.sub_categories.keys()})
        return attribute_names

    @classmethod
    def from_dict(cls, **kwargs: JsonDict) -> "ImageAnnotationBaseView":
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

    def get_attribute_names(self) -> Set[str]:
        return set(WordType).union(super().get_attribute_names()).union({Relationships.reading_order})


class Layout(ImageAnnotationBaseView):
    """
    Layout specific subclass of `ImageAnnotationBaseView`. In order check what ImageAnnotation will be wrapped
    into `Layout`, please consult `IMAGE_ANNOTATION_TO_LAYOUTS`.

    text_container: Pass the `LayoutObject` that is supposed to be used for `words`. It is possible that the
                    text_container is equal to `self.category_name`, in which case `words` returns `self`.
    """

    text_container: Optional[ObjectTypes] = None

    @property
    def words(self) -> List[ImageAnnotationBaseView]:
        """
        Get a list of `ImageAnnotationBaseView` objects with `LayoutType` defined by `text_container`.
        It will only select those among all annotations that have an entry in `Relationships.child` .
        """
        if self.category_name != self.text_container:
            text_ids = self.get_relationship(Relationships.child)
            return self.base_page.get_annotation(annotation_ids=text_ids, category_names=self.text_container)
        return [self]

    @property
    def text(self) -> str:
        """
        Text captured within the instance respecting the reading order of each word.
        """
        words = self.get_ordered_words()
        return " ".join([word.characters for word in words])  # type: ignore

    def get_ordered_words(self) -> List[ImageAnnotationBaseView]:
        """Returns a list of words order by reading order. Words with no reading order will not be returned"""
        words_with_reading_order = [word for word in self.words if word.reading_order is not None]
        words_with_reading_order.sort(key=lambda x: x.reading_order)  # type: ignore
        return words_with_reading_order

    @property
    def text_(self) -> Dict[str, Union[str, List[str]]]:
        """Returns a dict `{"text": text string,
        "text_list": list of single words,
        "annotation_ids": word annotation ids`"""
        words = self.get_ordered_words()
        return {
            "text": " ".join([word.characters for word in words]),  # type: ignore
            "text_list": [word.characters for word in words],  # type: ignore
            "annotation_ids": [word.annotation_id for word in words],
        }

    def get_attribute_names(self) -> Set[str]:
        return {"words", "text"}.union(super().get_attribute_names()).union({Relationships.reading_order})

    def __len__(self) -> int:
        """len of text counted by number of characters"""
        return len(self.text)


class Cell(Layout):
    """
    Cell specific subclass of `ImageAnnotationBaseView` modelled by `CellType`.
    """

    def get_attribute_names(self) -> Set[str]:
        return set(CellType).union(super().get_attribute_names())


class Table(Layout):
    """
    Table specific sub class of `ImageAnnotationBaseView` modelled by `TableType`.
    """

    @property
    def cells(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a table cells.
        """
        all_relation_ids = self.get_relationship(Relationships.child)
        cell_anns = self.base_page.get_annotation(
            annotation_ids=all_relation_ids,
            category_names=[
                LayoutType.cell,
                CellType.header,
                CellType.body,
                CellType.projected_row_header,
                CellType.spanning,
                CellType.row_header,
                CellType.column_header,
            ],
        )
        return cell_anns

    @property
    def rows(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a table rows.
        """
        all_relation_ids = self.get_relationship(Relationships.child)
        row_anns = self.base_page.get_annotation(annotation_ids=all_relation_ids, category_names=[LayoutType.row])
        return row_anns

    @property
    def columns(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a table columns.
        """
        all_relation_ids = self.get_relationship(Relationships.child)
        col_anns = self.base_page.get_annotation(annotation_ids=all_relation_ids, category_names=[LayoutType.column])
        return col_anns

    @property
    def html(self) -> str:
        """
        The html representation of the table
        """

        html_list = []
        if TableType.html in self.sub_categories:
            ann = self.get_sub_category(TableType.html)
            if isinstance(ann, ContainerAnnotation):
                if isinstance(ann.value, list):
                    html_list = copy(ann.value)
        for cell in self.cells:
            try:
                html_index = html_list.index(cell.annotation_id)
                html_list.pop(html_index)
                html_list.insert(html_index, cell.text)  # type: ignore
            except ValueError:
                logger.warning(LoggingRecord("html construction not possible", {"annotation_id": cell.annotation_id}))

        return "".join(html_list)

    def get_attribute_names(self) -> Set[str]:
        return (
            set(TableType)
            .union(super().get_attribute_names())
            .union({"cells", "rows", "columns", "html", "csv", "text"})
        )

    @property
    def csv(self) -> List[List[str]]:
        """Returns a csv-style representation of a table as list of lists of string. Cell content of cell with higher
        row or column spans will be shown at the upper left cell tile. All other tiles covered by the cell will be left
        as blank
        """
        cells = self.cells
        table_list = [["" for _ in range(self.number_of_columns)] for _ in range(self.number_of_rows)]  # type: ignore
        for cell in cells:
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
    def text_(self) -> Dict[str, Union[str, List[str]]]:
        cells = self.cells
        if not cells:
            return super().text_
        text_list: List[str] = []
        annotation_id_list: List[str] = []
        for cell in cells:
            text_list.extend(cell.text_["text_list"])  # type: ignore
            annotation_id_list.extend(cell.text_["annotation_ids"])  # type: ignore
        return {
            "text": " ".join([cell.text for cell in cells]),  # type: ignore
            "text_list": text_list,
            "annotation_ids": annotation_id_list,
        }

    @property
    def words(self) -> List[ImageAnnotationBaseView]:
        """
        Get a list of `ImageAnnotationBaseView` objects with `LayoutType` defined by `text_container`.
        It will only select those among all annotations that have an entry in `Relationships.child` .
        """
        all_words: List[ImageAnnotationBaseView] = []
        cells = self.cells
        if not cells:
            return super().words
        for cell in cells:
            all_words.extend(cell.words)  # type: ignore
        return all_words

    def get_ordered_words(self) -> List[ImageAnnotationBaseView]:
        """Returns a list of words order by reading order. Words with no reading order will not be returned"""
        try:
            cells = self.cells
            all_words = []
            cells.sort(key=lambda x: (x.row_number, x.column_number))
            for cell in cells:
                all_words.extend(cell.get_ordered_words())  # type: ignore
            return all_words
        except (TypeError, AnnotationError):
            return super().get_ordered_words()


IMAGE_ANNOTATION_TO_LAYOUTS: Dict[ObjectTypes, Type[Union[Layout, Table, Word]]] = {
    **{i: Layout for i in LayoutType if (i not in {LayoutType.table, LayoutType.word, LayoutType.cell})},
    LayoutType.table: Table,
    LayoutType.table_rotated: Table,
    LayoutType.word: Word,
    LayoutType.cell: Cell,
    CellType.projected_row_header: Cell,
    CellType.spanning: Cell,
    CellType.row_header: Cell,
    CellType.column_header: Cell,
}

IMAGE_DEFAULTS: Dict[str, Union[LayoutType, Sequence[ObjectTypes]]] = {
    "text_container": LayoutType.word,
    "floating_text_block_categories": [
        LayoutType.text,
        LayoutType.title,
        LayoutType.figure,
        LayoutType.list,
    ],
    "text_block_categories": [LayoutType.text, LayoutType.title, LayoutType.figure, LayoutType.list, LayoutType.cell],
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
        layout_class = IMAGE_ANNOTATION_TO_LAYOUTS[LayoutType.word]
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
    floating_text_block_categories: List[ObjectTypes]
    image_orig: Image
    _attribute_names: Set[str] = {
        "text",
        "chunks",
        "tables",
        "layouts",
        "words",
        "file_name",
        "location",
        "document_id",
        "page_number",
    }

    @no_type_check
    def get_annotation(
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        annotation_types: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[ImageAnnotationBaseView]:
        """
        Identical to its base class method for having correct return types. If the base class changes, please
        change this method as well.
        """
        cat_names = [category_names] if isinstance(category_names, (ObjectTypes, str)) else category_names
        if cat_names is not None:
            cat_names = [get_type(cat_name) for cat_name in cat_names]
        ann_ids = [annotation_ids] if isinstance(annotation_ids, str) else annotation_ids
        ann_types = [annotation_types] if isinstance(annotation_types, str) else annotation_types

        anns = filter(lambda x: x.active, self.annotations)

        if ann_types is not None:
            for type_name in ann_types:
                anns = filter(lambda x: isinstance(x, eval(type_name)), anns)  # pylint: disable=W0123, W0640

        if cat_names is not None:
            anns = filter(lambda x: x.category_name in cat_names, anns)

        if ann_ids is not None:
            anns = filter(lambda x: x.annotation_id in ann_ids, anns)

        return list(anns)

    def __getattr__(self, item: str) -> Any:
        if item not in self.get_attribute_names():
            raise ImageError(f"Attribute {item} is not supported for {type(self)}")
        if self.summary is not None:
            if item in self.summary.sub_categories:
                sub_cat = self.summary.get_sub_category(get_type(item))
                if item != sub_cat.category_name:
                    return sub_cat.category_name
                if isinstance(sub_cat, ContainerAnnotation):
                    return sub_cat.value
                return int(sub_cat.category_id)
        return None

    @property
    def layouts(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a layouts. Layouts are all exactly all floating text block categories
        """
        return self.get_annotation(category_names=self.floating_text_block_categories)

    @property
    def words(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a words. Word are all text containers
        """
        return self.get_annotation(category_names=self.text_container)

    @property
    def tables(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a tables.
        """
        return self.get_annotation(category_names=LayoutType.table)

    @classmethod
    def from_image(
        cls,
        image_orig: Image,
        text_container: Optional[ObjectTypes] = None,
        floating_text_block_categories: Optional[Sequence[ObjectTypes]] = None,
        include_residual_text_container: bool = True,
        base_page: Optional["Page"] = None,
    ) -> "Page":
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
            text_container = IMAGE_DEFAULTS["text_container"]  # type: ignore

        if not floating_text_block_categories:
            floating_text_block_categories = copy(IMAGE_DEFAULTS["floating_text_block_categories"])  # type: ignore

        if include_residual_text_container and LayoutType.line not in floating_text_block_categories:  # type: ignore
            floating_text_block_categories.append(LayoutType.line)  # type: ignore

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
                        image, text_container, floating_text_block_categories, base_page=page
                    )
            layout_ann.base_page = base_page if base_page is not None else page
            page.dump(layout_ann)
        if summary_dict := img_kwargs.get("_summary"):
            page.summary = SummaryAnnotation.from_dict(**summary_dict)
        page.floating_text_block_categories = floating_text_block_categories  # type: ignore
        page.text_container = text_container  # type: ignore
        return page

    def _order(self, block: str) -> List[ImageAnnotationBaseView]:
        blocks_with_order = [layout for layout in getattr(self, block) if layout.reading_order is not None]
        blocks_with_order.sort(key=lambda x: x.reading_order)
        return blocks_with_order

    def _make_text(self, line_break: bool = True) -> str:
        text: str = ""
        block_with_order = self._order("layouts")
        break_str = "\n" if line_break else " "
        for block in block_with_order:
            text += f"{block.text}{break_str}"
        return text

    @property
    def text(self) -> str:
        """
        Get text of all layouts.
        """
        return self._make_text()

    @property
    def text_(self) -> Dict[str, Union[str, List[str]]]:
        """Returns a dict `{"text": text string,
        "text_list": list of single words,
        "annotation_ids": word annotation ids`"""
        block_with_order = self._order("layouts")
        text_list: List[str] = []
        annotation_id_list: List[str] = []
        for block in block_with_order:
            text_list.extend(block.text_["text_list"])  # type: ignore
            annotation_id_list.extend(block.text_["annotation_ids"])  # type: ignore
        return {"text": self.text, "text_list": text_list, "annotation_ids": annotation_id_list}

    def get_layout_context(self, annotation_id: str, context_size: int = 3) -> List[ImageAnnotationBaseView]:
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
    def chunks(self) -> List[Tuple[str, str, int, str, str, str, str]]:
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

    @no_type_check
    def viz(
        self,
        show_tables: bool = True,
        show_layouts: bool = True,
        show_cells: bool = True,
        show_table_structure: bool = True,
        show_words: bool = False,
        show_token_class: bool = True,
        ignore_default_token_class: bool = False,
        interactive: bool = False,
        **debug_kwargs: str,
    ) -> Optional[ImageType]:
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
        :param show_cells: Will display cells within tables. (Only available if `show_tables=True`)
        :param show_table_structure: Will display rows and columns
        :param show_words: Will display bounding boxes around words labeled with token class and bio tag (experimental)
        :param show_token_class: Will display token class instead of token tags (i.e. token classes with tags)
        :param interactive: If set to True will open an interactive image, otherwise it will return a numpy array that
                            can be displayed differently.
        :param ignore_default_token_class: Will ignore displaying word bounding boxes with default or None token class
                                           label
        :return: If `interactive=False` will return a numpy array.
        """

        category_names_list: List[Union[str, None]] = []
        box_stack = []
        cells_found = False

        if debug_kwargs:
            anns = self.get_annotation(category_names=list(debug_kwargs.keys()))
            for ann in anns:
                box_stack.append(ann.bbox)
                category_names_list.append(str(getattr(ann, debug_kwargs[ann.category_name])))

        if show_layouts and not debug_kwargs:
            for item in self.layouts:
                box_stack.append(item.bbox)
                category_names_list.append(item.category_name.value)

        if show_tables and not debug_kwargs:
            for table in self.tables:
                box_stack.append(table.bbox)
                category_names_list.append(LayoutType.table.value)
                if show_cells:
                    for cell in table.cells:
                        if cell.category_name in {
                            LayoutType.cell,
                            CellType.projected_row_header,
                            CellType.spanning,
                            CellType.row_header,
                            CellType.column_header,
                        }:
                            cells_found = True
                            box_stack.append(cell.bbox)
                            category_names_list.append(None)
                if show_table_structure:
                    rows = table.rows
                    cols = table.columns
                    for row in rows:
                        box_stack.append(row.bbox)
                        category_names_list.append(None)
                    for col in cols:
                        box_stack.append(col.bbox)
                        category_names_list.append(None)

        if show_cells and not cells_found and not debug_kwargs:
            for ann in self.annotations:
                if isinstance(ann, Cell) and ann.active:
                    box_stack.append(ann.bbox)
                    category_names_list.append(None)

        if show_words and not debug_kwargs:
            all_words = []
            for layout in self.layouts:
                all_words.extend(layout.words)
            for table in self.tables:
                all_words.extend(table.words)
            if not all_words:
                all_words = self.get_annotation(category_names=LayoutType.word)
            if not ignore_default_token_class:
                for word in all_words:
                    box_stack.append(word.bbox)
                    if show_token_class:
                        category_names_list.append(word.token_class.value if word.token_class is not None else None)
                    else:
                        category_names_list.append(word.token_tag.value if word.token_tag is not None else None)
            else:
                for word in all_words:
                    if word.token_class is not None and word.token_class != TokenClasses.other:
                        box_stack.append(word.bbox)
                        if show_token_class:
                            category_names_list.append(word.token_class.value if word.token_class is not None else None)
                        else:
                            category_names_list.append(word.token_tag.value if word.token_tag is not None else None)

        if self.image is not None:
            if box_stack:
                boxes = np.vstack(box_stack)
                if show_words:
                    img = draw_boxes(
                        self.image,
                        boxes,
                        category_names_list,
                        font_scale=1.0,
                        rectangle_thickness=4,
                    )
                else:
                    img = draw_boxes(self.image, boxes, category_names_list)
                scale_fx, scale_fy = 1.3, 1.3
                scaled_width, scaled_height = int(self.width * scale_fx), int(self.height * scale_fy)
                img = viz_handler.resize(img, scaled_width, scaled_height, "VIZ")
            else:
                img = self.image

            if interactive:
                interactive_imshow(img)
                return None
            return img
        return None

    @classmethod
    def get_attribute_names(cls) -> Set[str]:
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
        path: Optional[Pathlike] = None,
        dry: bool = False,
    ) -> Optional[JsonDict]:
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
        floating_text_block_categories: Optional[List[ObjectTypes]] = None,
        include_residual_text_container: bool = True,
    ) -> "Page":
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

    def get_token(self) -> List[Mapping[str, str]]:
        """Return a list of tuples with word and non default token tags"""
        block_with_order = self._order("layouts")
        all_words = []
        for block in block_with_order:
            all_words.extend(block.get_ordered_words())  # type: ignore
        return [
            {"word": word.characters, "entity": word.token_tag}
            for word in all_words
            if word.token_tag not in (TokenClasses.other, None)
        ]


class Document:
    """
    Represents a higher-level concept of a document, potentially encompassing multiple pages.
    This class encapsulates document-wide data and metadata, including multipage entities
    and XFA information.
    """
    def __init__(self, pages: List[Page], 
                 metadata: Dict[str, Any] = None, 
                 dynamic_forms: Optional[Any] = None,
                 ):
        self.pages = pages
        self.metadata = metadata if metadata is not None else {}
        self.dynamic_forms = dynamic_forms
        self.decision_funcs = self.define_multipage_entity_decision_functions()

    
    def define_multipage_entity_decision_functions(self, cfg)
    
    
    def get_multipage_entities(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Identifies and returns multipage entities like tables or paragraphs that span across multiple pages.
        """
        tables = []
        paragraphs = []

        # Implement logic to detect multipage tables and paragraphs
        for i, page in enumerate(self.pages):
            tables.extend(self._detect_multipage_tables(page))
            paragraphs.extend(self._detect_multipage_paragraphs(page))

        # self.multipage_entities.tables = tables
        # self.multipage_entities.paragraphs = paragraphs
        return {
            "tables": tables,
            "paragraphs": paragraphs
        }

    def _detect_multipage_tables(self, page: Page) -> List[Dict[str, Any]]:
        """
        Detects tables that span across multiple pages.
        """
        # Placeholder for table detection logic
        return []

    def _detect_multipage_paragraphs(self, page: Page) -> List[Dict[str, Any]]:
        """
        Detects paragraphs that span across multiple pages.
        """
        # Placeholder for paragraph detection logic
        return []

    
    def read_pdf_bytes(self, pdf_file_path):
        """
        Reads a PDF file and returns its content as bytes.

        Args:
            pdf_file_path (str): The path to the PDF file to be read.

        Returns:
            bytes: The bytes content of the PDF file.
        """
        try:
            with open(pdf_file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pdf_bytes = file.read()  # Read the entire PDF file as bytes
                return pdf_bytes
        except FileNotFoundError:
            print(f"The file {pdf_file_path} was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
    @staticmethod
    def from_pages(pages: List[Page]) -> "Document":
        return Document(pages)

    # Additional methods as needed