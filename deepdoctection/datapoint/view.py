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
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, no_type_check

import cv2
import numpy as np

from ..utils.detection_types import ImageType, JsonDict, Pathlike
from ..utils.logger import logger
from ..utils.settings import CellType, LayoutType, ObjectTypes, PageType, Relationships, TableType, WordType, get_type
from ..utils.viz import draw_boxes, interactive_imshow
from .annotation import ContainerAnnotation, ImageAnnotation, SummaryAnnotation, ann_from_dict
from .box import BoundingBox
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

    base_page: "Page"

    @property
    def bbox(self) -> List[float]:
        """
        Get the bounding box as list and in absolute coordinates of the base page.
        """
        if self.image:
            bounding_box = self.image.get_embedding(self.base_page.image_id)
        else:
            bounding_box = self.bounding_box
        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(self.base_page.width, self.base_page.height, absolute_coords=True)
        return bounding_box.to_list(mode="xyxy")

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
            raise AttributeError(f"Attribute {item} is not supported for {type(self)}")
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
                    sub_cat = self.image.summary.get_sub_category(get_type(item))
                    if item != sub_cat.category_name:
                        return sub_cat.category_name
                    if isinstance(sub_cat, ContainerAnnotation):
                        return sub_cat.value
                    return int(sub_cat.category_id)
        return None

    def get_attribute_names(self) -> Set[str]:  # pylint: disable=R0201
        """
        :return: A set of registered attributes. When sub classing modify this method accordingly.
        """
        return {"bbox"}

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
        words_with_reading_order = [word for word in self.words if word.reading_order is not None]
        words_with_reading_order.sort(key=lambda x: x.reading_order)  # type: ignore
        return " ".join([word.characters for word in words_with_reading_order])  # type: ignore

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
                logger.warning("html construction not possible due to ValueError in: %s", cell.annotation_id)

        return "".join(html_list)

    def get_attribute_names(self) -> Set[str]:
        return set(TableType).union(super().get_attribute_names()).union({"cells", "rows", "columns", "html"})

    @property
    def csv(self) -> List[List[str]]:
        """Returns a csv-style representation of a table as list of lists of string. Cell content of cell with higher
        row or column spans will be shown at the upper left cell tile. All other tiles covered by the cell will be left
        as blank
        """
        cells = self.cells
        table_list = [["" for _ in range(self.number_of_columns)] for _ in range(self.number_of_rows)]  # type: ignore
        for cell in cells:
            table_list[cell.row_number - 1][cell.column_number - 1] = cell.text  # type: ignore
        return table_list

    def __str__(self) -> str:
        out = " ".join([" ".join(row + ["\n"]) for row in self.csv])
        return out


IMAGE_ANNOTATION_TO_LAYOUTS: Dict[ObjectTypes, Type[Union[Layout, Table, Word]]] = {
    **{i: Layout for i in LayoutType if (i not in {LayoutType.table, LayoutType.word, LayoutType.cell})},
    LayoutType.table: Table,
    LayoutType.word: Word,
    LayoutType.cell: Cell,
    CellType.projected_row_header: Cell,
    CellType.spanning: Cell,
    CellType.row_header: Cell,
    CellType.column_header: Cell,
}

IMAGE_DEFAULTS: Dict[str, Union[LayoutType, Sequence[ObjectTypes]]] = {
    "text_container": LayoutType.word,
    "top_level_text_block_names": [
        LayoutType.table,
        LayoutType.text,
        LayoutType.title,
        LayoutType.figure,
        LayoutType.list,
    ],
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

    text_block_names: layout objects that have associated text

    image_orig: Base image

    text_container: LayoutType to take the text from
    """

    top_level_text_block_names: List[ObjectTypes]
    text_block_names: Optional[List[ObjectTypes]]
    image_orig: Image
    text_container: ObjectTypes

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
            raise AttributeError(f"Attribute {item} is not supported for {type(self)}")
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
        A list of a layouts.
        """
        layouts = [layout for layout in self.top_level_text_block_names if layout != LayoutType.table]
        return self.get_annotation(category_names=layouts)

    @property
    def words(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a words.
        """
        return self.get_annotation(category_names=self.text_container)

    @property
    def residual_words(self) -> List[ImageAnnotationBaseView]:
        """
        A list of a words that have not been assigned to any text block but have a reading order.
        Words having this property appear, once `text_containers_to_text_block=True`.
        """
        if self.text_block_names is None:
            return []
        text_block_anns = self.get_annotation(category_names=self.text_block_names)
        text_ann_ids = list(
            chain(*[text_block.get_relationship(Relationships.child) for text_block in text_block_anns])
        )
        text_container_anns = self.get_annotation(category_names=self.text_container)
        residual_words = [ann for ann in text_container_anns if ann.annotation_id not in text_ann_ids]
        return residual_words

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
        top_level_text_block_names: Optional[List[ObjectTypes]] = None,
        text_block_names: Optional[List[ObjectTypes]] = None,
        base_page: Optional["Page"] = None,
    ) -> "Page":
        """
        Factory function for generating a `Page` instance from `image_orig` .

        :param image_orig: `Image` instance to convert
        :param text_container: A LayoutType to get the text from. It will steer the output of `Layout.words`.
        :param top_level_text_block_names: A list of top level layout objects
        :param text_block_names: name of image annotation that have a relation with text containers (or which might be
                                 text containers themselves). This is only necessary, when residual text_container (e.g.
                                 words that have not been assigned to any text block) should be displayed in `page.text`
        :param base_page: For top level objects that are images themselves, pass the page that encloses all objects.
                          In doubt, do not populate this value.
        :return:
        """

        if text_container is None:
            text_container = IMAGE_DEFAULTS["text_container"]  # type: ignore

        if top_level_text_block_names is None:
            top_level_text_block_names = IMAGE_DEFAULTS["top_level_text_block_names"]  # type: ignore

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
                        image, text_container, top_level_text_block_names, text_block_names, page
                    )
            layout_ann.base_page = base_page if base_page is not None else page
            page.dump(layout_ann)
        if summary_dict := img_kwargs.get("_summary"):
            page.summary = SummaryAnnotation.from_dict(**summary_dict)
        page.top_level_text_block_names = top_level_text_block_names  # type: ignore
        page.text_block_names = text_block_names
        page.text_container = text_container  # type: ignore
        return page

    def _order(self, block: str) -> List[ImageAnnotationBaseView]:
        blocks_with_order = [layout for layout in getattr(self, block) if layout.reading_order is not None]
        if self.residual_words:
            blocks_with_order.extend(self.residual_words)
        blocks_with_order.sort(key=lambda x: x.reading_order)
        return blocks_with_order

    @property
    def text(self) -> str:
        """
        Get text of all layouts.
        """
        text: str = ""
        block_name = "layouts" if self.layouts else "words"
        block_with_order = self._order(block_name)
        linebreak = "\n" if block_name == "layouts" else " "
        for block in block_with_order:
            block_attr = "text" if not isinstance(block, Word) else "characters"
            text += f"{linebreak}{getattr(block, block_attr)}"
        return text

    @property
    def chunks(self) -> List[Tuple[str, str, str, str, str, str]]:
        """
        :return: Returns a "chunk" of a layout element or a table as 6-tuple containing

                    - document id
                    - image id
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
        text: str = ""
        layouts_with_order = self._order("layouts")
        for layout in layouts_with_order:
            text += " " + layout.text  # type: ignore

        return text

    @no_type_check
    def viz(
        self,
        show_tables: bool = True,
        show_layouts: bool = True,
        show_cells: bool = True,
        show_table_structure: bool = True,
        show_words: bool = False,
        show_token_class: bool = True,
        interactive: bool = False,
    ) -> Optional[ImageType]:
        """
        Display a page detected bounding boxes. One can select bounding boxes of tables or other layout components.

        **Example:**

                from matplotlib import pyplot as plt

                img = page.viz()
                plt.imshow(img)

        :param show_tables: Will display all tables boxes as well as cells, rows and columns
        :param show_layouts: Will display all other layout components.
        :param show_cells: Will display cells within tables. (Only available if `show_tables=True`)
        :param show_table_structure: Will display rows and columns
        :param show_words: Will display bounding boxes around words labeled with token class and bio tag (experimental)
        :param show_token_class: Will display token class instead of token tags (i.e. token classes with tags)
        :param interactive: If set to True will open an interactive image, otherwise it will return a numpy array that
                            can be displayed differently.
        :return: If interactive will return nothing else a numpy array.
        """

        category_names_list: List[Union[str, None]] = []
        box_stack = []
        cells_found = False

        if show_layouts:
            for item in self.layouts:
                box_stack.append(item.bbox)
                category_names_list.append(item.category_name)

        if show_tables:
            for table in self.tables:
                box_stack.append(table.bbox)
                category_names_list.append(LayoutType.table)
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

        if show_cells and not cells_found:
            for ann in self.annotations:
                if isinstance(ann, Cell) and ann.active:
                    box_stack.append(ann.bbox)
                    category_names_list.append(None)

        if show_words:
            all_words = []
            for layout in self.layouts:
                all_words.extend(layout.words)
            for word in all_words:
                box_stack.append(word.bbox)
                if show_token_class:
                    category_names_list.append(str(word.token_class).replace("TokenClasses", ""))
                else:
                    category_names_list.append(str(word.token_tag))

        if self.image is not None:
            if box_stack:
                boxes = np.vstack(box_stack)
                if show_words:
                    img = draw_boxes(
                        self.image,
                        boxes,
                        category_names_list,
                        color=(255, 222, 173),
                        font_scale=0.25,
                        rectangle_thickness=1,
                    )
                else:
                    img = draw_boxes(self.image, boxes, category_names_list)
                img = cv2.resize(img, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
            else:
                img = self.image

            if interactive:
                interactive_imshow(img)
                return None
            return img
        return None

    @staticmethod
    def get_attribute_names() -> Set[str]:
        """
        :return: A set of registered attributes.
        """
        return set(PageType).union({"text", "tables", "layouts", "words", "residual_words"})

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
        top_level_text_block_names: Optional[List[ObjectTypes]] = None,
        text_block_names: Optional[List[ObjectTypes]] = None,
    ) -> "Page":
        image = Image.from_file(file_path)
        return cls.from_image(image, text_container, top_level_text_block_names, text_block_names)
