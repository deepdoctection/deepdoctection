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

import json
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Type, Union, no_type_check

import cv2
import numpy as np

from ..utils.detection_types import ImageType, JsonDict, Pathlike
from ..utils.logger import logger
from ..utils.settings import CellType, LayoutType, ObjectTypes, PageType, Relationships, TableType, WordType, get_type
from ..utils.viz import draw_boxes, interactive_imshow
from .annotation import ContainerAnnotation, ImageAnnotation, SummaryAnnotation, ann_from_dict
from .box import BoundingBox
from .convert import convert_np_array_to_b64
from .image import Image


class ImageAnnotationBaseView(ImageAnnotation):
    """
    Consumption class for having easier access to categories added to an ImageAnnotation.

    ImageAnnotation is a generic class in the sense that different categories might have different
    sub categories collected while running through a pipeline. In order to get properties for a specific
    category one has to understand the internal data structure.

    To circumvent this obstacle :class:`ImageAnnotationBaseView` provides the :meth:`__getattr__` so that
    to gather values defined by :class:`ObjectTypes`. To be more precise: A sub class will have attributes either
    defined explicitly by a `@property` or by the set of :meth:`get_attribute_names()` . Do not define any attribute
    setter method and regard this class as a view to the super class.

    The class does contain its base page, which mean, that it is possible to retrieve all annotations that have a
    relation.

    :param: base_page: `Page` class instantiated by the lowest hierarchy :class:`Image`
    """

    base_page: "Page"

    @property
    def bbox(self) -> List[float]:
        """
        :return: Get the bounding box as list and in absolute coordinates of the base page.
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
    Word specific sub class of `ImageAnnotationBaseView` modelled by `WordType`.
    """

    def get_attribute_names(self) -> Set[str]:
        return set(WordType).union(super().get_attribute_names()).union({Relationships.reading_order})


class Layout(ImageAnnotationBaseView):
    """
    Layout specific sub class of `ImageAnnotationBaseView`. In order check what ImageAnnotation will be wrapped
    into :class:`Layout`, please consult `IMAGE_ANNOTATION_TO_LAYOUTS`.

    :param: Pass the `LayoutObject` that is supposed to be used for :attr:`words`. It is possible that the
            text_container is equal to `self.category_name`, in which case :attr:`words` returns `self`.
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
        :return: Text captured within the instance respecting the reading order of each word.
        """
        words_with_reading_order = [word for word in self.words if word.reading_order is not None]
        words_with_reading_order.sort(key=lambda x: x.reading_order)  # type: ignore
        return " ".join([word.characters for word in words_with_reading_order])  # type: ignore

    def get_attribute_names(self) -> Set[str]:
        return {"words", "text"}.union(super().get_attribute_names()).union({Relationships.reading_order})


class Cell(Layout):
    """
    Cell specific sub class of `ImageAnnotationBaseView` modelled by `CellType`.
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
        :return: A list of a table cells.
        """
        all_relation_ids = self.get_relationship(Relationships.child)
        cell_anns = self.base_page.get_annotation(
            annotation_ids=all_relation_ids, category_names=[LayoutType.cell, CellType.header, CellType.body]
        )
        return cell_anns

    @property
    def rows(self) -> List[ImageAnnotationBaseView]:
        """
        :return: A list of a table rows.
        """
        all_relation_ids = self.get_relationship(Relationships.child)
        row_anns = self.base_page.get_annotation(annotation_ids=all_relation_ids, category_names=[LayoutType.row])
        return row_anns

    @property
    def columns(self) -> List[ImageAnnotationBaseView]:
        """
        :return: A list of a table columns.
        """
        all_relation_ids = self.get_relationship(Relationships.child)
        col_anns = self.base_page.get_annotation(annotation_ids=all_relation_ids, category_names=[LayoutType.column])
        return col_anns

    @property
    def html(self) -> str:
        """
        :return: The html representation of the table
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


IMAGE_ANNOTATION_TO_LAYOUTS: Dict[ObjectTypes, Type[Union[Layout, Table, Word]]] = {
    **{i: Layout for i in LayoutType if (i not in {LayoutType.table, LayoutType.word, LayoutType.cell})},
    LayoutType.table: Table,
    LayoutType.word: Word,
    LayoutType.cell: Cell,
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
    custom :meth:`__getattr__` to give easier access to various information that are stored in the base class
    as `ImageAnnotation` or `CategoryAnnotation`.

    Its factory function `Page().from_image(image, text_container, text_block_names)` creates for every
    `ImageAnnotation` a corresponding sub class of `ImageAnnotationBaseView` which drives the object towards
    less generic classes with custom attributes that are controlled some `ObjectTypes`.

    :param layout_types: Top level layout objects, e.g. `LayoutType.text` or `LayoutType.table`.
    """

    layout_types: List[ObjectTypes]
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
        :return: A list of a layouts.
        """
        layouts = [layout for layout in self.layout_types if layout != LayoutType.table]
        return self.get_annotation(category_names=layouts)

    @property
    def words(self) -> List[ImageAnnotationBaseView]:
        """
        :return: A list of a words.
        """
        return self.get_annotation(category_names=self.text_container)

    @property
    def tables(self) -> List[ImageAnnotationBaseView]:
        """
        :return: A list of a tables.
        """
        return self.get_annotation(category_names=LayoutType.table)

    @classmethod
    def from_image(
        cls,
        image_orig: Image,
        text_container: ObjectTypes,
        text_block_names: List[ObjectTypes],
        base_page: Optional["Page"] = None,
    ) -> "Page":
        """
        Factory function for generating a `Page` instance from `image_orig` .

        :param image_orig: `Image` instance to convert
        :param text_container: A LayoutType to get the text from. It will steer the output of `Layout.words`.
        :param text_block_names: A list of top level layout objects
        :param base_page: For top level objects that are images themselves, pass the page that encloses all objects.
                          In doubt, do not populate this value.
        :return:
        """
        img_kwargs = image_orig.as_dict()
        page = cls(
            img_kwargs.get("file_name"), img_kwargs.get("location"), img_kwargs.get("external_id")  # type: ignore
        )
        page.image_orig = image_orig
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
                    layout_ann.image = cls.from_image(image, text_container, text_block_names, page)
            layout_ann.base_page = base_page if base_page is not None else page
            page.dump(layout_ann)
        if summary_dict := img_kwargs.get("_summary"):
            page.summary = SummaryAnnotation.from_dict(**summary_dict)
        page.layout_types = text_block_names
        page.text_container = text_container
        return page

    def _order(self, block: str) -> List[ImageAnnotationBaseView]:
        blocks_with_order = [layout for layout in getattr(self, block) if layout.reading_order is not None]
        blocks_with_order.sort(key=lambda x: x.reading_order)
        return blocks_with_order

    @property
    def text(self) -> str:
        """
        Get text of all layouts.

        :return: Text string
        """
        text: str = ""
        block = "layouts" if self.layouts else "words"
        block_content = "text" if block == "layouts" else "characters"
        block_with_order = self._order(block)
        linebreak = "\n" if block == "layouts" else " "
        for block in block_with_order:  # type: ignore
            text += f"{linebreak}{getattr(block, block_content)}"
        return text

    @property
    def text_no_line_break(self) -> str:
        """
        Get text of all layouts. While :attr:`text` will do a line break for each layout block this here will return the
        string in one single line.

        :return: Text string
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
            for item in self.layouts:
                box_stack.append(item.bbox)
                category_names_list.append(item.category_name)

        if show_tables:
            for table in self.tables:
                box_stack.append(table.bbox)
                category_names_list.append(LayoutType.table)
                if show_cells:
                    for cell in table.cells:
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

        if show_words:
            all_words = []
            for layout in self.layouts:
                all_words.extend(layout.words)
            for word in all_words:
                box_stack.append(word.bbox)
                category_names_list.append(str(word.tag) + "-" + str(word.token_class))

        if self.image is not None:
            if box_stack:
                boxes = np.vstack(box_stack)
                if show_words:
                    img = draw_boxes(self.image, boxes, category_names_list, font_scale=0.4, rectangle_thickness=1)
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
        return set(PageType).union({"text", "tables", "layouts", "words"})

    def save(
        self,
        image_to_json: bool = True,
        highest_hierarchy_only: bool = False,
        path: Optional[Pathlike] = None,
        dry: bool = False,
    ) -> None:
        """
        Export image as dictionary. As numpy array cannot be serialized :attr:`image` values will be converted into
        base64 encodings.
        :param image_to_json: If True will save the image as b64 encoded string in output
        :param highest_hierarchy_only: If True it will remove all image attributes of ImageAnnotations
        :param path: Path to save the .json file to
        :param dry: Will run dry, i.e. without saving anything but returning the dict
        :return: Dict that e.g. can be saved to a file.
        """
        if isinstance(path, str):
            path = Path(path)
        elif path is None:
            path = Path(self.image_orig.location)
        suffix = path.suffix
        path_json = path.as_posix().replace(suffix, ".json")
        if highest_hierarchy_only:
            self.image_orig.remove_image_from_lower_hierachy()
        export_dict = self.image_orig.as_dict()
        export_dict["location"] = str(export_dict["location"])
        if image_to_json and self.image_orig.image is not None:
            export_dict["_image"] = convert_np_array_to_b64(self.image_orig.image)
        if dry:
            return None
        with open(path_json, "w", encoding="UTF-8") as file:
            json.dump(export_dict, file)
