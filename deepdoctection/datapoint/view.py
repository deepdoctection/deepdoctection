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

from typing import List, Optional, Union, Dict, Type, Set, overload, Sequence
from copy import copy

import cv2
import numpy as np

from ..datapoint.annotation import ContainerAnnotation, ImageAnnotation, SummaryAnnotation
from ..datapoint.box import BoundingBox
from ..datapoint.image import Image
from ..utils.detection_types import ImageType
from ..utils.settings import CellType, LayoutType, PageType, Relationships, TableType, WordType, ObjectTypes
from ..utils.viz import draw_boxes, interactive_imshow


class ImageAnnotationBaseView(ImageAnnotation):

    base_page: Optional["Page"] = None

    @property
    def bbox(self) -> List[float]:
        if self.image:
            bounding_box = self.image.get_embedding(self.base_page.image_id)
        else:
            bounding_box = self.bounding_box
        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(self.base_page.width, self.base_page.height, absolute_coords=True)
        return bounding_box.to_list(mode="xyxy")

    def __getattr__(self, item) -> Optional[Union[str,int]]:
        if item not in self.get_attribute_names():
            raise AttributeError(f"Attribute {item} is not supported for {type(self)}")
        if item in self.sub_categories:
            sub_cat = self.get_sub_category(item)
            if item != sub_cat.category_name:
                return sub_cat.category_name
            elif isinstance(sub_cat, ContainerAnnotation):
                return sub_cat.value
            else:
                return int(sub_cat.category_id)
        return None

    def get_attribute_names(self) -> Set[str]:
        return {"bbox"}


class Word(ImageAnnotationBaseView):

    def get_attribute_names(self):
        return {word_attr for word_attr in WordType}\
            .union(super().get_attribute_names())\
            .union({Relationships.reading_order})


class Layout(ImageAnnotationBaseView):

    text_container: Optional[ObjectTypes] = None
    layout_types: List[ObjectTypes] = None

    @property
    def words(self) -> List[ImageAnnotationBaseView]:
        if self.category_name != self.text_container:
            text_ids = self.get_relationship(Relationships.child)
            return self.base_page.get_annotation(annotation_ids=text_ids, category_names=self.text_container)
        return [self]

    @property
    def text(self) -> str:
        words_with_reading_order = [word for word in self.words if word.reading_order is not None]
        words_with_reading_order.sort(key=lambda x: x.reading_order)
        return " ".join([word.characters for word in words_with_reading_order])

    def get_attribute_names(self):
        return {"words", "text"}\
            .union(super().get_attribute_names())\
            .union({Relationships.reading_order})


class Cell(Layout):

    def get_attribute_names(self):
        return {cell_attr for cell_attr in CellType}.union(super().get_attribute_names())


class Table(Layout):

    @property
    def cells(self):
        all_relation_ids = self.get_relationship(Relationships.child)
        if self.image is not None:
            cell_anns = self.image.get_annotation(
                annotation_ids=all_relation_ids, category_names=[LayoutType.cell, CellType.header, CellType.body]
            )
            return cell_anns
        return None

    @property
    def html(self):
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
                html_list.insert(html_index, cell.text)
            except ValueError:
                pass

        return "".join(html_list)

    def get_attribute_names(self):
        return {table_attr for table_attr in TableType}.union(super().get_attribute_names()).union({"cell", "html"})


IMAGE_ANNOTATION_TO_LAYOUTS: Dict[LayoutType, Type[Union[Layout, Table, Word]]] = {**{i: Layout for i in LayoutType
                                                                                      if (i not in {LayoutType.table,
                                                                                                LayoutType.word,
                                                                                                LayoutType.cell})},
                                                                                   LayoutType.table: Table, LayoutType.word: Word, LayoutType.cell: Cell}


def ann_obj_view_factory(annotation, text_container, text_block_names) -> ImageAnnotationBaseView:
    layout_class = IMAGE_ANNOTATION_TO_LAYOUTS[annotation.category_name]
    ann_dict = annotation.as_dict()
    layout = layout_class.from_dict(**ann_dict)
    if image_dict := ann_dict.get("image"):
        layout.image = Page.from_dict(**image_dict)
    layout.text_container = text_container
    layout.layout_types = text_block_names
    return layout


class Page(Image):

    layout_types : List[ObjectTypes] = None

    @overload
    def get_annotation(
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        annotation_types: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[ImageAnnotationBaseView]:
        ...

    def get_annotation(
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        annotation_types: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[ImageAnnotationBaseView]:
        return super().get_annotation(category_names,annotation_ids,annotation_types) # type: ignore

    def __getattr__(self, item):
        if item not in self.get_attribute_names():
            raise AttributeError(f"Attribute {item} is not supported for {type(self)}")
        if self.summary is not None:
            if item in self.summary.sub_categories:
                sub_cat = self.summary.get_sub_category(item)
                if item != sub_cat.category_name:
                    return sub_cat.category_name
                elif isinstance(sub_cat, ContainerAnnotation):
                    return sub_cat.value
                else:
                    return int(sub_cat.category_id)
        return None

    @property
    def layouts(self) -> List[ImageAnnotationBaseView]:
        layouts = [layout for layout in self.layout_types if layout!=LayoutType.table]
        return self.get_annotation(category_names=layouts)

    @property
    def tables(self) -> List[ImageAnnotationBaseView]:
        return self.get_annotation(category_names=LayoutType.table)

    @classmethod
    def from_image(cls, image_orig: Image,
                   text_container: ObjectTypes,
                   text_block_names: List[ObjectTypes],
                   base_page: Optional["Page"]=None):
        img_kwargs = image_orig.as_dict()
        page = cls(img_kwargs.get("file_name"), img_kwargs.get("location"), img_kwargs.get("external_id"))
        page._image_id = img_kwargs.get("_image_id")
        if b64_image := img_kwargs.get("_image"):
            page.image = b64_image
        if box_kwargs := img_kwargs.get("_bbox"):
            page._bbox = BoundingBox.from_dict(**box_kwargs)
        if embeddings := img_kwargs.get("embeddings"):
            for image_id, box_dict in embeddings.items():
                page.set_embedding(image_id, BoundingBox.from_dict(**box_dict))
        for ann_dict in img_kwargs.get("annotations"):
            image_ann = ImageAnnotation.from_dict(**ann_dict)
            layout_ann = ann_obj_view_factory(image_ann, text_container, text_block_names)
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
        return page

    def _order_layouts(self) -> List[ImageAnnotationBaseView]:
        layouts_with_order = [layout for layout in self.layouts if layout.reading_order is not None]
        layouts_with_order.sort(key=lambda x: x.reading_order)
        return layouts_with_order

    @property
    def text(self) -> str:
        """
        Get text of all layouts.

        :return: Text string
        """
        text: str = ""
        layouts_with_order = self._order_layouts()
        for layout in layouts_with_order:
            text += "\n" + layout.text

        return text

    @property
    def text_no_line_break(self) -> str:
        text: str = ""
        layouts_with_order = self._order_layouts()
        for layout in layouts_with_order:
            text += " " + layout.text

        return text

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
    def get_attribute_names():
        return {page_attr for page_attr in PageType}.union({"text", "tables", "layouts"})
