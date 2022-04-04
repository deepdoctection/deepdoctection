# -*- coding: utf-8 -*-
# File: conftest.py

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
Module for globally accessible fixtures
"""

from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
from pytest import fixture

from deepdoctection.datapoint import (
    BoundingBox,
    CategoryAnnotation,
    ContainerAnnotation,
    Image,
    ImageAnnotation,
    local_to_global_coords,
)
from deepdoctection.datasets import DatasetCategories
from deepdoctection.extern.base import DetectionResult, TokenClassResult
from deepdoctection.utils.detection_types import ImageType, JsonDict
from deepdoctection.utils.settings import names
from deepdoctection.utils.systools import get_package_path

from .data import Annotations, get_layoutlm_input, get_token_class_result
from .mapper.data import DatapointImage


def get_image_results() -> DatapointImage:
    """
    DatapointImage
    """
    return DatapointImage()


@fixture(name="image_results")
def fixture_image_results() -> DatapointImage:
    """
    DatapointImage
    """
    return DatapointImage()


@fixture(name="path_to_tp_frcnn_yaml")
def fixture_path_to_tp_frcnn_yaml() -> str:
    """
    path to tp frcnn yaml file
    """
    return get_package_path() + "/configs/tp/layout/conf_frcnn_layout.yaml"


@fixture(name="categories")
def fixture_categories() -> Dict[str, str]:
    """
    Categories as Dict
    """
    return {"1": names.C.TEXT, "2": names.C.TITLE, "3": names.C.TAB, "4": names.C.FIG, "5": names.C.LIST}


@fixture(name="dataset_categories")
def fixture_dataset_categories_categories() -> DatasetCategories:
    """
    fixture categories
    """
    _categories = [names.C.TAB, names.C.CELL, names.C.ROW, names.C.COL]
    _sub_categories: JsonDict = {
        names.C.ROW: {names.C.RN: []},
        names.C.COL: {names.C.CN: []},
        names.C.CELL: {names.C.RN: [], names.C.CN: [], names.C.RS: [], names.C.CS: []},
    }
    return DatasetCategories(_categories, _sub_categories)


@fixture(name="np_image")
def fixture_np_image() -> ImageType:
    """
    np_array image
    """
    return np.ones([4, 6, 3], dtype=np.float32)


@fixture(name="path_to_tesseract_yaml")
def fixture_path_to_tesseract_yaml() -> str:
    """
    path to tesseract yaml file
    """
    return get_package_path() + "/deepdoctection/configs/conf_tesseract.yaml"


@fixture(name="dp_image")
def fixture_dp_image() -> Image:
    """fixture Image datapoint"""
    img = Image(location="/test/to/path", file_name="test_name")
    img.image = np.ones([400, 600, 3], dtype=np.float32)
    return deepcopy(img)


@fixture(name="layout_detect_results")
def fixture_layout_detect_results() -> List[DetectionResult]:
    """fixture layout_detect_results"""
    return deepcopy(Annotations().get_layout_detect_results())


@fixture(name="layout_annotations")
def fixture_layout_annotations() -> List[ImageAnnotation]:
    """fixture layout_annotations"""
    return deepcopy(Annotations().get_layout_annotation())


@fixture(name="layout_annotations_for_ordering")
def fixture_layout_annotations_for_ordering() -> List[ImageAnnotation]:
    """fixture layout_annotations"""
    return deepcopy(Annotations().get_layout_ann_for_ordering())


@fixture(name="cell_detect_results")
def fixture_cell_detect_results() -> List[List[DetectionResult]]:
    """fixture cell_detect_results"""
    return deepcopy(Annotations().get_cell_detect_results())


@fixture(name="cell_annotation")
def fixture_cell_annotations() -> List[List[ImageAnnotation]]:
    """fixture cell_annotation"""
    return deepcopy(Annotations().get_cell_annotations())


@fixture(name="dp_image_with_layout_anns")
def fixture_dp_image_with_layout_anns(dp_image: Image, layout_annotations: List[ImageAnnotation]) -> Image:
    """fixture dp_image_with_anns"""
    for img_ann in layout_annotations:
        dp_image.dump(img_ann)
        dp_image.image_ann_to_image(img_ann.annotation_id, True)
    return deepcopy(dp_image)


@fixture(name="global_cell_boxes")
def fixture_global_cell_boxes() -> List[List[BoundingBox]]:
    """fixture global_cell_boxes"""
    return Annotations().get_global_cell_boxes()


@fixture(name="dp_image_tab_cell_item")
def fixture_dp_image_tab_cell_item(dp_image: Image) -> Image:
    """fixture dp_image_tab_cell_item"""
    anns = Annotations().get_layout_annotation(segmentation=True)
    for ann in anns:
        dp_image.dump(ann)
    table = dp_image.get_annotation(category_names=names.C.TAB)[0]
    dp_image.image_ann_to_image(table.annotation_id, True)
    table_anns = dp_image.get_annotation_iter(category_names=[names.C.CELL, names.C.ROW, names.C.COL])
    for ann in table_anns:
        assert isinstance(table.image, Image)
        table.image.dump(ann)
        table.image.image_ann_to_image(ann.annotation_id)
        ann_global_box = local_to_global_coords(
            ann.bounding_box, table.image.get_embedding(dp_image.image_id)  # type: ignore
        )
        assert isinstance(ann.image, Image)
        ann.image.set_embedding(table.annotation_id, ann.bounding_box)  # type: ignore
        ann.image.set_embedding(dp_image.image_id, ann_global_box)
        table.dump_relationship(names.C.CHILD, ann.annotation_id)
    return deepcopy(dp_image)


@fixture(name="dp_image_item_stretched")
def fixture_dp_image_item_stretched(dp_image_tab_cell_item: Image) -> Image:
    """fixture dp_image_tab_cell_item"""
    dp = dp_image_tab_cell_item
    table = dp.get_annotation(category_names=names.C.TAB)[0]
    assert isinstance(table, ImageAnnotation)
    rows = dp.get_annotation_iter(category_names=names.C.ROW)
    cols = dp.get_annotation_iter(category_names=names.C.COL)
    assert isinstance(table.image, Image)
    table_embedding_box = table.image.get_embedding(dp.image_id)
    for row in rows:
        assert isinstance(row, ImageAnnotation)
        embedding_box = row.image.get_embedding(dp.image_id)  # type: ignore
        embedding_box.ulx = table_embedding_box.ulx + 1.0
        embedding_box.lrx = table_embedding_box.lrx - 1.0

    for col in cols:
        assert isinstance(col, ImageAnnotation)
        embedding_box = col.image.get_embedding(dp.image_id)  # type: ignore
        embedding_box.uly = table_embedding_box.uly + 1.0
        embedding_box.lry = table_embedding_box.lry - 1.0

    return deepcopy(dp)


@fixture(name="row_sub_cats")
def fixture_row_sub_cats() -> List[CategoryAnnotation]:
    """fixture row_sub_cats"""
    return deepcopy(Annotations().get_row_sub_cats())


@fixture(name="col_sub_cats")
def fixture_col_sub_cats() -> List[CategoryAnnotation]:
    """fixture col_sub_cats"""
    return deepcopy(Annotations().get_col_sub_cats())


@fixture(name="cell_sub_cats")
def fixture_cell_sub_cats() -> List[
    Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]
]:
    """fixture cell_sub_cats"""
    return deepcopy(Annotations().get_cell_sub_cats())


@fixture(name="word_layout_annotations_for_ordering")
def fixture_word_layout_annotations_for_ordering() -> List[ImageAnnotation]:
    """fixture word_layout_annotations_for_ordering"""
    return deepcopy(Annotations().get_word_layout_annotations_for_ordering())


@fixture(name="word_sub_cats_for_ordering")
def fixture_word_sub_cats_for_ordering() -> List[List[CategoryAnnotation]]:
    """fixture word_sub_cats_for_ordering"""
    return deepcopy(Annotations().get_word_sub_cats_for_ordering())


@fixture(name="words_annotations_with_sub_cats")
def fixture_words_annotations_with_sub_cats(
    word_layout_annotations_for_ordering: List[ImageAnnotation],
    word_sub_cats_for_ordering: List[List[CategoryAnnotation]],
) -> List[ImageAnnotation]:
    """fixture words_annotations_with_sub_cats"""
    for ann, sub_cat_list in zip(word_layout_annotations_for_ordering, word_sub_cats_for_ordering):
        ann.dump_sub_category(names.C.CHARS, sub_cat_list[0])
        ann.dump_sub_category(names.C.BLOCK, sub_cat_list[1])
        ann.dump_sub_category(names.C.LINE, sub_cat_list[2])
    return word_layout_annotations_for_ordering


@fixture(name="dp_image_with_layout_and_word_annotations")
def fixture_dp_image_with_layout_and_word_annotations(
    dp_image: Image,
    layout_annotations_for_ordering: List[ImageAnnotation],
    words_annotations_with_sub_cats: List[ImageAnnotation],
) -> Image:
    """
    fixture dp_image_with_layout_and_word_annotations
    """
    layout_anns = layout_annotations_for_ordering
    word_anns = words_annotations_with_sub_cats
    dp_image.dump(layout_anns[0])
    dp_image.dump(layout_anns[1])
    dp_image.dump(word_anns[0])
    layout_anns[0].dump_relationship(names.C.CHILD, word_anns[0].annotation_id)
    dp_image.dump(word_anns[1])
    layout_anns[0].dump_relationship(names.C.CHILD, word_anns[1].annotation_id)

    dp_image.dump(word_anns[2])
    layout_anns[1].dump_relationship(names.C.CHILD, word_anns[2].annotation_id)
    dp_image.dump(word_anns[3])
    layout_anns[1].dump_relationship(names.C.CHILD, word_anns[3].annotation_id)
    return dp_image


@fixture(name="dp_image_fully_segmented")
def fixture_dp_image_fully_segmented(
    dp_image_item_stretched: Image,
    row_sub_cats: List[CategoryAnnotation],
    col_sub_cats: List[CategoryAnnotation],
    cell_sub_cats: List[Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]],
) -> Image:
    """fixture dp_image_fully_segmented"""

    dp = dp_image_item_stretched
    rows = dp.get_annotation_iter(category_names=names.C.ROW)
    cols = dp.get_annotation_iter(category_names=names.C.COL)
    for row, col, row_sub_cat, col_sub_cat in zip(rows, cols, row_sub_cats, col_sub_cats):
        row.dump_sub_category(names.C.RN, row_sub_cat)
        col.dump_sub_category(names.C.CN, col_sub_cat)

    cells = dp.get_annotation_iter(category_names=[names.C.CELL, names.C.HEAD, names.C.BODY])

    for cell, sub_cats in zip(cells, cell_sub_cats):
        cell.dump_sub_category(names.C.RN, sub_cats[0])
        cell.dump_sub_category(names.C.CN, sub_cats[1])
        cell.dump_sub_category(names.C.RS, sub_cats[2])
        cell.dump_sub_category(names.C.CS, sub_cats[3])

    return deepcopy(dp)


@fixture(name="cell_sub_cats_when_table_fully_tiled")
def fixture_cell_sub_cats_when_table_fully_tiled() -> List[
    Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]
]:
    """fixture cell_sub_cats_when_table_fully_tiled"""
    return deepcopy(Annotations().get_cell_sub_cats_when_table_fully_tiled())


@fixture(name="summary_sub_cats_when_table_fully_tiled")
def fixture_summary_sub_cats_when_table_fully_tiled() -> Tuple[
    CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation
]:
    """fixture summary_sub_cats_when_table_fully_tiled"""
    return deepcopy(Annotations().get_summary_sub_cats_when_table_fully_tiled())


@fixture(name="summary_htab_sub_cat")
def fixture_summary_htab_sub_cat() -> ContainerAnnotation:
    """fixture summary_htab_sub_cat"""
    return deepcopy(Annotations().get_summary_htab_sub_cat())


@fixture(name="dp_image_fully_segmented_fully_tiled")
def fixture_dp_image_fully_segmented_fully_tiled(
    dp_image_item_stretched: Image,
    row_sub_cats: List[CategoryAnnotation],
    col_sub_cats: List[CategoryAnnotation],
    cell_sub_cats_when_table_fully_tiled: List[
        Tuple[CategoryAnnotation, CategoryAnnotation, CategoryAnnotation, CategoryAnnotation]
    ],
) -> Image:
    """
    fixture datapoint_fully_segmented_when_table_fully_tiled. Note that bounding boxes of row and cols are not adjusted
    """

    dp = dp_image_item_stretched
    rows = dp.get_annotation_iter(category_names=names.C.ROW)
    cols = dp.get_annotation_iter(category_names=names.C.COL)
    for row, col, row_sub_cat, col_sub_cat in zip(rows, cols, row_sub_cats, col_sub_cats):
        row.dump_sub_category(names.C.RN, row_sub_cat)
        col.dump_sub_category(names.C.CN, col_sub_cat)

    cell_sub_cats = cell_sub_cats_when_table_fully_tiled
    cells = dp.get_annotation_iter(category_names=[names.C.CELL, names.C.HEAD, names.C.BODY])

    for cell, sub_cats in zip(cells, cell_sub_cats):
        cell.dump_sub_category(names.C.RN, sub_cats[0])
        cell.dump_sub_category(names.C.CN, sub_cats[1])
        cell.dump_sub_category(names.C.RS, sub_cats[2])
        cell.dump_sub_category(names.C.CS, sub_cats[3])

    return deepcopy(dp)


@fixture(name="word_detect_result")
def fixture_word_detect_result() -> List[DetectionResult]:
    """fixture word_detect_result"""
    return Annotations().get_word_detect_results()


@fixture(name="double_word_detect_results")
def fixture_double_word_detect_results() -> List[List[DetectionResult]]:
    """fixture double_word_detect_results"""
    return Annotations().get_double_word_detect_results()


@fixture(name="word_layout_annotation")
def fixture_word_layout_annotation() -> List[ImageAnnotation]:
    """fixture word_layout_annotation"""
    return Annotations().get_word_layout_ann()


@fixture(name="word_layout_annotation_for_matching")
def fixture_word_layout_annotation_for_matching() -> List[ImageAnnotation]:
    """fixture word_layout_annotation_for_matching"""
    return Annotations().get_word_layout_ann_for_matching()


@fixture(name="word_box_global")
def fixture_word_box_global() -> List[BoundingBox]:
    """fixture word_box_global"""
    return Annotations().get_word_box_global()


@fixture(name="dp_image_fully_segmented_unrelated_words")
def fixture_dp_image_fully_segmented_unrelated_words(
    dp_image_fully_segmented_fully_tiled: Image, word_layout_annotation_for_matching: List[ImageAnnotation]
) -> Image:
    """
    fixture dp_image_fully_segmented_unrelated_words. Word annotations are not related to layout detections
    """
    dp = dp_image_fully_segmented_fully_tiled

    for word in word_layout_annotation_for_matching:
        dp.dump(word)
        dp.image_ann_to_image(word.annotation_id, False)
        word.image.set_embedding(dp.image_id, word.bounding_box)  # type: ignore
    return deepcopy(dp)


@fixture(name="row_box_tiling_table")
def fixture_row_box_tiling_table() -> List[BoundingBox]:
    """fixture row_box_tiling_table"""
    return Annotations().get_row_box_tiling_table()


@fixture(name="col_box_tiling_table")
def fixture_col_box_tiling_table() -> List[BoundingBox]:
    """fixture col_box_tiling_table"""
    return Annotations().get_col_box_tiling_table()


@fixture(name="layoutlm_input")
def fixture_textract_response() -> JsonDict:
    """fixture textract_response"""
    return get_layoutlm_input()


@fixture(name="token_class_result")
def fixture_token_class_result() -> List[TokenClassResult]:
    """fixture token_class_result"""
    return get_token_class_result()


@fixture(name="text_lines")
def fixture_text_lines() -> List[Tuple[str, ImageType]]:
    """fixture text_lines"""
    return [
        ("cf234ec9-52cf-4710-94ce-288f0e055091", np.zeros((3, 3, 3), dtype=np.float32)),
        ("cf234ec9-52cf-4710-94ce-288f0e055092", np.zeros((3, 3, 3), dtype=np.float32)),
    ]
