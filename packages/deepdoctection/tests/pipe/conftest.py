# -*- coding: utf-8 -*-
# File: conftest.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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
This module provides pytest fixtures for testing various components of the deep learning-based
document analysis framework. The fixtures include assets like sample images, annotations,
detection results, and other related data required for testing functionality.

These fixtures simulate different scenarios such as annotated images, fresh detection capabilities,
and layout-specific or word-level annotations associated with bounding boxes for detailed testing.

"""

import os
from copy import deepcopy
from typing import Any

import pytest

import shared_test_utils as stu
from dd_core.datapoint.annotation import CategoryAnnotation, ContainerAnnotation, ImageAnnotation
from dd_core.datapoint.box import BoundingBox
from dd_core.datapoint.image import Image
from dd_core.utils.object_types import get_type
from dd_core.utils.types import PathLikeOrStr
from deepdoctection.extern.base import DetectionResult, TokenClassResult


@pytest.fixture
def pdf_path() -> PathLikeOrStr:
    """fixture pdf_path"""
    # `stu.asset_path('pdf_file_two_pages')` points to a pdf file
    return stu.asset_path("pdf_file_two_pages")


@pytest.fixture
def image_dir_and_file() -> tuple[str, PathLikeOrStr]:
    """fixture image_dir_and_file"""
    # `stu.asset_path('sample_image')` points to a png file; we need its directory for dir test
    img_path = stu.asset_path("sample_image")
    img_dir = os.path.dirname(img_path)
    return img_dir, img_path


@pytest.fixture
def image_bytes(image_dir_and_file: tuple[str, PathLikeOrStr]) -> bytes:  # pylint:disable=W0621
    """fixture image_bytes"""
    # Build bytes for the single-image test
    _, img_path = image_dir_and_file
    with open(img_path, "rb") as f:
        return f.read()


@pytest.fixture
def image() -> Image:
    """fixture image"""
    path = stu.asset_path("page_json")
    return Image.from_file(os.fspath(path))


@pytest.fixture
def anns(image: Image) -> list[ImageAnnotation]:  # pylint:disable=W0621
    """fixture anns"""
    # Capture current annotations for building DetectionResults
    return image.get_annotation()


@pytest.fixture
def image_without_anns(image: Image, anns: list[ImageAnnotation]) -> Image:  # pylint:disable=W0621
    """fixture image_without_anns"""
    # Remove all annotations from the image to simulate fresh detection
    image.remove(annotation_ids=[ann.annotation_id for ann in anns])
    return image


@pytest.fixture
def cell_detect_results() -> list[list[DetectionResult]]:
    """fixture cell_detect_results"""
    return [
        [
            DetectionResult(box=[20.0, 20.0, 25.0, 30.0], score=0.8, class_id=1, class_name=get_type("column_header")),
            DetectionResult(box=[40.0, 40.0, 50.0, 50.0], score=0.53, class_id=2, class_name=get_type("body")),
        ],
        [DetectionResult(box=[15.0, 20.0, 20.0, 30.0], score=0.4, class_id=1, class_name=get_type("body"))],
    ]


@pytest.fixture
def dp_image_item_stretched(dp_image_tab_cell_item: Image) -> Image:
    """fixture dp_image_tab_cell_item"""
    dp = dp_image_tab_cell_item
    table = dp.get_annotation(category_names=get_type("table"))[0]
    assert isinstance(table, ImageAnnotation)
    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))
    table_embedding_box = table.get_bounding_box(dp.image_id)
    for row in rows:
        assert isinstance(row, ImageAnnotation)
        embedding_box = row.get_bounding_box(dp.image_id)
        embedding_box.ulx = table_embedding_box.ulx + 1.0 / dp.width
        embedding_box.lrx = table_embedding_box.lrx - 1.0 / dp.height

    for col in cols:
        assert isinstance(col, ImageAnnotation)
        embedding_box = col.get_bounding_box(dp.image_id)
        embedding_box.uly = table_embedding_box.uly + 1.0 / dp.width
        embedding_box.lry = table_embedding_box.lry - 1.0 / dp.height

    return deepcopy(dp)


@pytest.fixture
def word_layout_annotations_for_ordering() -> list[ImageAnnotation]:
    """fixture word_layout_annotations_for_ordering"""
    return [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=110.0, uly=165.0, lrx=130.0, lry=180.0, absolute_coords=True),
            score=0.9,
            category_name=get_type("word"),
            category_id=8,
            service_id="test_service_word",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=140.0, uly=162.0, lrx=180.0, lry=180.0, absolute_coords=True),
            score=0.8,
            category_name=get_type("word"),
            category_id=8,
            service_id="test_service_word",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=320.0, lrx=130.0, lry=340.0, absolute_coords=True),
            score=0.7,
            category_name=get_type("word"),
            category_id=8,
            service_id="test_service_word",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=175.0, uly=320.0, lrx=205.0, lry=340.0, absolute_coords=True),
            score=0.6,
            category_name=get_type("word"),
            category_id=8,
            service_id="test_service_word",
        ),
    ]


@pytest.fixture
def word_sub_cats_for_ordering() -> list[list[CategoryAnnotation]]:
    """fixture word_sub_cats_for_ordering"""
    return [
        [
            ContainerAnnotation(category_name=get_type("characters"), value="hello"),
            CategoryAnnotation(category_name=get_type("block"), category_id=1),
            CategoryAnnotation(category_name=get_type("text_line"), category_id=1),
        ],
        [
            ContainerAnnotation(category_name=get_type("characters"), value="world"),
            CategoryAnnotation(category_name=get_type("block"), category_id=1),
            CategoryAnnotation(category_name=get_type("text_line"), category_id=2),
        ],
        [
            ContainerAnnotation(category_name=get_type("characters"), value="bye"),
            CategoryAnnotation(category_name=get_type("block"), category_id=2),
            CategoryAnnotation(category_name=get_type("text_line"), category_id=2),
        ],
        [
            ContainerAnnotation(category_name=get_type("characters"), value="world"),
            CategoryAnnotation(category_name=get_type("block"), category_id=2),
            CategoryAnnotation(category_name=get_type("text_line"), category_id=2),
        ],
    ]


@pytest.fixture
def words_annotations_with_sub_cats(
    word_layout_annotations_for_ordering: list[ImageAnnotation],  # pylint:disable=W0621
    word_sub_cats_for_ordering: list[list[CategoryAnnotation]],  # pylint:disable=W0621
) -> list[ImageAnnotation]:
    """fixture words_annotations_with_sub_cats"""
    for ann, sub_cat_list in zip(word_layout_annotations_for_ordering, word_sub_cats_for_ordering):
        ann.dump_sub_category(get_type("characters"), sub_cat_list[0])
        ann.dump_sub_category(get_type("block"), sub_cat_list[1])
        ann.dump_sub_category(get_type("line"), sub_cat_list[2])
    return word_layout_annotations_for_ordering


@pytest.fixture
def layout_annotations_for_ordering() -> list[ImageAnnotation]:
    """fixture layout_annotations"""
    return [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=160.0, lrx=200.0, lry=260.0, absolute_coords=True),
            score=0.9,
            category_name=get_type("title"),
            category_id=2,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=300.0, lrx=250.0, lry=350.0, absolute_coords=True),
            score=0.8,
            category_name=get_type("text"),
            category_id=1,
            model_id="test_model",
            service_id="test_service",
        ),
    ]


@pytest.fixture
def dp_image_with_layout_and_word_annotations(
    dp_image: Image,
    layout_annotations_for_ordering: list[ImageAnnotation],  # pylint:disable=W0621
    words_annotations_with_sub_cats: list[ImageAnnotation],  # pylint:disable=W0621
) -> Image:
    """
    fixture dp_image_with_layout_and_word_annotations
    """
    dp_image = deepcopy(dp_image)
    layout_anns = layout_annotations_for_ordering
    word_anns = words_annotations_with_sub_cats
    dp_image.dump(layout_anns[0])
    dp_image.dump(layout_anns[1])
    dp_image.dump(word_anns[0])
    layout_anns[0].dump_relationship(get_type("child"), word_anns[0].annotation_id)
    dp_image.dump(word_anns[1])
    layout_anns[0].dump_relationship(get_type("child"), word_anns[1].annotation_id)

    dp_image.dump(word_anns[2])
    layout_anns[1].dump_relationship(get_type("child"), word_anns[2].annotation_id)
    dp_image.dump(word_anns[3])
    layout_anns[1].dump_relationship(get_type("child"), word_anns[3].annotation_id)
    return dp_image


@pytest.fixture
def dp_image_fully_segmented_fully_tiled(
    dp_image_tab_cell_item: Image,
    row_sub_cats: list[CategoryAnnotation],
    column_sub_cats: list[CategoryAnnotation],
) -> Image:
    """
    fixture datapoint_fully_segmented_when_table_fully_tiled. Note that bounding boxes of row and cols are not adjusted
    """
    dp = deepcopy(dp_image_tab_cell_item)
    table = dp.get_annotation(category_names=get_type("table"))[0]
    assert isinstance(table, ImageAnnotation)
    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))
    table_embedding_box = table.get_bounding_box(dp.image_id)
    for row in rows:
        assert isinstance(row, ImageAnnotation)
        embedding_box = row.get_bounding_box(dp.image_id)
        embedding_box.ulx = table_embedding_box.ulx + 1.0 / dp.width
        embedding_box.lrx = table_embedding_box.lrx - 1.0 / dp.width

    for col in cols:
        assert isinstance(col, ImageAnnotation)
        embedding_box = col.get_bounding_box(dp.image_id)
        embedding_box.uly = table_embedding_box.uly + 1.0 / dp.height
        embedding_box.lry = table_embedding_box.lry - 1.0 / dp.height

    rows = dp.get_annotation(category_names=get_type("row"))
    cols = dp.get_annotation(category_names=get_type("column"))
    for row, col, row_sub_cat, col_sub_cat in zip(rows, cols, row_sub_cats, column_sub_cats):
        row.dump_sub_category(get_type("row_number"), row_sub_cat)
        col.dump_sub_category(get_type("column_number"), col_sub_cat)

    cell_sub_cats = [
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
        (
            CategoryAnnotation(category_name=get_type("row_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_number"), category_id=2, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("row_span"), category_id=1, service_id="dbf4f87c"),
            CategoryAnnotation(category_name=get_type("column_span"), category_id=1, service_id="dbf4f87c"),
        ),
    ]
    cells = dp.get_annotation(category_names=[get_type("cell"), get_type("column_header"), get_type("body")])

    for cell, sub_cats in zip(cells, cell_sub_cats):
        cell.dump_sub_category(get_type("row_number"), sub_cats[0])
        cell.dump_sub_category(get_type("column_number"), sub_cats[1])
        cell.dump_sub_category(get_type("row_span"), sub_cats[2])
        cell.dump_sub_category(get_type("column_span"), sub_cats[3])

    return dp


@pytest.fixture
def token_class_result() -> list[Any]:
    """fixture token_class_result"""
    uuids = [
        "CLS",
        "e9c4b3e7-0b2c-3d45-89f3-db6e3ef864ad",
        "44ef758d-92f5-3f57-b6a3-aa95b9606f70",
        "1413d499-ce19-3a50-861c-7d8c5a7ba772",
        "fd78767a-227d-3c17-83cb-586d24cb0c55",
        "SEP",
    ]
    input_ids = [[101, 9875, 3207, 15630, 8569, 102]]
    token_class_predictions = [0, 1, 1, 2, 2, 0]
    tokens = ["CLS", "hello", "world", "bye", "word", "SEP"]
    class_name = [
        get_type("O"),
        get_type("B-header"),
        get_type("B-header"),
        get_type("I-header"),
        get_type("I-header"),
        get_type("O"),
    ]
    semantic_name = [
        get_type("other"),
        get_type("header"),
        get_type("header"),
        get_type("header"),
        get_type("header"),
        get_type("other"),
    ]
    bio_tag = [get_type("O"), get_type("B"), get_type("B"), get_type("I"), get_type("I"), get_type("O")]
    return [
        TokenClassResult(
            uuid=out[0],
            token_id=out[1],
            class_id=out[2],
            token=out[3],
            class_name=out[4],
            semantic_name=out[5],
            bio_tag=out[6],
        )
        for out in zip(uuids, input_ids[0], token_class_predictions, tokens, class_name, semantic_name, bio_tag)
    ]
