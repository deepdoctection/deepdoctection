# -*- coding: utf-8 -*-
# File: xxx.py

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

import os
import numpy as np

import pytest

from dd_core.utils.object_types import LayoutType, WordType, Relationships, BioTag, TokenClassWithTag, TokenClasses
from dd_core.datapoint.box import BoundingBox
from dd_core.datapoint.annotation import ImageAnnotation, CategoryAnnotation, ContainerAnnotation
from dd_core.datapoint.image import Image
from deepdoctection.extern.base import TokenClassResult

import shared_test_utils as stu


@pytest.fixture
def dp_image() -> Image:
    """fixture Image datapoint"""
    img = Image(location="/test/to/path", file_name="test_name")
    img.image = np.ones([400, 600, 3], dtype=np.float32)
    return img


@pytest.fixture
def pdf_path():
    # `stu.asset_path('pdf_file_two_pages')` points to a pdf file
    return stu.asset_path("pdf_file_two_pages")


@pytest.fixture
def image_dir_and_file():
    # `stu.asset_path('sample_image')` points to a png file; we need its directory for dir test
    img_path = stu.asset_path("sample_image")
    img_dir = os.path.dirname(img_path)
    return img_dir, img_path


@pytest.fixture
def image_bytes(image_dir_and_file):
    # Build bytes for the single-image test
    _, img_path = image_dir_and_file
    with open(img_path, "rb") as f:
        return f.read()


@pytest.fixture
def image() -> Image:
    path = stu.asset_path("page_json")
    return Image.from_file(path)


@pytest.fixture
def anns(image: Image):
    # Capture current annotations for building DetectionResults
    return image.get_annotation()


@pytest.fixture
def image_without_anns(image: Image, anns):
    # Remove all annotations from the image to simulate fresh detection
    image.remove(annotation_ids=[ann.annotation_id for ann in anns])
    return image


@pytest.fixture
def word_layout_annotations_for_ordering() -> list[ImageAnnotation]:
    """fixture word_layout_annotations_for_ordering"""
    return [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=110.0, uly=165.0, lrx=130.0, lry=180.0, absolute_coords=True),
            score=0.9,
            category_name=LayoutType.WORD,
            category_id=8,
            service_id="test_service_word",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=140.0, uly=162.0, lrx=180.0, lry=180.0, absolute_coords=True),
            score=0.8,
            category_name=LayoutType.WORD,
            category_id=8,
            service_id="test_service_word",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=320.0, lrx=130.0, lry=340.0, absolute_coords=True),
            score=0.7,
            category_name=LayoutType.WORD,
            category_id=8,
            service_id="test_service_word",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=175.0, uly=320.0, lrx=205.0, lry=340.0, absolute_coords=True),
            score=0.6,
            category_name=LayoutType.WORD,
            category_id=8,
            service_id="test_service_word",
        ),
    ]

@pytest.fixture
def word_sub_cats_for_ordering() -> list[list[CategoryAnnotation]]:
    """fixture word_sub_cats_for_ordering"""
    return [
        [
            ContainerAnnotation(category_name=WordType.CHARACTERS, value="hello"),
            CategoryAnnotation(category_name=WordType.BLOCK, category_id=1),
            CategoryAnnotation(category_name=WordType.TEXT_LINE, category_id=1),
        ],
        [
            ContainerAnnotation(category_name=WordType.CHARACTERS, value="world"),
            CategoryAnnotation(category_name=WordType.BLOCK, category_id=1),
            CategoryAnnotation(category_name=WordType.TEXT_LINE, category_id=2),
        ],
        [
            ContainerAnnotation(category_name=WordType.CHARACTERS, value="bye"),
            CategoryAnnotation(category_name=WordType.BLOCK, category_id=2),
            CategoryAnnotation(category_name=WordType.TEXT_LINE, category_id=2),
        ],
        [
            ContainerAnnotation(category_name=WordType.CHARACTERS, value="world"),
            CategoryAnnotation(category_name=WordType.BLOCK, category_id=2),
            CategoryAnnotation(category_name=WordType.TEXT_LINE, category_id=2),
        ],
    ]

@pytest.fixture
def words_annotations_with_sub_cats(
    word_layout_annotations_for_ordering: list[ImageAnnotation],
    word_sub_cats_for_ordering: list[list[CategoryAnnotation]],
) -> list[ImageAnnotation]:
    """fixture words_annotations_with_sub_cats"""
    for ann, sub_cat_list in zip(word_layout_annotations_for_ordering, word_sub_cats_for_ordering):
        ann.dump_sub_category(WordType.CHARACTERS, sub_cat_list[0])
        ann.dump_sub_category(WordType.BLOCK, sub_cat_list[1])
        ann.dump_sub_category(LayoutType.LINE, sub_cat_list[2])
    return word_layout_annotations_for_ordering


@pytest.fixture
def layout_annotations_for_ordering() -> list[ImageAnnotation]:
    """fixture layout_annotations"""
    return [
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=160.0, lrx=200.0, lry=260.0, absolute_coords=True),
            score=0.9,
            category_name=LayoutType.TITLE,
            category_id=2,
            model_id="test_model",
            service_id="test_service",
        ),
        ImageAnnotation(
            bounding_box=BoundingBox(ulx=100.0, uly=300.0, lrx=250.0, lry=350.0, absolute_coords=True),
            score=0.8,
            category_name=LayoutType.TEXT,
            category_id=1,
            model_id="test_model",
            service_id="test_service",
        ),
    ]


@pytest.fixture
def dp_image_with_layout_and_word_annotations(
    dp_image: Image,
    layout_annotations_for_ordering: list[ImageAnnotation],
    words_annotations_with_sub_cats: list[ImageAnnotation],
) -> Image:
    """
    fixture dp_image_with_layout_and_word_annotations
    """
    layout_anns = layout_annotations_for_ordering
    word_anns = words_annotations_with_sub_cats
    dp_image.dump(layout_anns[0])
    dp_image.dump(layout_anns[1])
    dp_image.dump(word_anns[0])
    layout_anns[0].dump_relationship(Relationships.CHILD, word_anns[0].annotation_id)
    dp_image.dump(word_anns[1])
    layout_anns[0].dump_relationship(Relationships.CHILD, word_anns[1].annotation_id)

    dp_image.dump(word_anns[2])
    layout_anns[1].dump_relationship(Relationships.CHILD, word_anns[2].annotation_id)
    dp_image.dump(word_anns[3])
    layout_anns[1].dump_relationship(Relationships.CHILD, word_anns[3].annotation_id)
    return dp_image


@pytest.fixture
def token_class_result() -> list:
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
        BioTag.OUTSIDE,
        TokenClassWithTag.B_HEADER,
        TokenClassWithTag.B_HEADER,
        TokenClassWithTag.I_HEADER,
        TokenClassWithTag.I_HEADER,
        BioTag.OUTSIDE,
    ]
    semantic_name = [
        TokenClasses.OTHER,
        TokenClasses.HEADER,
        TokenClasses.HEADER,
        TokenClasses.HEADER,
        TokenClasses.HEADER,
        TokenClasses.OTHER,
    ]
    bio_tag = [BioTag.OUTSIDE, BioTag.BEGIN, BioTag.BEGIN, BioTag.INSIDE, BioTag.INSIDE, BioTag.OUTSIDE]
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
        for out in zip(
            uuids, input_ids[0], token_class_predictions, tokens, class_name, semantic_name, bio_tag  # type: ignore
        )
    ]
