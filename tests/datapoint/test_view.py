# -*- coding: utf-8 -*-
# File: test_page.py

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
Testing the module datapoint.page
"""

from numpy import float32, ones
from pytest import mark

from deepdoctection.datapoint.annotation import CategoryAnnotation, ImageAnnotation
from deepdoctection.datapoint.box import BoundingBox
from deepdoctection.datapoint.image import Image
from deepdoctection.datapoint.view import Page
from deepdoctection.utils.settings import LayoutType, Relationships

from ..test_utils import get_test_path
from .conftest import WhiteImage


@mark.basic
def test_page_from_image(dp_image_with_layout_and_word_annotations: Image) -> None:
    """
    test page gets converted from an image correctly
    """
    # Arrange
    dp_image = dp_image_with_layout_and_word_annotations
    title_ann = dp_image.get_annotation(category_names=["title"])[0]
    title_ann.dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="1")
    )
    text_ann = dp_image.get_annotation(category_names=["text"])[0]
    text_ann.dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="2")
    )

    word_anns = dp_image.get_annotation(category_names="word")

    word_anns[0].dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="1")
    )
    word_anns[1].dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="2")
    )
    word_anns[2].dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="1")
    )
    word_anns[3].dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="2")
    )

    # Act
    page = Page.from_image(
        dp_image,
        LayoutType.word,
        [LayoutType.text, LayoutType.title, LayoutType.list],
    )

    # Assert
    assert page.text == "hello world\nbye world"


@mark.basic
def test_image_with_anns_can_be_saved(image: WhiteImage) -> None:
    """
    test  save does not raise any exception
    """

    # Arrange
    test_image = Image(location=image.loc, file_name=image.file_name)
    test_image.image = ones((24, 85, 3), dtype=float32)
    cat_1 = ImageAnnotation(
        category_name="table",
        bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
    )
    test_image.dump(cat_1)

    # Act
    page = Page.from_image(test_image, LayoutType.table, [LayoutType.table])

    try:
        page.save(dry=True)
    except Exception as exception:  # pylint: disable=W0703
        assert False, f"{exception}"


@mark.basic
def test_load_page_from_file() -> None:
    """
    test class from_file returns a page
    """
    test_file_path = get_test_path() / "test_image.json"
    image = Page.from_file(test_file_path.as_posix())
    assert isinstance(image, Page)


@mark.basic
def test_get_layout_context() -> None:
    """
    test get_layout_context with various context sizes
    """

    # Arrange
    test_file_path = get_test_path() / "FRFPE" / "7406fd39ef9ab74660be111dca703065_0.json"
    page = Page.from_file(test_file_path.as_posix())
    ann_id = "41c5cb4b-f7b2-3c7c-93de-b2556d560de9"

    # Act
    out = page.get_layout_context(ann_id, 0)

    # Assert
    assert len(out) == 1
    assert out[0].annotation_id == ann_id

    # Act
    out = page.get_layout_context(ann_id, 1)

    # Assert
    assert len(out) == 3
    assert [ann.annotation_id for ann in out] == [
        "84a540e8-0d37-3aeb-9905-af2870c7b514",
        "41c5cb4b-f7b2-3c7c-93de-b2556d560de9",
        "e8785459-890e-3a97-823b-f07aa5eff5a2",
    ]

    # Act
    out = page.get_layout_context(ann_id, 2)

    # Assert
    assert len(out) == 4
    assert [ann.annotation_id for ann in out] == [
        "b45b3c5f-9c8e-35a1-bf5c-daaeb271ef38",
        "84a540e8-0d37-3aeb-9905-af2870c7b514",
        "41c5cb4b-f7b2-3c7c-93de-b2556d560de9",
        "e8785459-890e-3a97-823b-f07aa5eff5a2",
    ]
