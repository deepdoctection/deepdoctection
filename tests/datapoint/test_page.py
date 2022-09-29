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

import json
from pytest import mark

from deepdoctection.datapoint.annotation import CategoryAnnotation
from deepdoctection.datapoint.image import Image
from deepdoctection.datapoint.page import Page
from deepdoctection.utils.settings import Relationships

from ..test_utils import get_test_path


@mark.basic
def test_page_from_image(dp_image_with_layout_and_word_annotations: Image) -> None:
    """
    test page gets converted from an image correctly
    """
    # Arrange
    dp_image = dp_image_with_layout_and_word_annotations
    title_ann = dp_image.get_annotation(category_names=["TITLE"])[0]
    title_ann.dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="1")
    )
    text_ann = dp_image.get_annotation(category_names=["TEXT"])[0]
    text_ann.dump_sub_category(
        Relationships.reading_order, CategoryAnnotation(category_name=Relationships.reading_order, category_id="2")
    )

    word_anns = dp_image.get_annotation(category_names="WORD")

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
        "WORD",
        ["TEXT", "TITLE", "LIST"],
        ["TEXT", "TITLE", "LIST", "CELL"],
    )

    # Assert
    assert page.get_text() == "\nhello world\nbye world"


@mark.basic
def test_page_from_page_dict() -> None:
    """
    test page gets generated from a page dict
    """

    # Arrange
    path_json = get_test_path() / "sample_2_page_dict.json"

    with open(path_json, "r", encoding="UTF-8") as file:
        page_dict = json.load(file)

    # Act
    page = Page.from_dict(**page_dict)

    # Assert
    assert page.file_name == "sample_2.png"
    assert len(page.items) == 12
    assert len(page.tables) == 1
    assert page.width == 1654
    assert page.height == 2339
