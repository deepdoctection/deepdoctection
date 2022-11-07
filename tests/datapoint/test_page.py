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
from deepdoctection.datapoint.view import Page
from deepdoctection.utils.settings import Relationships, LayoutType

from ..test_utils import get_test_path


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
    assert page.text == "\nhello world\nbye world"
