# -*- coding: utf-8 -*-
# File: test_xfundstruct.py

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
Testing module mapper.xfundstruct
"""

from typing import Dict, Mapping
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.mapper.xfundstruct import xfund_to_image
from deepdoctection.utils.settings import CellType, LayoutType, ObjectTypes, TokenClasses, WordType
from deepdoctection.utils.types import JsonDict

from .conftest import get_always_pubtabnet_white_image


@mark.basic
@patch(
    "deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(side_effect=get_always_pubtabnet_white_image)
)
def test_xfund_to_image(
    datapoint_xfund: JsonDict,
    xfund_category_dict: Mapping[ObjectTypes, str],
    xfund_category_names: Dict[str, str],
    ner_token_to_id_mapping: JsonDict,
) -> None:
    """
    testing xfund_to_image is mapping correctly
    """

    # Act
    xfund_to_image_func = xfund_to_image(
        False, False, xfund_category_dict, xfund_category_names, ner_token_to_id_mapping
    )
    img = xfund_to_image_func(datapoint_xfund)

    # Assert
    assert img
    word_anns = img.get_annotation(category_names=LayoutType.WORD)
    words = [ann.get_sub_category(WordType.CHARACTERS).value for ann in word_anns]  # type: ignore
    assert words == ["Akademisches", "Auslandsamt", "Bewerbungsformular"]

    sub_cats_category_names = [ann.get_sub_category(WordType.TOKEN_CLASS).category_name for ann in word_anns]
    assert sub_cats_category_names == [TokenClasses.OTHER, TokenClasses.OTHER, CellType.HEADER]

    sub_cats_ner_tags = [ann.get_sub_category(WordType.TAG).category_name for ann in word_anns]
    assert sub_cats_ner_tags == ["O", "O", "B"]

    text_anns = img.get_annotation(category_names=LayoutType.TEXT)
    sub_cats_category_names = [ann.get_sub_category(WordType.TOKEN_CLASS).category_name for ann in text_anns]
    assert sub_cats_category_names == [TokenClasses.OTHER, TokenClasses.HEADER]
