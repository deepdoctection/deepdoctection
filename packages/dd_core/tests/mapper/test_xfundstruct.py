# -*- coding: utf-8 -*-
# File: test_xfundstruct.py

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
Tests for the module handling XFUND structure conversion to image representation.

This module provides tests for verifying the mapping of XFUND data points to
`Image` objects, with specific configurations such as image loading and fake
score assignment. It ensures correct handling of word-level annotations and
verifies expected structures in the resulting `Image`.

"""


from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from dd_core.datapoint import Image
from dd_core.mapper.xfundstruct import xfund_to_image
from dd_core.utils.object_types import (
    BioTag,
    LayoutType,
    TokenClasses,
    WordType,
    token_class_tag_to_token_class_with_tag,
)


def test_xfund_to_image_load_image_fake_score(monkeypatch: pytest.MonkeyPatch, xfund_datapoint: Dict[str, Any]) -> None:
    """Map XFUND with image loading and fake scores; validate word-level annotations."""

    CATEGORIES_DICT_NAME_AS_KEY: Dict[LayoutType, int] = {LayoutType.TEXT: 1, LayoutType.WORD: 2}
    NER_TOKEN_TO_ID_MAPPING: Dict[LayoutType, Dict[WordType, Dict[Any, int]]] = {
        LayoutType.WORD: {
            WordType.TAG: {
                BioTag.BEGIN: 51,
                BioTag.OUTSIDE: 52,
                BioTag.INSIDE: 53,
            },
            WordType.TOKEN_CLASS: {
                TokenClasses.OTHER: 61,
                TokenClasses.HEADER: 62,
            },
            WordType.TOKEN_TAG: {
                BioTag.OUTSIDE: 71,
                token_class_tag_to_token_class_with_tag(TokenClasses.HEADER, BioTag.BEGIN): 72,
                token_class_tag_to_token_class_with_tag(TokenClasses.HEADER, BioTag.INSIDE): 73,
            },
        }
    }
    TOKEN_CLASS_NAMES_MAPPING: Dict[str, str] = {
        "other": "other",
        "header": "header",
    }

    monkeypatch.setattr(
        "dd_core.mapper.xfundstruct.load_image_from_file",
        lambda fn: np.zeros((3508, 2480, 3), dtype=np.uint8),
    )

    img: Optional[Image] = xfund_to_image(
        load_image=True,
        fake_score=True,
        categories_dict_name_as_key=CATEGORIES_DICT_NAME_AS_KEY,
        token_class_names_mapping=TOKEN_CLASS_NAMES_MAPPING,
        ner_token_to_id_mapping=NER_TOKEN_TO_ID_MAPPING,
    )(xfund_datapoint)

    assert img is not None
    assert isinstance(img.image, np.ndarray)
    assert img.image.shape == (3508, 2480, 3)

    # Collect word annotations
    word_anns: List[Any] = img.get_annotation(category_names=LayoutType.WORD)
    assert len(word_anns) == 3
    assert all(ann.score is not None for ann in word_anns)

    observed = []
    for ann in word_anns:
        chars = ann.get_sub_category(WordType.CHARACTERS).value
        token_class = ann.get_sub_category(WordType.TOKEN_CLASS).category_name
        tag = ann.get_sub_category(WordType.TAG).category_name
        token_tag_cat = ann.get_sub_category(WordType.TOKEN_TAG).category_name
        token_tag_id = ann.get_sub_category(WordType.TOKEN_TAG).category_id
        observed.append((chars, token_class, tag, token_tag_cat, token_tag_id))

    expected = [
        ("Akademisches", "other", BioTag.OUTSIDE, BioTag.OUTSIDE, 71),
        ("Auslandsamt", "other", BioTag.OUTSIDE, BioTag.OUTSIDE, 71),
        (
            "Bewerbungsformular",
            "header",
            BioTag.BEGIN,
            token_class_tag_to_token_class_with_tag(TokenClasses.HEADER, BioTag.BEGIN),
            72,
        ),
    ]

    for exp in expected:
        assert any(
            o[0] == exp[0] and o[1] == exp[1] and o[2] == exp[2] and o[3] == exp[3] and o[4] == exp[4] for o in observed
        ), f"Expected annotation {exp} not found. Got {observed}"
