# -*- coding: utf-8 -*-
# File: test_hflayoutlm.py

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
Testing module extern.hflayoutlm
"""
from typing import List
from unittest.mock import MagicMock, patch

from pytest import mark, raises

from deepdoctection.extern.base import TokenClassResult
from deepdoctection.extern.hflayoutlm import HFLayoutLmTokenClassifier
from deepdoctection.utils.detection_types import JsonDict

from ..mapper.data import DatapointXfund


def get_token_class_results(  # type: ignore
    uuids: List[str], input_ids, attention_mask, token_type_ids, boxes, tokens, model  # pylint: disable=W0613
) -> List[TokenClassResult]:
    """
    token class result list
    """
    return DatapointXfund().get_token_class_results()


class TestHFLayoutLmTokenClassifier:
    """
    Test HFLayoutLmTokenClassifier
    """

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.pt.ptutils.pytorch_available", MagicMock(return_value=False))
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForTokenClassification.from_pretrained", MagicMock())
    def test_hf_layout_lm_does_not_build_when_pt_not_available() -> None:
        """
        HFLayoutLmTokenClassifier needs pytorch. Construction fails, when requirement is not satisfied
        """

        # Arrange, Act & Assert
        with raises(ImportError):
            HFLayoutLmTokenClassifier(["foo"], ["B", "I", "O"])

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForTokenClassification.from_pretrained", MagicMock())
    def test_categories_are_constructed_properly() -> None:
        """
        HFLayoutLmTokenClassifier creates a full category set depending on semantics, tagging or by passing the
        set of categories directly
        """

        # Arrange, Act & Assert
        with raises(AssertionError):
            HFLayoutLmTokenClassifier(["foo"], None)

        # Arrange
        categories_semantics = ["FOO"]
        categories_bio = ["B", "I", "O"]

        # Act
        model = HFLayoutLmTokenClassifier(categories_semantics, categories_bio)

        # Assert
        assert model.categories == {0: "B-FOO", 1: "I-FOO", 2: "O"}

        # Arrange
        categories_explicit_list = ["FOO", "BAK", "O"]
        categories_explicit = {0: "FOO", 1: "BAK", 2: "O"}

        # Act
        model = HFLayoutLmTokenClassifier(categories_explicit=categories_explicit_list)

        # Assert
        assert model.categories == categories_explicit

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.hflayoutlm.LayoutLMForTokenClassification.from_pretrained", MagicMock())
    @patch("deepdoctection.extern.hflayoutlm.predict_token_classes", MagicMock(side_effect=get_token_class_results))
    def test_hf_layout_lm_predicts_token(
        layoutlm_input: JsonDict,
        categories_semantics: List[str],
        categories_bio: List[str],
        token_class_names: List[str],
    ) -> None:
        """
        HFLayoutLmTokenClassifier calls predict_token_classes and post processes TokenClassResult correctly
        """

        # Arrange
        categories_semantics = ["FOO"]
        categories_bio = ["B", "I", "O"]
        layoutlm = HFLayoutLmTokenClassifier(categories_semantics, categories_bio)

        # Act
        results = layoutlm.predict(**layoutlm_input)

        # Assert
        assert len(results) == 18
        class_names = [res.class_name for res in results]
        assert class_names == token_class_names
