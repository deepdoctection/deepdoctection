# -*- coding: utf-8 -*-
# File: test_tokenclass.py

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
Testing module pipe.tokenclass
"""

from typing import List
from unittest.mock import MagicMock

from pytest import mark

from deepdoctection.datapoint import Image
from deepdoctection.extern.base import SequenceClassResult, TokenClassResult
from deepdoctection.pipe import LMSequenceClassifierService, LMTokenClassifierService
from deepdoctection.utils.file_utils import transformers_available
from deepdoctection.utils.object_types import BioTag, PageType, TokenClasses, WordType

if transformers_available():
    from transformers import LayoutLMForSequenceClassification, LayoutLMForTokenClassification, LayoutLMTokenizerFast


class TestLMTokenClassifierService:
    """
    Test LMTokenClassifierService
    """

    @staticmethod
    @mark.pt_deps
    def test_pass_datapoint_2(
        dp_image_with_layout_and_word_annotations: Image,
        token_class_result: List[TokenClassResult],
    ) -> None:
        """
        Testing pass_datapoint with fast tokenizer
        """

        # Arrange
        tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

        language_model = MagicMock()
        language_model.predict = MagicMock(return_value=token_class_result)
        language_model.image_to_features_mapping = MagicMock(return_value="image_to_layoutlm_features")
        language_model.name = "test"
        language_model.default_kwargs_for_input_mapping = MagicMock(return_value={})
        language_model.model = MagicMock(spec=LayoutLMForTokenClassification)
        language_model.model.config = MagicMock()
        language_model.model.config.tokenizer_class = "LayoutLMTokenizerFast"
        lm_service = LMTokenClassifierService(tokenizer_fast, language_model)

        dp = dp_image_with_layout_and_word_annotations

        # Act
        dp = lm_service.pass_datapoint(dp)

        # Assert
        words = dp.get_annotation(annotation_ids="e9c4b3e7-0b2c-3d45-89f3-db6e3ef864ad")
        assert words[0].get_sub_category(WordType.TOKEN_CLASS).category_name == TokenClasses.HEADER
        assert words[0].get_sub_category(WordType.TAG).category_name == BioTag.BEGIN

        words = dp.get_annotation(annotation_ids="44ef758d-92f5-3f57-b6a3-aa95b9606f70")
        assert words[0].get_sub_category(WordType.TOKEN_CLASS).category_name == TokenClasses.HEADER
        assert words[0].get_sub_category(WordType.TAG).category_name == BioTag.BEGIN

        words = dp.get_annotation(annotation_ids="1413d499-ce19-3a50-861c-7d8c5a7ba772")
        assert words[0].get_sub_category(WordType.TOKEN_CLASS).category_name == TokenClasses.HEADER
        assert words[0].get_sub_category(WordType.TAG).category_name == BioTag.INSIDE

        words = dp.get_annotation(annotation_ids="fd78767a-227d-3c17-83cb-586d24cb0c55")
        assert words[0].get_sub_category(WordType.TOKEN_CLASS).category_name == TokenClasses.HEADER
        assert words[0].get_sub_category(WordType.TAG).category_name == BioTag.INSIDE


class TestLMSequenceClassifierService:
    """
    Test LMSequenceClassifierService
    """

    @staticmethod
    @mark.pt_deps
    def test_pass_datapoint(
        dp_image_with_layout_and_word_annotations: Image,
        sequence_class_result: SequenceClassResult,
    ) -> None:
        """
        Testing pass_datapoint
        """

        # Arrange
        tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")

        language_model = MagicMock()
        language_model.model = MagicMock(spec=LayoutLMForSequenceClassification)
        language_model.model.config = MagicMock()
        language_model.model.config.tokenizer_class = "LayoutLMTokenizerFast"
        language_model.predict = MagicMock(return_value=sequence_class_result)
        language_model.image_to_features_mapping = MagicMock(return_value="image_to_layoutlm_features")
        lm_service = LMSequenceClassifierService(tokenizer_fast, language_model)

        dp = dp_image_with_layout_and_word_annotations

        # Act
        dp = lm_service.pass_datapoint(dp)

        # Assert
        assert dp.summary.get_sub_category(PageType.DOCUMENT_TYPE).category_name == "FOO"
