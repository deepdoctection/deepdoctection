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

from copy import copy
from typing import List
from unittest.mock import MagicMock

from pytest import mark

from deepdoctection.datapoint import Image
from deepdoctection.extern.base import TokenClassResult
from deepdoctection.mapper.laylmstruct import image_to_layoutlm
from deepdoctection.pipe import LMTokenClassifierService
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.settings import names


class TestLMTokenClassifierService:  # pylint: disable=R0903
    """
    Test LMTokenClassifierService
    """

    @staticmethod
    @mark.requires_pt
    def test_pass_datapoint(
        dp_image_with_layout_and_word_annotations: Image,
        layoutlm_input: JsonDict,
        token_class_result: List[TokenClassResult],
    ) -> None:
        """
        Testing pass_datapoint
        """

        # Arrange
        tokenizer_output = {
            "input_ids": layoutlm_input["input_ids"],
            "attention_mask": layoutlm_input["attention_mask"],
            "token_type_ids": layoutlm_input["token_type_ids"],
        }
        tokenizer = MagicMock(return_value=tokenizer_output)
        word_output = copy(layoutlm_input["tokens"])
        word_output.pop(0)
        word_output.pop(-1)

        word_output = [word_output[0], word_output[1], word_output[2], word_output[3]]
        tokenizer.tokenize = MagicMock(side_effect=word_output)
        tokenizer.cls_token_id = 101
        tokenizer.sep_token_id = 102
        tokenizer.pad_token_id = 0

        lm = MagicMock()  # pylint: disable=C0103
        lm.predict = MagicMock(return_value=token_class_result)
        lm_service = LMTokenClassifierService(tokenizer, lm, image_to_layoutlm)

        dp = dp_image_with_layout_and_word_annotations

        # Act
        dp = lm_service.pass_datapoint(dp)

        # Assert
        words = dp.get_annotation(annotation_ids="3a696daf-15d5-3b88-be63-02912ef35cfb")
        assert words[0].get_sub_category(names.C.SE).category_name == "FOO"
        assert words[0].get_sub_category(names.NER.TAG).category_name == "B"

        words = dp.get_annotation(annotation_ids="37d79fd7-ab87-30fe-b460-9b6e62e901b9")
        assert words[0].get_sub_category(names.C.SE).category_name == "FOO"
        assert words[0].get_sub_category(names.NER.TAG).category_name == "B"

        words = dp.get_annotation(annotation_ids="5d40236e-430c-3d56-a8a3-fe9e46b872ac")
        assert words[0].get_sub_category(names.C.SE).category_name == "FOO"
        assert words[0].get_sub_category(names.NER.TAG).category_name == "I"

        words = dp.get_annotation(annotation_ids="f8227d59-ea7f-342a-97fa-23df1f189762")
        assert words[0].get_sub_category(names.C.SE).category_name == "FOO"
        assert words[0].get_sub_category(names.NER.TAG).category_name == "I"
