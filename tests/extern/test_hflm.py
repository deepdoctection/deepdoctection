# -*- coding: utf-8 -*-
# File: test_hflm.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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

from pytest import mark

from deepdoctection.extern.base import SequenceClassResult, TokenClassResult
from deepdoctection.extern.hflm import HFLmSequenceClassifier, HFLmTokenClassifier
from deepdoctection.utils.file_utils import pytorch_available
from deepdoctection.utils.settings import BioTag, TokenClasses, get_type
from deepdoctection.utils.types import JsonDict

from ..mapper.data import DatapointXfund
from ..test_utils import get_mock_patch

if pytorch_available():
    import torch


def get_sequence_class_result(  # type: ignore
    input_ids, attention_mask, token_type_ids, model  # pylint: disable=W0613
) -> SequenceClassResult:
    """
    sequence class result
    """
    return DatapointXfund().get_sequence_class_results()


# pylint: disable=W0613
def get_token_class_results(  # type: ignore
    uuids: List[str],
    input_ids,
    attention_mask,
    token_type_ids,
    tokens,
    model,
) -> List[TokenClassResult]:
    """
    token class result list
    """
    return DatapointXfund().get_token_class_results()


class TestHFLmSequenceClassifier:
    """
    Test HFLmSequenceClassifier
    """

    @staticmethod
    @mark.pt_deps
    @patch("deepdoctection.extern.hflm.predict_sequence_classes", MagicMock(side_effect=get_sequence_class_result))
    def test_hf_layout_lm_predicts_sequence_class(
        layoutlm_input_for_predictor: JsonDict,
    ) -> None:
        """
        HFLmSequenceClassifier calls predict_sequence_classes and post processes SequenceClassResult correctly
        """

        # Arrange
        HFLmSequenceClassifier.get_wrapped_model = MagicMock(  # type: ignore
            return_value=get_mock_patch("XLMRobertaForSequenceClassification")
        )
        categories = {1: get_type("FOO"), 2: get_type("BAK")}
        layoutlm = HFLmSequenceClassifier("path/to/json", "path/to/model", categories)
        layoutlm.model.device = "cpu"

        # Act
        inputs = {
            "input_ids": torch.tensor(layoutlm_input_for_predictor["input_ids"]),
            "attention_mask": torch.tensor(layoutlm_input_for_predictor["attention_mask"]),
            "token_type_ids": torch.tensor(layoutlm_input_for_predictor["token_type_ids"]),
        }

        results = layoutlm.predict(**inputs)

        # Assert
        assert results.class_name == "BAK"


class TestHFLmTokenClassifier:
    """
    Test HFLmTokenClassifier
    """

    @staticmethod
    @mark.pt_deps
    @patch("deepdoctection.extern.hflm.predict_token_classes", MagicMock(side_effect=get_token_class_results))
    def test_hf_lm_predicts_token(
        layoutlm_input_for_predictor: JsonDict,
        token_class_names: List[str],
    ) -> None:
        """
        HFLayoutLmTokenClassifier calls predict_token_classes and post processes TokenClassResult correctly
        """

        # Arrange
        HFLmTokenClassifier.get_wrapped_model = (  # type: ignore
            MagicMock(return_value=get_mock_patch("XLMRobertaForTokenClassification"))
        )
        categories_semantics = [TokenClasses.HEADER]
        categories_bio = [BioTag.BEGIN, BioTag.INSIDE, BioTag.OUTSIDE]
        lm = HFLmTokenClassifier(
            "path/to/json", "path/to/model", categories_semantics, categories_bio, use_xlm_tokenizer=True
        )
        lm.model.device = "cpu"

        # Act
        inputs = {
            "image_ids": layoutlm_input_for_predictor["image_ids"],
            "width": layoutlm_input_for_predictor["width"],
            "height": layoutlm_input_for_predictor["height"],
            "ann_ids": layoutlm_input_for_predictor["ann_ids"],
            "tokens": layoutlm_input_for_predictor["tokens"],
            "input_ids": torch.tensor(layoutlm_input_for_predictor["input_ids"]),
            "attention_mask": torch.tensor(layoutlm_input_for_predictor["attention_mask"]),
            "token_type_ids": torch.tensor(layoutlm_input_for_predictor["token_type_ids"]),
        }

        results = lm.predict(**inputs)

        # Assert
        assert len(results) == 18
        class_names = [res.class_name for res in results]
        assert class_names == token_class_names
