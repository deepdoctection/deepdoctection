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
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.extern.base import SequenceClassResult
from deepdoctection.extern.hflm import HFLmSequenceClassifier
from deepdoctection.utils.detection_types import JsonDict
from deepdoctection.utils.file_utils import pytorch_available
from deepdoctection.utils.settings import get_type

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
        categories = {"1": get_type("FOO"), "2": get_type("BAK")}
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
