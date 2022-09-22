# -*- coding: utf-8 -*-
# File: test_fastlang.py

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
Testing module extern.fastlang
"""
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
from numpy import float32
from pytest import mark

from deepdoctection.extern.fastlang import FasttextLangDetector
from deepdoctection.extern.model import ModelCatalog


def get_mock_lang_detect_result(text_string: str) -> Tuple[Tuple[str], npt.NDArray[float32]]:  # pylint: disable = W0613
    """return raw result from fasttext language detection model"""
    return (("__label__it",), np.array([0.99414486]))


class TestFasttextLangDetector:
    """
    Test FasttextLangDetector
    """

    @staticmethod
    @mark.requires_tf_or_pt
    @patch("deepdoctection.extern.fastlang.load_model", MagicMock(return_value=MagicMock()))
    def test_fasttext_lang_detector_predicts_language() -> None:
        """
        Detector calls model.predict(text_string) and processes returned results correctly
        """

        # Arrange
        path_weights = "/path/to/dir"
        profile = ModelCatalog.get_profile("fasttext/lid.176.bin")

        assert profile.categories
        fasttest_predictor = FasttextLangDetector(path_weights, profile.categories)
        fasttest_predictor.model.predict = MagicMock(side_effect=get_mock_lang_detect_result)

        # Act
        result = fasttest_predictor.predict("Un leggero dialetto italiano")

        # Assert
        assert result.text == "ita"
        assert result.score == 0.99414486
