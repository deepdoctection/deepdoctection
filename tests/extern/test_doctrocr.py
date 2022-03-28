# -*- coding: utf-8 -*-
# File: test_doctrocr.py

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
Testing module extern.doctrocr
"""


from typing import List, Tuple
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.doctrocr import DoctrTextlineDetector, DoctrTextRecognizer
from deepdoctection.utils.detection_types import ImageType
from tests.data import Annotations


def get_mock_word_results(np_img: ImageType, predictor) -> List[DetectionResult]:  # type: ignore  # pylint: disable=W0613
    """
    Returns WordResults attr: word_results_list
    """
    return Annotations().get_word_detect_results()


def get_mock_text_line_results(  # type: ignore
    inputs: List[Tuple[str, ImageType]], predictor  # pylint: disable=W0613
) -> List[DetectionResult]:

    """
    Returns two DetectionResult
    """

    return [
        DetectionResult(score=0.1, text="Foo", uuid="cf234ec9-52cf-4710-94ce-288f0e055091"),
        DetectionResult(score=0.4, text="Bak", uuid="cf234ec9-52cf-4710-94ce-288f0e055092"),
    ]


class TestDoctrTextlineDetector:  # pylint: disable=R0903
    """
    Test DoctrTextlineDetector
    """

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text_lines", MagicMock(side_effect=get_mock_word_results))
    def test_doctr_detector_predicts_image(np_image: ImageType) -> None:
        """
        Detector calls doctr_predict_text_lines
        """

        # Arrange
        doctr = DoctrTextlineDetector()

        # Act
        results = doctr.predict(np_image)

        # Assert
        assert len(results) == 2


class TestDoctrTextRecognizer:  # pylint: disable=R0903
    """
    Test DoctrTextRecognizer
    """

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text", MagicMock(side_effect=get_mock_text_line_results))
    def test_doctr_detector_predicts_text(text_lines: List[Tuple[str, ImageType]]) -> None:
        """
        Detector calls doctr_predict_text
        """

        # Arrange
        doctr = DoctrTextRecognizer()

        # Act
        results = doctr.predict(text_lines)

        # Assert
        assert len(results) == 2
