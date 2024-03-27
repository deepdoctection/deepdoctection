# -*- coding: utf-8 -*-
# File: test_tessocr.py

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
Testing module extern.tessocr
"""
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.tessocr import (
    TesseractOcrDetector,
    TesseractRotationTransformer,
    tesseract_line_to_detectresult,
)
from deepdoctection.utils.detection_types import ImageType
from deepdoctection.utils.error import DependencyError
from tests.data import Annotations

from .data import WORD_RESULTS


@pytest.fixture(name="word_result_list_same_line")
def fixture_pdf_bytes_page_2() -> List[DetectionResult]:
    """
    fixture list of word results. Words are in the same line
    """
    return WORD_RESULTS


def get_mock_word_results(
    np_img: ImageType, supported_languages: str, text_lines: bool, config: str  # pylint: disable=W0613
) -> List[DetectionResult]:
    """
    Returns WordResults attr: word_results_list
    """
    return Annotations().get_word_detect_results()


class TestTesseractOcrDetector:
    """
    Test TesseractOcrDetector
    """

    @staticmethod
    @pytest.mark.basic
    @patch("deepdoctection.utils.subprocess.check_output", MagicMock(side_effect=OSError))
    def test_tesseract_ocr_raises_tesseract_not_found_error_when_dependencies_not_satisfied(
        path_to_tesseract_yaml: str,
    ) -> None:
        """
        This tests shows that the dependency implementation of the base class and is representative for
        the dependency logic of all derived classes.
        """

        # Act and Assert
        with pytest.raises(DependencyError):
            TesseractOcrDetector(path_yaml=path_to_tesseract_yaml)

    @staticmethod
    @pytest.mark.basic
    @patch("deepdoctection.utils.file_utils.get_tesseract_version", MagicMock(return_value=3.15))
    @patch("deepdoctection.extern.tessocr.predict_text", MagicMock(side_effect=get_mock_word_results))
    def test_tesseract_ocr_predicts_image(path_to_tesseract_yaml: str, np_image: ImageType) -> None:
        """
        Detector calls predict_text
        """

        # Arrange
        tess = TesseractOcrDetector(path_yaml=path_to_tesseract_yaml)

        # Act
        results = tess.predict(np_image)

        # Assert
        assert len(results) == 2


@pytest.mark.basic
def test_line_detect_result_returns_line(word_result_list_same_line: List[DetectionResult]) -> None:
    """
    Testing tesseract_line_to_detectresult generates Line DetectionResult
    """

    # Act
    detect_result_list = tesseract_line_to_detectresult(word_result_list_same_line)

    # Assert
    assert len(detect_result_list) == 3
    line_detect_result = detect_result_list[2]
    assert line_detect_result.box == [10.0, 10.0, 38.0, 24.0]
    assert line_detect_result.class_id == 2
    assert line_detect_result.text == "foo bak"


class TestTesseractRotationTransformer:
    """
    Test TesseractRotationTransformer
    """

    @staticmethod
    @pytest.mark.basic
    @patch(
        "deepdoctection.extern.tessocr.predict_rotation",
        MagicMock(
            return_value={
                "Orientation confidence": "8.70",
                "Orientation in degrees": "180",
                "Page number": "0",
                "Rotate": "180",
                "Script": "Latin",
                "Script confidence": "4.62",
            }
        ),
    )
    def test_tesseract_rotation_transformer_predicts_image(np_image: ImageType) -> None:
        """
        TesseractRotationTransformer calls predict and returns correct DetectionResult
        """

        # Arrange
        tess = TesseractRotationTransformer()

        # Act
        result = tess.predict(np_image)

        # Assert
        assert result.angle == 180.0
        assert result.score == 8.70

    @staticmethod
    @pytest.mark.basic
    def test_tesseract_rotation_transformer_rotates_image(
        np_image: ImageType, angle_detection_result: DetectionResult
    ) -> None:
        """
        TesseractRotationTransformer rotates image according to angle_detection_result
        """

        # Arrange
        tess = TesseractRotationTransformer()

        # Act
        np_output = tess.transform(np_image, angle_detection_result)

        # Assert
        assert np_output.shape[0] == np_image.shape[1]
        assert np_output.shape[1] == np_image.shape[0]
