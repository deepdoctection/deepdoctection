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
from deepdoctection.extern.tessocr import TesseractOcrDetector
from deepdoctection.utils.detection_types import ImageType
from deepdoctection.utils.file_utils import TesseractNotFound
from tests.data import Annotations


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
    @patch("deepdoctection.utils.subprocess.check_output", MagicMock(side_effect=OSError))
    def test_tesseract_ocr_raises_tesseract_not_found_error_when_dependencies_not_satisfied(
        path_to_tesseract_yaml: str,
    ) -> None:
        """
        This tests shows that the dependency implementation of the base class and is representative for
        the dependency logic of all derived classes.
        """

        # Act and Assert
        with pytest.raises(TesseractNotFound):
            TesseractOcrDetector(path_yaml=path_to_tesseract_yaml)

    @staticmethod
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
