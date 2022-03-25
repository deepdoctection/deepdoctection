# -*- coding: utf-8 -*-
# File: test_texocr.py

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
Testing module extern.texocr
"""

from unittest.mock import MagicMock, patch

from deepdoctection.extern.texocr import TextractOcrDetector
from deepdoctection.utils.detection_types import ImageType, JsonDict


class TestTextractOcrDetector:  # pylint: disable=R0903
    """
    Test TextractOcrDetector
    """

    @staticmethod
    @patch("deepdoctection.extern.texocr.get_aws_requirement", MagicMock(return_value=("boto3", True, "")))
    @patch("deepdoctection.extern.texocr.get_boto3_requirement", MagicMock(return_value=("aws", True, "")))
    @patch("deepdoctection.extern.texocr.boto3", MagicMock())
    def test_textract_ocr_predicts_image(np_image: ImageType, textract_response: JsonDict) -> None:
        """
        Detector calls predict_text and returns only detected word blocks
        """

        # Arrange
        tex = TextractOcrDetector()
        tex.client.detect_document_text = MagicMock(return_value=textract_response)

        # Act
        results = tex.predict(np_image)

        # Assert
        assert len(results) == 12
