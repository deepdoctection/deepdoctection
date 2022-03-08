# -*- coding: utf-8 -*-
# File: test_pdftext.py

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
Testing module extern.pdftext
"""

from deepdoctection.extern.pdftext import PdfPlumberTextDetector


class TestPdfPlumberTextDetector:
    """
    Test PdfPlumberTextDetector
    """

    @staticmethod
    def test_pdf_plumber_detector_predicts_image(pdf_bytes: bytes) -> None:
        """
        PdfPlumber returns words from pdf_bytes correctly
        """

        # Arrange
        pdf_text_predictor = PdfPlumberTextDetector()

        # Act
        detect_results_list = pdf_text_predictor.predict(pdf_bytes)

        # Assert
        assert len(detect_results_list) == 109
        assert detect_results_list[0].text == "A"
        assert detect_results_list[1].text == "Simple"

    @staticmethod
    def test_pdf_plumber_detector_returns_width_height(pdf_bytes: bytes, pdf_bytes_page_2: bytes) -> None:
        """
        PdfPlumber returns pdf width and height correctly
        """

        # Arrange
        pdf_text_predictor = PdfPlumberTextDetector()
        _ = pdf_text_predictor.predict(pdf_bytes)

        # Act
        width, height = pdf_text_predictor.get_width_height(pdf_bytes)

        # Assert
        assert width == 612
        assert height == 792

        # Act
        width, height = pdf_text_predictor.get_width_height(pdf_bytes_page_2)
        assert width == 1112
        assert height == 1792
