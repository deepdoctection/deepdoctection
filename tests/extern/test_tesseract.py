# -*- coding: utf-8 -*-
# File: test_tesseract.py

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
Testing module extern.tesseract.tesseract
"""


from typing import List

from pytest import fixture

from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.tessocr import tesseract_line_to_detectresult

from .data import WORD_RESULTS


@fixture(name="word_result_list_same_line")
def fixture_pdf_bytes_page_2() -> List[DetectionResult]:
    """
    fixture list of word results. Words are in the same line
    """
    return WORD_RESULTS


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
