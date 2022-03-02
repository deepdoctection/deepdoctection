# -*- coding: utf-8 -*-
# File: pdfminertext.py

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
PDFPlumber text extraction engine
"""

from typing import List, Dict

from ..utils.file_utils import pdfplumber_available, get_pdfplumber_requirement
from ..utils.settings import names
from ..utils.detection_types import Requirement
from ..utils.context import  save_tmp_file
from .base import PdfMiner, DetectionResult

if pdfplumber_available():
    from pdfplumber.pdf import PDF


def _to_detect_result(word: Dict[str,str]):
    return DetectionResult(box=[float(word["x0"]),float(word["top"]),
                                float(word["x1"]),float(word["bottom"])],
                           class_id=1, text=word["text"],
                           class_name=names.C.WORD)


class PDFPlumberTextDetector(PdfMiner):
    """
    Text miner based on the pdfminer.six engine. To convert pdfminers result, especially group character to get word
    level results we use pdfplumber.
    """

    def predict(self, pdf_bytes: bytes) -> List[DetectionResult]:
        """
        Call pdfminer.six and returns detected text as detection results

        :param pdf_bytes: bytes of a single pdf page
        :return: A list of DetectionResult
        """

        with save_tmp_file(pdf_bytes, "pdf_") as (tmp_name, input_file_name):
            with open(tmp_name, 'rb') as fin:
                pdf = PDF(fin)
                page = pdf.pages[0]
                words = page.extract_words()
        detect_results = list(map(_to_detect_result,words))
        return detect_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pdfplumber_requirement()]
