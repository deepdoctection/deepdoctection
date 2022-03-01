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
PDFMinerSix text extraction engine
"""

from typing import List

from ..utils.file_utils import pdfminer_six_available, get_pdfminer_six_requirement
from ..utils.settings import names
from ..utils.detection_types import Requirement
from .base import PdfMiner, DetectionResult

if pdfminer_six_available():
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator


def predict_text(pdf_bytes,interpreter,device):
    pass


class PdfMinerTextDetector(PdfMiner):
    """
    Text miner based on the pdfminer.six engine. To run the detector make sure to install pdfminer.six as this module
    is not part of Deep-Doctection's setup.

    The detector only retrieves the text information and does not handle other elements as lines, figures or images.
    """

    def __init__(self) -> None:
        _resource_manager = PDFResourceManager(caching=True)
        self._device = PDFPageAggregator(_resource_manager)
        self._interpreter = PDFPageInterpreter(_resource_manager, self._device)

    def predict(self, pdf_bytes: bytes) -> List[DetectionResult]:
        """
        Call pdfminer.six and returns detected text as detection results

        :param pdf_bytes: bytes of a single pdf page
        :return: A list of DetectionResult
        """

        detection_results = predict_text(pdf_bytes, self._interpreter, self._device)
        return PdfMinerTextDetector._map_category_names(detection_results)

    @staticmethod
    def _map_category_names(detection_results: List[DetectionResult]) -> List[DetectionResult]:
        for result in detection_results:
            result.class_name = names.C.WORD
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pdfminer_six_requirement()]
