# -*- coding: utf-8 -*-
# File: tesseract.py

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
Module for calling  :func:`pytesseract.image_to_data`
"""

from typing import List
import numpy as np

from pytesseract import image_to_data, Output  # type: ignore

from ...utils.detection_types import ImageType
from ..base import DetectionResult


__all__ = ["predict_text"]


def predict_text(np_img: ImageType, supported_languages: str, config: str) -> List[DetectionResult]:
    """
    Calls pytesseract :func:`pytesseract.image_to_data` with some given configs. It requires Tesseract to be installed.

    :param np_img: Image in np.array.
    :param supported_languages: To improve ocr extraction quality it is helpful to pre-select the language of the
                                detected text, if this in known in advance. Combinations are possible, e.g. "deu",
                                "fr+eng".
    :param config: The config parameter passing to Tesseract. Consult also https://guides.nyu.edu/tesseract/usage
    :return: A list of tesseract extractions wrapped in DetectionResult
    """

    np_img = np_img.astype(np.uint8)
    results = image_to_data(np_img, lang=supported_languages, output_type=Output.DICT, config=config)
    all_results = []

    for caption in zip(
        results["left"],
        results["top"],
        results["width"],
        results["height"],
        results["conf"],
        results["text"],
        results["block_num"],
        results["line_num"],
    ):
        if int(caption[4]) != -1:
            word = DetectionResult(
                box=[caption[0], caption[1], caption[0] + caption[2], caption[1] + caption[3]],
                score=caption[4] / 100,
                text=caption[5],
                block=str(caption[6]),
                line=str(caption[7]),
                class_id=1,
            )
            all_results.append(word)

    return all_results
