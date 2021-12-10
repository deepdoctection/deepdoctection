# -*- coding: utf-8 -*-
# File: textract.py

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
Module for calling :meth:`detect_document_text`
"""

from typing import List

from ...datapoint.convert import convert_np_array_to_b64_b
from ...utils.detection_types import ImageType, JsonDict
from ..base import DetectionResult

__all__ = ["predict_text"]


def _textract_to_detectresult(response: JsonDict, width: int, height: int) -> List[DetectionResult]:
    all_results: List[DetectionResult] = []
    blocks = response.get("Blocks")

    if blocks:
        for block in blocks:
            if block["BlockType"] == "WORD":
                word = DetectionResult(
                    box=[
                        block["Geometry"]["Polygon"][0]["X"] * width,
                        block["Geometry"]["Polygon"][0]["Y"] * height,
                        block["Geometry"]["Polygon"][2]["X"] * width,
                        block["Geometry"]["Polygon"][2]["Y"] * height,
                    ],
                    score=block["Confidence"] / 100,
                    text=block["Text"],
                    class_id=1,
                )
                all_results.append(word)

    return all_results


def predict_text(np_img: ImageType, client) -> List[DetectionResult]:  # type:ignore
    """
    Calls AWS Textract client (:meth:`detect_document_text`) and returns plain OCR results.
    AWS account required.

    :param client: botocore textract client
    :param np_img: Image in np.array.
    :return: A list of textract extractions wrapped in DetectionResult
    """

    width, height = np_img.shape[1], np_img.shape[0]
    b_img = convert_np_array_to_b64_b(np_img)
    response = client.detect_document_text(Document={"Bytes": b_img})
    all_results = _textract_to_detectresult(response, width, height)
    return all_results
