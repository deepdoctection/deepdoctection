# -*- coding: utf-8 -*-
# File: texocr.py

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
AWS Textract OCR engine for text extraction
"""

from typing import List

from ..datapoint.convert import convert_np_array_to_b64_b
from ..utils.detection_types import ImageType, JsonDict, Requirement
from ..utils.file_utils import boto3_available, get_aws_requirement, get_boto3_requirement
from ..utils.settings import names
from .base import DetectionResult, ObjectDetector, PredictorBase

if boto3_available():
    import boto3  # type:ignore


def _textract_to_detectresult(response: JsonDict, width: int, height: int, text_lines: bool) -> List[DetectionResult]:
    all_results: List[DetectionResult] = []
    blocks = response.get("Blocks")

    if blocks:
        for block in blocks:
            if (block["BlockType"] in "WORD") or (block["BlockType"] in "LINE" and text_lines):
                word = DetectionResult(
                    box=[
                        block["Geometry"]["Polygon"][0]["X"] * width,
                        block["Geometry"]["Polygon"][0]["Y"] * height,
                        block["Geometry"]["Polygon"][2]["X"] * width,
                        block["Geometry"]["Polygon"][2]["Y"] * height,
                    ],
                    score=block["Confidence"] / 100,
                    text=block["Text"],
                    class_id=1 if block["BlockType"] == "WORD" else 2,
                    class_name=names.C.WORD if block["BlockType"] == "WORD" else names.C.LINE,
                )
                all_results.append(word)

    return all_results


def predict_text(np_img: ImageType, client, text_lines: bool) -> List[DetectionResult]:  # type:ignore
    """
    Calls AWS Textract client (:meth:`detect_document_text`) and returns plain OCR results.
    AWS account required.

    :param client: botocore textract client
    :param np_img: Image in np.array.
    :param text_lines: If True, it will return DetectionResults of Text lines as well.
    :return: A list of textract extractions wrapped in DetectionResult
    """

    width, height = np_img.shape[1], np_img.shape[0]
    b_img = convert_np_array_to_b64_b(np_img)
    response = client.detect_document_text(Document={"Bytes": b_img})
    all_results = _textract_to_detectresult(response, width, height, text_lines)
    return all_results


class TextractOcrDetector(ObjectDetector):
    """
    Text object detector based on AWS Textract OCR engine. Note that an AWS account as well as some additional
    installations are required. Note further, that the service is not free of charge. Additional information can
    be found at: https://docs.aws.amazon.com/textract/?id=docs_gateway .

    The detector only calls the base OCR engine and does not return additional Textract document analysis features.
    """

    def __init__(self, text_lines: bool = False) -> None:
        """
        :param text_lines: If True, it will return DetectionResults of Text lines as well.
        """
        self.text_lines = text_lines
        self.client = boto3.client("textract")

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Transfer of a numpy array and call textract client. Return of the detection results.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """

        return predict_text(np_img, self.client, self.text_lines)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_aws_requirement(), get_boto3_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__()
