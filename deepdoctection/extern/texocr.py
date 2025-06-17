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
from __future__ import annotations

import sys
import traceback

from lazy_imports import try_import

from ..datapoint.convert import convert_np_array_to_b64_b
from ..utils.file_utils import get_boto3_requirement
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import LayoutType, ObjectTypes
from ..utils.types import JsonDict, PixelValues, Requirement
from .base import DetectionResult, ModelCategories, ObjectDetector

with try_import() as import_guard:
    import boto3  # type:ignore


def _textract_to_detectresult(response: JsonDict, width: int, height: int, text_lines: bool) -> list[DetectionResult]:
    all_results: list[DetectionResult] = []
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
                    class_name=LayoutType.WORD if block["BlockType"] == "WORD" else LayoutType.LINE,
                )
                all_results.append(word)

    return all_results


def predict_text(np_img: PixelValues, client: boto3.client, text_lines: bool) -> list[DetectionResult]:
    """
    Calls AWS Textract client (`detect_document_text`) and returns plain OCR results.
    AWS account required.

    Args:
        np_img: Image in `np.array`.
        client: botocore textract client
        text_lines: If `True`, it will return `DetectionResult`s of Text lines as well.

    Returns:
        A list of `DetectionResult`
    """

    width, height = np_img.shape[1], np_img.shape[0]
    b_img = convert_np_array_to_b64_b(np_img)
    try:
        response = client.detect_document_text(Document={"Bytes": b_img})
    except:  # pylint: disable=W0702
        _, exc_val, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[0]
        log_dict = {
            "file_name": "NN",
            "error_type": type(exc_val).__name__,
            "error_msg": str(exc_val),
            "orig_module": frame_summary.filename,
            "line": frame_summary.lineno,
        }
        logger.warning(LoggingRecord("botocore InvalidParameterException", log_dict))  # type: ignore
        response = {}

    all_results = _textract_to_detectresult(response, width, height, text_lines)
    return all_results


class TextractOcrDetector(ObjectDetector):
    """
    Text object detector based on AWS Textract OCR engine. Note that an AWS account as well as some additional
    installations are required, i.e `AWS CLI` and `boto3`.
    Note:
        The service is not free of charge. Additional information can be found at:
        <https://docs.aws.amazon.com/textract/?id=docs_gateway> .

    The detector only calls the base `OCR` engine and does not return additional Textract document analysis features.

    Example:

        ```python
        textract_predictor = TextractOcrDetector()
        detection_result = textract_predictor.predict(bgr_image_as_np_array)
        ```

        or

        ```python
        textract_predictor = TextractOcrDetector()
        text_extract = TextExtractionService(textract_predictor)

        pipe = DoctectionPipe([text_extract])
        df = pipe.analyze(path="path/to/document.pdf")

        for dp in df:
            ...
        ```

    """

    def __init__(self, text_lines: bool = False, **credentials_kwargs: str) -> None:
        """
        Args:
            text_lines: If `True`, it will return `DetectionResult`s of Text lines as well.
            credentials_kwargs: `aws_access_key_id`, `aws_secret_access_key` or `aws_session_token`
        """
        self.name = "textract"
        self.model_id = self.get_model_id()

        self.text_lines = text_lines
        self.client = boto3.client("textract", **credentials_kwargs)
        if self.text_lines:
            self.categories = ModelCategories(init_categories={1: LayoutType.WORD, 2: LayoutType.LINE})
        else:
            self.categories = ModelCategories(init_categories={1: LayoutType.WORD})

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Transfer of a `np.array` and call textract `client`. Return of the `DetectionResult`s.

        Args:
            np_img: image as `np.array`

        Returns:
            A list of `DetectionResult`s
        """

        return predict_text(np_img, self.client, self.text_lines)

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_boto3_requirement()]

    def clone(self) -> TextractOcrDetector:
        return self.__class__()

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)
