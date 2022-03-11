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

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import boto3_available, get_aws_requirement, get_boto3_requirement
from ..utils.settings import names
from .base import DetectionResult, ObjectDetector, PredictorBase
from .textract.textract import predict_text

if boto3_available():
    import boto3  # type:ignore


class TextractOcrDetector(ObjectDetector):
    """
    Text object detector based on AWS Textract OCR engine. Note that an AWS account as well as some additional
    installations are required. Note further, that the service is not free of charge. Additional information can
    be found at: https://docs.aws.amazon.com/textract/?id=docs_gateway .

    The detector only calls the base OCR engine and does not return additional Textract document analysis features.
    """

    def __init__(self) -> None:
        self.client = boto3.client("textract")

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Transfer of a numpy array and call textract client. Return of the detection results.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """

        detection_results = predict_text(np_img, self.client)
        return TextractOcrDetector._map_category_names(detection_results)

    @staticmethod
    def _map_category_names(detection_results: List[DetectionResult]) -> List[DetectionResult]:
        for result in detection_results:
            result.class_name = names.C.WORD
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_aws_requirement(), get_boto3_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__()
