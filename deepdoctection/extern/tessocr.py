# -*- coding: utf-8 -*-
# File: tessocr.py

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
Tesseract OCR engine for text extraction
"""

from typing import List, Optional

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import get_tesseract_requirement
from ..utils.metacfg import config_to_cli_str, set_config_by_yaml
from ..utils.settings import names
from .base import DetectionResult, ObjectDetector, PredictorBase
from .tesseract.tesseract import predict_text


class TesseractOcrDetector(ObjectDetector):  # pylint: disable=R0903
    """
    Text object detector based on Tesseracts OCR engine. Note that tesseract has to be installed separately.

    The current Tesseract release is 4.1.1. A version 5.xx can be integrated via direct installation at
    https://github.com/tesseract-ocr/tesseract. Building from source is necessary here.

    Documentation can be found here: https://tesseract-ocr.github.io/

    All configuration options that are available via pytesseract can be given via the configuration. The best overview
    can be found at https://pypi.org/project/pytesseract/.
    """

    def __init__(
        self,
        path_yaml: str,
        config_overwrite: Optional[List[str]] = None,
    ):
        """
        Set up the configuration which is stored in a yaml-file, that need to be passed through.

        :param path_yaml: The path to the yaml config
        :param config_overwrite: Overwrite config parameters defined by the yaml file with new values.
                                 E.g. ["oem=14"]
        """

        if config_overwrite is None:
            config_overwrite = []

        hyper_param_config = set_config_by_yaml(path_yaml)
        if len(config_overwrite):
            hyper_param_config.update_args(config_overwrite)

        self.path_yaml = path_yaml
        self.config_overwrite = config_overwrite
        self.config = hyper_param_config

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Transfer of a numpy array and call of pytesseract. Return of the detection results.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        detection_results = predict_text(
            np_img, supported_languages=self.config.LANGUAGES, config=config_to_cli_str(self.config, "LANGUAGES")
        )
        return TesseractOcrDetector._map_category_names(detection_results)

    @staticmethod
    def _map_category_names(detection_results: List[DetectionResult]) -> List[DetectionResult]:
        for result in detection_results:
            result.class_name = names.C.WORD
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_tesseract_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_yaml, self.config_overwrite)
