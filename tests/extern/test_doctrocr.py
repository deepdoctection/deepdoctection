# -*- coding: utf-8 -*-
# File: test_doctrocr.py

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
Testing module extern.doctrocr
"""


from typing import List, Tuple
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.doctrocr import DoctrTextlineDetector, DoctrTextRecognizer
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager
from deepdoctection.utils.detection_types import ImageType
from tests.data import Annotations


def get_mock_word_results(np_img: ImageType, predictor, device) -> List[DetectionResult]:  # type: ignore  # pylint: disable=W0613
    """
    Returns WordResults attr: word_results_list
    """
    return Annotations().get_word_detect_results()


def get_mock_text_line_results(  # type: ignore
    inputs: List[Tuple[str, ImageType]], predictor, device  # pylint: disable=W0613
) -> List[DetectionResult]:

    """
    Returns two DetectionResult
    """

    return [
        DetectionResult(score=0.1, text="Foo", uuid="cf234ec9-52cf-4710-94ce-288f0e055091"),
        DetectionResult(score=0.4, text="Bak", uuid="cf234ec9-52cf-4710-94ce-288f0e055092"),
    ]


class TestDoctrTextlineDetector:
    """
    Test DoctrTextlineDetector
    """

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text_lines", MagicMock(side_effect=get_mock_word_results))
    def test_pt_doctr_detector_predicts_image(np_image: ImageType) -> None:
        """
        Detector calls doctr_predict_text_lines. Only runs in pt environment
        """

        # Arrange
        path_weights = ModelDownloadManager.maybe_download_weights_and_configs(
            "doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt"
        )
        categories = ModelCatalog.get_profile("doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt").categories
        doctr = DoctrTextlineDetector("db_resnet50", path_weights, categories, "cpu")  # type: ignore

        # Act
        results = doctr.predict(np_image)

        # Assert
        assert len(results) == 2

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text_lines", MagicMock(side_effect=get_mock_word_results))
    def test_tf_doctr_detector_predicts_image(np_image: ImageType) -> None:
        """
        Detector calls doctr_predict_text_lines. Only runs in tf environment
        """

        # Arrange
        path_weights = ModelDownloadManager.maybe_download_weights_and_configs(
            "doctr/db_resnet50/tf/db_resnet50-adcafc63.zip"
        )
        categories = ModelCatalog.get_profile("doctr/db_resnet50/tf/db_resnet50-adcafc63.zip").categories
        doctr = DoctrTextlineDetector("db_resnet50", path_weights, categories, "cpu")  # type: ignore

        # Act
        results = doctr.predict(np_image)

        # Assert
        assert len(results) == 2


class TestDoctrTextRecognizer:
    """
    Test DoctrTextRecognizer
    """

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text", MagicMock(side_effect=get_mock_text_line_results))
    def test_doctr_pt_recognizer_predicts_text(text_lines: List[Tuple[str, ImageType]]) -> None:
        """
        Detector calls doctr_predict_text. Only runs in pt environment
        """

        # Arrange
        path_weights = ModelDownloadManager.maybe_download_weights_and_configs(
            "doctr/crnn_vgg16_bn/pt/crnn_vgg16_bn-9762b0b0.pt"
        )
        doctr = DoctrTextRecognizer("crnn_vgg16_bn", path_weights, "cpu")

        # Act
        results = doctr.predict(text_lines)

        # Assert
        assert len(results) == 2

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text", MagicMock(side_effect=get_mock_text_line_results))
    def test_doctr_tf_recognizer_predicts_text(text_lines: List[Tuple[str, ImageType]]) -> None:
        """
        Detector calls doctr_predict_text. Only runs in tf environment
        """

        # Arrange
        path_weights = ModelDownloadManager.maybe_download_weights_and_configs(
            "doctr/crnn_vgg16_bn/tf/crnn_vgg16_bn-76b7f2c6.zip"
        )
        doctr = DoctrTextRecognizer("crnn_vgg16_bn", path_weights, "cpu")

        # Act
        results = doctr.predict(text_lines)

        # Assert
        assert len(results) == 2
