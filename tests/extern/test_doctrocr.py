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

import os
from typing import List, Tuple
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.extern.base import DetectionResult
from deepdoctection.extern.doctrocr import DocTrRotationTransformer, DoctrTextlineDetector, DoctrTextRecognizer
from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager
from deepdoctection.utils.types import PixelValues
from tests.data import Annotations


def get_mock_word_results(np_img: PixelValues, predictor) -> List[DetectionResult]:  # type: ignore  # pylint: disable=W0613
    """
    Returns WordResults attr: word_results_list
    """
    return Annotations().get_word_detect_results()


def get_mock_text_line_results(  # type: ignore
    inputs: List[Tuple[str, PixelValues]], predictor  # pylint: disable=W0613
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
    @mark.pt_deps
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text_lines", MagicMock(side_effect=get_mock_word_results))
    def test_doctr_detector_predicts_image(np_image: PixelValues) -> None:
        """
        Detector calls doctr_predict_text_lines. Only runs in pt environment
        """

        # Arrange
        path_weights = ModelDownloadManager.maybe_download_weights_and_configs(
            "doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt"
        )
        categories = ModelCatalog.get_profile("doctr/db_resnet50/db_resnet50-ac60cadc.pt").categories
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
    @mark.pt_deps
    @patch("deepdoctection.extern.doctrocr.doctr_predict_text", MagicMock(side_effect=get_mock_text_line_results))
    def test_doctr_recognizer_predicts_text(text_lines: List[Tuple[str, PixelValues]]) -> None:
        """
        Detector calls doctr_predict_text. Only runs in pt environment
        """

        # Arrange
        path_weights = ModelDownloadManager.maybe_download_weights_and_configs(
            "doctr/crnn_vgg16_bn/crnn_vgg16_bn-9762b0b0.pt"
        )
        doctr = DoctrTextRecognizer("crnn_vgg16_bn", path_weights, "cpu")

        # Act
        results = doctr.predict(text_lines)

        # Assert
        assert len(results) == 2



class TestDocTrRotationTransformer:
    """
    Test DocTrRotationTransformer
    """

    @staticmethod
    @mark.pt_deps
    @patch("deepdoctection.extern.doctrocr.estimate_orientation", MagicMock(return_value=180.0))
    def test_doctr_rotation_transformer_predicts_image(np_image: PixelValues) -> None:
        """
        DocTrRotationTransformer calls predict and returns correct DetectionResult
        """

        # Arrange
        transformer = DocTrRotationTransformer()

        # Act
        result = transformer.predict(np_image)

        # Assert
        assert result.angle == 180.0

    @staticmethod
    @mark.pt_deps
    def test_doctr_rotation_transformer_rotates_image(
        np_image: PixelValues, angle_detection_result: DetectionResult
    ) -> None:
        """
        DocTrRotationTransformer calls transform and returns rotated image
        """

        # Arrange
        transformer = DocTrRotationTransformer()

        # Act
        np_output = transformer.transform_image(np_image, angle_detection_result)

        # Assert
        assert np_output.shape[0] == np_image.shape[1]
        assert np_output.shape[1] == np_image.shape[0]
