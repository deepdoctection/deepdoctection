# -*- coding: utf-8 -*-
# File: test_tpdetect.py

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
Testing module extern.tpdetect
"""

from typing import Dict, List
from unittest.mock import MagicMock, patch

from pytest import mark, raises

from deepdoctection.extern.base import DetectionResult
from deepdoctection.utils.detection_types import ImageType
from deepdoctection.utils.file_utils import tf_available

if tf_available():
    from deepdoctection.extern.tp.tpfrcnn.modeling.generalized_rcnn import ResNetFPNModel
    from deepdoctection.extern.tpdetect import TPFrcnnDetector


def set_num_gpu_to_zero() -> int:
    """
    set gpu number to zero
    """
    return 0


def set_num_gpu_to_one() -> int:
    """
    set gpu number to one
    """
    return 1


def get_mock_detection_results(  # type: ignore
    np_img: ImageType,  # pylint: disable=W0613
    predictor,  # pylint: disable=W0613
    preproc_short_edge_size,  # pylint: disable=W0613
    preproc_max_size,  # pylint: disable=W0613
    mrcnn_accurate_paste,  # pylint: disable=W0613
) -> List[DetectionResult]:
    """
    returns list with two DetectionResult
    """

    return [
        DetectionResult(box=[1.0, 1.6, 2.0, 4.6], score=0.97, class_id=2),
        DetectionResult(box=[12.0, 12.0, 12.0, 12.0], score=0.03, class_id=4),
    ]


class TestTPFrcnnDetector:
    """
    Test TPFrcnnDetector constructor
    """

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.tp.tpcompat.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_zero))
    def test_tp_frcnn_does_not_build_when_no_gpu(path_to_tp_frcnn_yaml: str, categories: Dict[str, str]) -> None:
        """
        TP FRCNN needs one GPU for predicting. Construction fails, when no GPU is found
        """
        # Arrange, Act & Assert
        with raises(AssertionError):
            TPFrcnnDetector(path_yaml=path_to_tp_frcnn_yaml, path_weights="", categories=categories)

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.tp.tpcompat.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_one))
    def test_tp_frcnn_returns_fpn_model(path_to_tp_frcnn_yaml: str, categories: Dict[str, str]) -> None:
        """
        TP FRCNN builds RestNetFPN model is construction is successful.
        """
        # Arrange, Act
        frcnn = TPFrcnnDetector(path_yaml=path_to_tp_frcnn_yaml, path_weights="", categories=categories)

        # Assert
        assert isinstance(frcnn._model, ResNetFPNModel)  # pylint: disable=W0212

    @staticmethod
    @mark.requires_tf
    @patch("deepdoctection.extern.tp.tpcompat.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_one))
    @patch("deepdoctection.extern.tp.tpcompat.TensorpackPredictor._build_config", MagicMock())
    @patch("deepdoctection.extern.tp.tpcompat.TensorpackPredictor.get_predictor", MagicMock())
    @patch("deepdoctection.extern.tpdetect.tp_predict_image", MagicMock(side_effect=get_mock_detection_results))
    def test_tp_frcnn_predicts_image(
        path_to_tp_frcnn_yaml: str, categories: Dict[str, str], np_image: ImageType
    ) -> None:
        """
        TP FRCNN calls predict_image and post processes DetectionResult correctly, e.g. adding class names
        """

        # Arrange
        frcnn = TPFrcnnDetector(path_yaml=path_to_tp_frcnn_yaml, path_weights="", categories=categories)

        # Act
        results = frcnn.predict(np_image)

        # Assert
        assert len(results) == 2
        first_detect_result = results[0]
        second_detect_result = results[1]
        assert first_detect_result.class_name == categories["2"]
        assert second_detect_result.class_name == categories["4"]
