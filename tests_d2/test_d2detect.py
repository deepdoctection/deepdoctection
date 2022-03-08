# -*- coding: utf-8 -*-
# File: test_d2detect.py

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
Testing module extern.d2detect
"""

from typing import Dict, List

from unittest.mock import MagicMock, patch
from pytest import mark, raises

from deepdoctection.utils.file_utils import pytorch_available, detectron2_available

from deepdoctection.extern.d2detect import D2FrcnnDetector
from deepdoctection.utils.detection_types import ImageType

if pytorch_available():
    import torch

if detectron2_available():
    from detectron2.structures import Instances, Boxes


def get_mock_instances() -> List[List[Dict[str,Instances]]]:
    """
    return Instances instance
    """
    pred_boxes = Boxes(torch.Tensor([[1.0, 1.6, 2.0, 4.6],[12.0, 12.0, 12.0, 12.0]]))
    scores = torch.Tensor([0.93,0.54])
    pred_classes = torch.Tensor([0,1]).to(torch.uint8)

    instance =  Instances((400,600))
    instance.pred_boxes = pred_boxes
    instance.scores = scores
    instance.pred_classes = pred_classes

    return [[{"instances": instance}]]


class TestD2FrcnnDetector:
    """
    Test D2FrcnnDetector
    """

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.utils.file_utils.detectron2_available",MagicMock(return_value=False))
    def test_d2_does_not_build_when_d2_not_available(path_to_d2_frcnn_yaml: str, categories: Dict[str,str]) -> None:
        """
        D2 FRCNN does only build when detectron2 is properly installed
        """

        # Arrange, Act & Assert
        with raises(ImportError):
            D2FrcnnDetector(path_yaml=path_to_d2_frcnn_yaml,path_weights="",categories=categories)

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.d2detect.D2FrcnnDetector.set_model", MagicMock(return_value=MagicMock))
    @patch("deepdoctection.extern.d2detect.D2FrcnnDetector._instantiate_d2_predictor", MagicMock())
    @patch("deepdoctection.extern.d2detect.D2FrcnnDetector.set_model", MagicMock(return_value=MagicMock))
    def test_d2_frcnn_predicts_image(path_to_d2_frcnn_yaml: str, categories: Dict[str,str], np_image: ImageType)-> None:
        """
        D2 FRCNN calls predict_image and post processes DetectionResult correctly, e.g. adding class names
        """

        # Arrange
        frcnn = D2FrcnnDetector(path_yaml=path_to_d2_frcnn_yaml,path_weights="",categories=categories)
        frcnn.d2_predictor = MagicMock(side_effect=get_mock_instances())
        # Act
        results = frcnn.predict(np_image)

        # Assert
        assert len(results)==2
        assert results[0].class_id == 1
        assert results[1].class_id == 2
