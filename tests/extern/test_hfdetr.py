# -*- coding: utf-8 -*-
# File: test_hfdetr.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Testing module extern.hfdetr
"""


from typing import Dict, Sequence
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.extern.hfdetr import HFDetrDerivedDetector
from deepdoctection.utils.detection_types import ImageType
from deepdoctection.utils.file_utils import pytorch_available, transformers_available
from deepdoctection.utils.settings import ObjectTypes

if pytorch_available():
    import torch

if transformers_available():
    from transformers import BatchFeature


def get_mock_features() -> "BatchFeature":
    """
    returns batch features inputs
    """
    data = {"pixel_values": torch.ones(1, 20, 20), "pixel_mask": torch.ones(1, 3, 20, 20)}
    return BatchFeature(data=data)


def get_mock_predictions() -> Dict[str, "torch.Tensor"]:
    """
    return detr predictions
    """

    outputs = {
        "encoder_last_hidden_state": torch.ones(10, 20, 3),
        "last_hidden_state": torch.ones(1, 2, 256),
        "logits": torch.tensor(
            [
                [
                    [-12.0546, -10.9208, -5.9281, -7.7109, -6.0824, -7.6029, 4.9027],
                    [-12.7695, -9.4253, -4.9289, -3.3625, -6.5684, -6.4647, 4.4205],
                ]
            ]
        ),
        "pred_boxes": torch.tensor([[[0.4978, 0.6739, 0.8992, 0.2794], [0.4972, 0.4949, 0.9043, 0.3606]]]),
    }

    return outputs


def get_mock_post_process_features() -> Sequence[Dict[str, "torch.Tensor"]]:
    """
    return feature_extractor.post_process_object_detection
    """

    return [
        {
            "boxes": torch.Tensor([[20, 10, 10, 10], [30, 40, 50, 60]]),
            "labels": torch.Tensor([0, 2]).to(torch.uint8),
            "scores": torch.Tensor([0.4695, 0.9522]),
        }
    ]


class TestHFDetrDerivedDetector:
    """
    Test HFDetrDerivedDetector
    """

    @staticmethod
    @mark.requires_pt
    @patch("deepdoctection.extern.hfdetr.HFDetrDerivedDetector.set_model", MagicMock(return_value=MagicMock()))
    @patch("deepdoctection.extern.hfdetr.HFDetrDerivedDetector.set_pre_processor", MagicMock())
    @patch("deepdoctection.extern.hfdetr.PretrainedConfig.from_pretrained", MagicMock())
    def test_hf_detr_predicts_image(detr_categories: Dict[str, ObjectTypes], np_image: ImageType) -> None:
        """
        D2 FRCNN calls predict_image and post processes DetectionResult correctly, e.g. adding class names
        """

        # Arrange
        detr = HFDetrDerivedDetector(
            path_config_json="",
            path_weights="",
            path_feature_extractor_config_json="",
            categories=detr_categories,
            device="cpu",
        )
        detr.hf_detr_predictor = MagicMock(return_value=get_mock_predictions())
        detr.feature_extractor = MagicMock(return_value=get_mock_features())
        detr.feature_extractor.post_process_object_detection = MagicMock(return_value=get_mock_post_process_features())
        # Act
        results = detr.predict(np_image)

        # Assert
        assert len(results) == 2
        assert results[0].class_id == 3
        assert results[1].class_id == 1
