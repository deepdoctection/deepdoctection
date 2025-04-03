# -*- coding: utf-8 -*-
# File: test_d2_frcnn_train.py

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
Testing module train.d2_frcnn_train
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.datasets import DatasetBase
from deepdoctection.utils import detectron2_available

if detectron2_available():
    from deepdoctection.train.d2_frcnn_train import train_d2_faster_rcnn


def set_num_gpu_to_one() -> int:
    """
    set gpu number to one
    """
    return 1


@mark.requires_gpu
@patch("deepdoctection.train.d2_frcnn_train.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_one))
@patch("deepdoctection.train.d2_frcnn_train.D2Trainer.train")
def test_train_faster_rcnn(
    patch_train: Any, path_to_d2_frcnn_yaml: str, test_dataset: DatasetBase, tmp_path: Path
) -> None:
    """
    test train faster rcnn runs until "trainer.train()"
    """

    # Arrange
    train_d2_faster_rcnn(
        path_config_yaml=path_to_d2_frcnn_yaml,
        dataset_train=test_dataset,
        path_weights="",
        log_dir=str(tmp_path),
        dataset_val=test_dataset,
        metric_name="coco",
        pipeline_component_name="ImageLayoutService",
    )

    # Assert
    patch_train.assert_called_once()
