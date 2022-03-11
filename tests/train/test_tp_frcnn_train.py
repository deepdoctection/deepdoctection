# -*- coding: utf-8 -*-
# File: test_tp_frcnn_train.py

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
Testing module train.tp_frcnn_train
"""
from typing import Any
from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.datasets import DatasetBase
from deepdoctection.utils.file_utils import tf_available
from deepdoctection.utils.metacfg import set_config_by_yaml

from ..test_utils import collect_datapoint_from_dataflow

if tf_available():
    from deepdoctection.extern.tp.tpfrcnn.config.config import model_frcnn_config
    from deepdoctection.train.tp_frcnn_train import get_train_dataflow, train_faster_rcnn


def set_num_gpu_to_one() -> int:
    """
    set gpu number to one
    """
    return 1


@mark.requires_tf
@patch("deepdoctection.mapper.tpstruct.os.path.isfile", MagicMock(return_value=True))
@patch("deepdoctection.train.tp_frcnn_train.set_mp_spawn")
def test_get_train_dataflow(
    set_mp_spawn: Any, test_dataset: DatasetBase, path_to_tp_frcnn_yaml: str  # pylint: disable=W0613
) -> None:
    """
    test get train dataflow
    """

    # Arrange
    config = set_config_by_yaml(path_to_tp_frcnn_yaml)
    categories = test_dataset.dataflow.categories.get_categories(filtered=True)  # type: ignore
    model_frcnn_config(config, categories)  # type: ignore

    # Act
    df = get_train_dataflow(test_dataset, config, False)
    df_list = collect_datapoint_from_dataflow(df)

    # Assert
    assert len(df_list) == 2
    dp = df_list[0]
    assert "image" in dp
    assert dp["image"].ndim == 3
    assert "gt_boxes" in dp
    assert "gt_labels" in dp


@mark.requires_tf
@patch("deepdoctection.mapper.tpstruct.os.path.isfile", MagicMock(return_value=True))
@patch("deepdoctection.extern.tp.tpcompat.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_one))
@patch("deepdoctection.extern.tp.tpfrcnn.config.config.get_num_gpu", MagicMock(side_effect=set_num_gpu_to_one))
@patch("deepdoctection.train.tp_frcnn_train.ModelSaver")
@patch("deepdoctection.train.tp_frcnn_train.PeriodicCallback")
@patch("tensorpack.utils.logger.set_logger_dir")
@patch("deepdoctection.train.tp_frcnn_train.launch_train_with_config")
@patch("deepdoctection.train.tp_frcnn_train.set_mp_spawn")
def test_train_faster_rcnn(
    set_mp_spawn: Any,  # pylint: disable=W0613
    patch_launch_train_with_config: Any,
    patch_set_logger_dir: Any,  # pylint: disable=W0613
    patch_periodic_cb: Any,  # pylint: disable=W0613
    patch_model_saver: Any,  # pylint: disable=W0613
    path_to_tp_frcnn_yaml: str,
    test_dataset: DatasetBase,
) -> None:
    """
    test train faster rcnn runs until "launch_train_with_config"
    """

    # Arrange
    train_faster_rcnn(
        path_config_yaml=path_to_tp_frcnn_yaml,
        dataset_train=test_dataset,
        path_weights="",
        log_dir="/test/log_dir",
        dataset_val=test_dataset,
        metric_name="coco",
        pipeline_component_name="ImageLayoutService",
    )

    # Assert
    patch_launch_train_with_config.assert_called_once()
