# -*- coding: utf-8 -*-
# File: test_adapter.py

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
Testing the module datasets.adapter
"""

from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.datasets import LayoutTest
from deepdoctection.utils import detectron2_available, pytorch_available

from ..test_utils import collect_datapoint_from_dataflow, get_test_path

if pytorch_available() and detectron2_available():
    from deepdoctection.datasets.adapter import DatasetAdapter
    from deepdoctection.mapper.d2struct import image_to_d2_frcnn_training


@mark.requires_pt
@patch("deepdoctection.mapper.tpstruct.os.path.isfile", MagicMock(return_value=True))
def test_adapter_with_cached_dataset() -> None:
    """
    test DatasetAdapter wraps a dd dataset into a torch dataset and yields datapoints correctly when the whole dataset
    is cached
    """

    # Arrange
    layouttest = LayoutTest()
    layouttest.dataflow.get_workdir = get_test_path  # type: ignore
    layouttest.dataflow.splits = {"test": ""}
    layouttest.dataflow.annotation_files = {"test": "test_layout.jsonl"}

    adapter = DatasetAdapter(layouttest, True, image_to_d2_frcnn_training(False))

    # Act & Assert
    dataset_iter = iter(adapter)
    df_list = collect_datapoint_from_dataflow(dataset_iter, max_datapoints=4)
    assert len(df_list) == 4


@mark.requires_pt
@patch("deepdoctection.mapper.tpstruct.os.path.isfile", MagicMock(return_value=True))
def test_adapter_with_uncached_dataset() -> None:
    """
    test DatasetAdapter wraps a dd dataset into a torch dataset and yields datapoints correctly when the dataset
    is not cached
    """

    # Arrange
    layouttest = LayoutTest()
    layouttest.dataflow.get_workdir = get_test_path  # type: ignore
    layouttest.dataflow.splits = {"test": ""}
    layouttest.dataflow.annotation_files = {"test": "test_layout.jsonl"}

    adapter = DatasetAdapter(layouttest, False, image_to_d2_frcnn_training(False))

    # Act & Assert
    dataset_iter = iter(adapter)
    df_list = collect_datapoint_from_dataflow(dataset_iter, max_datapoints=4)
    assert len(df_list) == 2
