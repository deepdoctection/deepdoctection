# -*- coding: utf-8 -*-
# File: test_pubtabnet.py

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
Testing module datasets.instances.pubtabnet
"""

from unittest.mock import MagicMock, patch

from deepdoctection.datasets import Pubtabnet

from ...test_utils import collect_datapoint_from_dataflow, get_test_path
from .conftest import get_white_image


def test_dataset_pubtabnet_returns_image() -> None:
    """
    test dataset pubtabnet returns image
    """

    # Arrange
    pubtabnet = Pubtabnet()
    pubtabnet.dataflow.get_workdir = get_test_path  # type: ignore
    pubtabnet.dataflow.annotation_files = {"all": "test_file.jsonl"}
    df = pubtabnet.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 3


@patch("deepdoctection.mapper.pubstruct.load_image_from_file", MagicMock(side_effect=get_white_image))
def test_dataset_pubtabnet_with_load_image_returns_image() -> None:
    """
    test dataset publaynet returns image
    """

    # Arrange
    pubtabnet = Pubtabnet()
    pubtabnet.dataflow.get_workdir = get_test_path  # type: ignore
    pubtabnet.dataflow.annotation_files = {"all": "test_file.jsonl"}
    df = pubtabnet.dataflow.build(load_image=True)

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 3
