# -*- coding: utf-8 -*-
# File: test_fintabnet.py

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
Testing module datasets.instances.fintabnet
"""

from unittest.mock import MagicMock, patch

from deepdoctection.datasets import Fintabnet

from ...test_utils import collect_datapoint_from_dataflow, get_test_path
from .conftest import get_white_image


@patch("deepdoctection.mapper.pubstruct.convert_pdf_bytes_to_np_array_v2", MagicMock(side_effect=get_white_image))
@patch("deepdoctection.mapper.pubstruct.load_bytes_from_pdf_file", MagicMock(return_value=b"\x01\x02"))
@patch("deepdoctection.datasets.instances.fintabnet.set_mp_spawn", MagicMock())
def test_dataset_fintabnet_returns_image() -> None:
    """
    test dataset fintabnet returns image
    """

    # Arrange
    fintabnet = Fintabnet()
    fintabnet.dataflow.get_workdir = get_test_path  # type: ignore
    fintabnet.dataflow.annotation_files = {"val": "test_file_fintab.jsonl"}
    df = fintabnet.dataflow.build(use_multi_proc=False)

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 4


@patch("deepdoctection.mapper.pubstruct.convert_pdf_bytes_to_np_array_v2", MagicMock(side_effect=get_white_image))
@patch("deepdoctection.mapper.pubstruct.load_bytes_from_pdf_file", MagicMock(return_value=b"\x01\x02"))
@patch("deepdoctection.datasets.instances.fintabnet.set_mp_spawn", MagicMock())
def test_dataset_fintabnet_with_load_image_returns_image() -> None:
    """
    test dataset fintabnet returns image
    """

    # Arrange
    fintabnet = Fintabnet()
    fintabnet.dataflow.get_workdir = get_test_path  # type: ignore
    fintabnet.dataflow.annotation_files = {"val": "test_file_fintab.jsonl"}
    df = fintabnet.dataflow.build(load_image=True, use_multi_proc=False)

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 4
    assert df_list[0].image is not None
