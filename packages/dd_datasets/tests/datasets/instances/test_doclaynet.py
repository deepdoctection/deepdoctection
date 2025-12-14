# -*- coding: utf-8 -*-
# File: test_doclaynet.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Testing module datasets.instances.doclaynet
"""

from pathlib import Path

import numpy as np
import pytest

import shared_test_utils as stu
from dd_core.datapoint.image import Image
from dd_datasets import DocLayNet, DocLayNetSeq


def test_dataset_doclaynet_returns_image(monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str) -> None:
    """
    test dataset DocLayNetSeqBuilder returns image
    """
    # Patch image loader to return a white image
    monkeypatch.setattr(
        "dd_core.utils.fs.load_image_from_file",
        lambda _fn: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    # Arrange
    doclaynet = DocLayNet()
    doclaynet.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / doclaynet.dataflow.location # type: ignore
    doclaynet.dataflow.annotation_files = {"val": "test_doclaynet.json"}
    df = doclaynet.dataflow.build()

    # Act
    df_list: list[Image] = stu.collect_datapoint_from_dataflow(df)
    assert len(df_list) == 2


def test_dataset_doclaynet_seq_returns_image(monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str) -> None:
    """
    test dataset DocLayNetSeq returns image
    """
    # Patch image loader to return a white image
    monkeypatch.setattr(
        "dd_datasets.instances.doclaynet.load_image_from_file",
        lambda _fn: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    # Arrange
    doclaynet = DocLayNetSeq()
    doclaynet.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / doclaynet.dataflow.location # type: ignore
    doclaynet.dataflow.annotation_files = {"val": "test_doclaynet.json"}
    df = doclaynet.dataflow.build()

    # Act
    df_list: list[Image] = stu.collect_datapoint_from_dataflow(df)
    assert len(df_list) == 2
