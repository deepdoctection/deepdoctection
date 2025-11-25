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
from pathlib import Path
import numpy as np

from unittest.mock import MagicMock, patch

import pytest

from dd_datasets import Pubtabnet

import shared_test_utils as stu


def test_dataset_pubtabnet_returns_image(monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str) -> None:
    """
    test dataset pubtabnet returns image
    """

    monkeypatch.setattr("dd_core.mapper.pubstruct.load_bytes_from_pdf_file", lambda _fn: b"\x01\x02")
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.convert_pdf_bytes_to_np_array_v2",
        lambda *args, **kwargs: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    # Arrange
    pubtabnet = Pubtabnet()
    pubtabnet.dataflow.get_workdir=lambda: Path(dataset_test_base_dir) / pubtabnet.dataflow.location
    df = pubtabnet.dataflow.build()

    # Act
    df_list = stu.collect_datapoint_from_dataflow(df)
    assert len(df_list) == 3


def test_dataset_pubtabnet_with_load_image_returns_image(monkeypatch: pytest.MonkeyPatch,dataset_test_base_dir: str) -> None:
    """
    test dataset publaynet returns image
    """

    monkeypatch.setattr("dd_core.mapper.pubstruct.load_bytes_from_pdf_file", lambda _fn: b"\x01\x02")
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.convert_pdf_bytes_to_np_array_v2",
        lambda *args, **kwargs: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    # Arrange
    pubtabnet = Pubtabnet()
    pubtabnet.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / pubtabnet.dataflow.location
    df = pubtabnet.dataflow.build(load_image=True)

    # Act
    df_list = stu.collect_datapoint_from_dataflow(df)
    assert len(df_list) == 3
    assert df_list[0].image is not None
