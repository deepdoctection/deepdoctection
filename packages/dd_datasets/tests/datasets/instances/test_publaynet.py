# -*- coding: utf-8 -*-
# File: test_publaynet.py

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
Testing module datasets.instances.publaynet
"""


from pathlib import Path

import numpy as np
import pytest

import shared_test_utils as stu
from dd_core.datapoint.image import Image
from dd_datasets import Publaynet


def test_dataset_publaynet_returns_image(monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str) -> None:
    publaynet = Publaynet()
    publaynet.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / publaynet.dataflow.location  # type: ignore
    publaynet.dataflow.annotation_files = {"val": "publaynet.json"}

    df = publaynet.dataflow.build()
    df_list: list[Image] = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 6


def test_dataset_publaynet_with_load_image_returns_image(
    monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str
) -> None:
    monkeypatch.setattr(
        "dd_core.mapper.cocostruct.load_image_from_file",
        lambda _fn: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    publaynet = Publaynet()
    publaynet.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / publaynet.dataflow.location  # type: ignore
    publaynet.dataflow.annotation_files = {"val": "publaynet.json"}

    df = publaynet.dataflow.build(load_image=True)
    df_list: list[Image] = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 6
    assert df_list[0].image is not None
