# -*- coding: utf-8 -*-
# File: test_pubtables1m.py

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
Testing module datasets.instances.pubtables1m
"""


from pathlib import Path
import numpy as np
import pytest

import shared_test_utils as stu

from dd_core.utils.file_utils import lxml_available
from dd_datasets import Pubtables1MDet, Pubtables1MStruct

@pytest.mark.skipif(not lxml_available(), reason="lxml not installed")
def test_dataset_pubtables1m_det_returns_image(dataset_test_base_dir: str) -> None:

    pubtables = Pubtables1MDet()
    pubtables.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / pubtables.dataflow.location


    df = pubtables.dataflow.build()
    df_list = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 1


@pytest.mark.skipif(not lxml_available(), reason="lxml not installed")
def test_dataset_pubtables1m_det_with_load_image_returns_image(
    monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str
) -> None:
    monkeypatch.setattr(
        "dd_core.mapper.pascalstruct.load_image_from_file",
        lambda *args, **kwargs: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    pubtables = Pubtables1MDet()
    pubtables.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / pubtables.dataflow.location

    df = pubtables.dataflow.build(load_image=True)
    df_list = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 1
    assert df_list[0].image is not None


@pytest.mark.skipif(not lxml_available(), reason="lxml not installed")
def test_dataset_pubtables1m_struct_returns_image(dataset_test_base_dir: str) -> None:

    pubtables = Pubtables1MStruct()
    pubtables.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / pubtables.dataflow.location

    df = pubtables.dataflow.build()
    df_list = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 1


@pytest.mark.skipif(not lxml_available(), reason="lxml not installed")
def test_dataset_pubtables1m_struct_with_load_image_returns_image(
    monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str
) -> None:
    monkeypatch.setattr(
        "dd_core.mapper.pascalstruct.load_image_from_file",
        lambda *args, **kwargs: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    pubtables = Pubtables1MStruct()
    pubtables.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / pubtables.dataflow.location

    df = pubtables.dataflow.build(load_image=True)
    df_list = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 1
    assert df_list[0].image is not None

