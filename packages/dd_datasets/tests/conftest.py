# -*- coding: utf-8 -*-
# File: conftest.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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

from pathlib import Path

import jsonlines
import numpy as np
import pytest

import shared_test_utils as stu
from dd_core.mapper.prodigystruct import prodigy_to_image
from dd_datasets import Fintabnet, Pubtabnet


@pytest.fixture(name="test_layout")
def fixture_test_layout(): # type:ignore
    path = stu.asset_path("testlayout")

    def test_layout(raw: bool = False): # type:ignore
        with jsonlines.open(path, "r") as reader:
            if raw:
                return list(reader)
            else:
                datapoints_raw = list(reader)
                images = []
                for dp in datapoints_raw:
                    images.append(
                        prodigy_to_image(
                            categories_name_as_key={"text": 1, "title": 2, "list": 3, "table": 4, "figure": 5},
                            load_image=True,
                            fake_score=False,
                        )(dp)
                    )
                return images

    return test_layout


@pytest.fixture(name="dataset_test_base_dir")
def fixture_dataset_test_base_dir() -> str:
    return (Path(__file__).parent / "assets" / "datasets").as_posix()


@pytest.fixture()
def fintabnet(monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str)-> Fintabnet:
    monkeypatch.setattr("dd_core.mapper.pubstruct.load_bytes_from_pdf_file", lambda _fn: b"\x01\x02")
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.convert_pdf_bytes_to_np_array_v2",
        lambda *args, **kwargs: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )
    ds = Fintabnet()
    ds.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / ds.dataflow.location
    return ds


@pytest.fixture()
def pubtabnet(dataset_test_base_dir: str)-> Pubtabnet:
    ds = Pubtabnet()
    ds.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / ds.dataflow.location
    return ds
