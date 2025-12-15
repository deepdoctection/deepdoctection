# -*- coding: utf-8 -*-
# File: test_funsd.py

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
Testing module datasets.instances.funsd
"""

from pathlib import Path

import numpy as np
import pytest

import shared_test_utils as stu
from dd_core.datapoint.image import Image
from dd_core.utils.object_types import LayoutType, WordType
from dd_datasets import Funsd


def test_dataset_funsd_returns_image_and_annotations(
    monkeypatch: pytest.MonkeyPatch, dataset_test_base_dir: str
) -> None:
    # Patch image loader to return a white image
    monkeypatch.setattr(
        "dd_core.mapper.xfundstruct.load_image_from_file",
        lambda _fn: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )

    funsd = Funsd()
    funsd.dataflow.get_workdir = lambda: Path(dataset_test_base_dir) / funsd.dataflow.location  # type: ignore
    funsd.dataflow.splits = {"test": ""}

    df = funsd.dataflow.build()
    df_list: list[Image] = stu.collect_datapoint_from_dataflow(df)

    assert len(df_list) == 1
    dp = df_list[0]

    word = dp.get_annotation(category_names=LayoutType.WORD)[0]
    assert word.get_sub_category(WordType.TOKEN_CLASS) is not None
    assert word.get_sub_category(WordType.CHARACTERS) is not None
    assert word.get_sub_category(WordType.TAG) is not None
    assert word.get_sub_category(WordType.TOKEN_TAG) is not None

    text = dp.get_annotation(category_names=LayoutType.TEXT)[0]
    assert text.get_sub_category(WordType.TOKEN_CLASS) is not None
