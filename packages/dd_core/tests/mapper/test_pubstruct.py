# -*- coding: utf-8 -*-
# File: test_pubstruct.py

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


import numpy as np
import pytest

from dd_core.mapper.pubstruct import pub_to_image
from dd_core.utils.object_types import LayoutType, TableType


@pytest.fixture
def categories_name_as_key() -> dict:
    return {
        LayoutType.CELL: 1,
        TableType.ITEM: 2,
        LayoutType.TABLE: 3,
        LayoutType.WORD: 4,
    }

def _count_cells_with_bbox(dp):
    return sum(1 for c in dp["html"]["cells"] if "bbox" in c)

def test_pub_to_image_basic(monkeypatch: pytest.MonkeyPatch, pubtabnet_datapoint, categories_name_as_key):
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.load_image_from_file",
        lambda fn: np.zeros((1200, 800, 3), dtype=np.uint8),
    )
    img = pub_to_image(categories_name_as_key, True, False, False, False, False, False)(pubtabnet_datapoint)
    assert img is not None
    cells = img.get_annotation(category_names=LayoutType.CELL)
    assert len(cells) == _count_cells_with_bbox(pubtabnet_datapoint)
    summary = img.summary
    assert summary.get_sub_category(TableType.NUMBER_OF_ROWS).category_id == 14
    assert summary.get_sub_category(TableType.NUMBER_OF_COLUMNS).category_id == 9

def test_pub_to_image_rows_cols_items(monkeypatch: pytest.MonkeyPatch, pubtabnet_datapoint, categories_name_as_key):
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.load_image_from_file",
        lambda fn: np.zeros((1000, 600, 3), dtype=np.uint8),
    )
    mapper = pub_to_image(categories_name_as_key, True, False, True, False, False, False)
    img = mapper(pubtabnet_datapoint)
    assert img is not None
    summary = img.summary
    n_rows = summary.get_sub_category(TableType.NUMBER_OF_ROWS).category_id
    n_cols = summary.get_sub_category(TableType.NUMBER_OF_COLUMNS).category_id
    items = img.get_annotation(category_names=TableType.ITEM)
    assert len(items) == n_rows + n_cols

def test_pub_to_image_dd_pipe_like(monkeypatch: pytest.MonkeyPatch, pubtabnet_datapoint, categories_name_as_key):
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.load_image_from_file",
        lambda fn: np.zeros((640, 480, 3), dtype=np.uint8),
    )
    mapper = pub_to_image(categories_name_as_key, True, False, True, True, False, False)
    img = mapper(pubtabnet_datapoint)
    assert img is not None
    tables = img.get_annotation(category_names=LayoutType.TABLE)
    assert len(tables) == 1
    table = tables[0]
    assert table.image is not None
    cells = img.get_annotation(category_names=LayoutType.CELL)
    words = img.get_annotation(category_names=LayoutType.WORD)
    # One word per cell in dd_pipe_like mapping
    assert len(words) == len(cells)
