# -*- coding: utf-8 -*-
# File: test_adapter.py

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

from dd_core.mapper import image_to_hf_detr_training
from dd_core.utils.file_utils import pytorch_available

if pytorch_available():
    from dd_datasets.adapter import DatasetAdapter


def _patch_pdf(monkeypatch):
    monkeypatch.setattr("dd_core.mapper.pubstruct.load_bytes_from_pdf_file", lambda _fn: b"\x01\x02")
    monkeypatch.setattr(
        "dd_core.mapper.pubstruct.convert_pdf_bytes_to_np_array_v2",
        lambda *args, **kwargs: np.ones((794, 596, 3), dtype=np.uint8) * 255,
    )


@pytest.mark.skipif(not pytorch_available(), reason="torch not installed")
def test_dataset_adapter_cache_d2_mapping(monkeypatch: pytest.MonkeyPatch, fintabnet):
    _patch_pdf(monkeypatch)
    adapter = DatasetAdapter(
        fintabnet,
        cache_dataset=True,
        number_repetitions=1,
        image_to_framework_func=image_to_hf_detr_training(),
        load_image=True,
    )
    assert len(adapter) == 4
    first = next(iter(adapter))
    # Mapping returns D2 dict
    assert isinstance(first, dict)
    for key in ("file_name", "width", "height", "image_id", "annotations"):
        assert key in first
    assert first["annotations"]


@pytest.mark.skipif(not pytorch_available(), reason="torch not installed")
def test_dataset_adapter_non_cache_infinite_raises(fintabnet):
    with pytest.raises(ValueError):
        DatasetAdapter(
            fintabnet,
            cache_dataset=False,
            number_repetitions=-1,
            image_to_framework_func=image_to_hf_detr_training(),
        )


@pytest.mark.skipif(not pytorch_available(), reason="torch not installed")
def test_dataset_adapter_non_cache_repetition(monkeypatch: pytest.MonkeyPatch, fintabnet):
    _patch_pdf(monkeypatch)
    adapter = DatasetAdapter(
        fintabnet,
        cache_dataset=False,
        number_repetitions=2,
        image_to_framework_func=image_to_hf_detr_training(),
        load_image=True,
    )
    # Collect 8 (4 datapoints * 2 repetitions)
    collected = []
    for i, dp in enumerate(adapter):
        collected.append(dp)
    assert len(collected) == 8


@pytest.mark.skipif(not pytorch_available(), reason="torch not installed")
def test_dataset_adapter_max_datapoints_limits(monkeypatch: pytest.MonkeyPatch, fintabnet):
    _patch_pdf(monkeypatch)
    adapter = DatasetAdapter(
        fintabnet,
        cache_dataset=True,
        number_repetitions=1,
        image_to_framework_func=image_to_hf_detr_training(),
        max_datapoints=2,
        load_image=True,
    )
    assert len(adapter) == 2
    all_items = list(iter(adapter))
    assert len(all_items) == 2


@pytest.mark.skipif(not pytorch_available(), reason="torch not installed")
def test_dataset_adapter_hf_detr_annotations_non_empty(monkeypatch: pytest.MonkeyPatch, fintabnet):
    _patch_pdf(monkeypatch)
    adapter = DatasetAdapter(
        fintabnet,
        cache_dataset=True,
        number_repetitions=1,
        image_to_framework_func=image_to_hf_detr_training(),
        load_image=True,
    )
    items = list(iter(adapter))
    assert all(item["annotations"] for item in items)
