# -*- coding: utf-8 -*-
# File: test_base.py

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

import numpy as np
import pytest

import shared_test_utils as stu
from dd_core.datapoint.image import Image
from dd_datasets import MergeDataset
from dd_datasets.base import SplitDataFlow


def test_splitdataflow_default_train_split(test_layout) -> None:  # type: ignore
    images = test_layout(raw=True)
    sdf = SplitDataFlow(train=images, val=[], test=None)
    df = sdf.build()
    collected: list[Image] = stu.collect_datapoint_from_dataflow(df)
    assert collected == images
    assert len(collected) == len(images)


def test_splitdataflow_val_split_max_datapoints_str(test_layout) -> None:  # type: ignore
    images = test_layout(raw=False)
    sdf = SplitDataFlow(train=[], val=images, test=None)
    df = sdf.build(split="val", max_datapoints=1)
    collected: list[Image] = stu.collect_datapoint_from_dataflow(df)
    assert len(collected) == 1
    assert collected[0] in images


def test_splitdataflow_invalid_split_type_raises(test_layout) -> None:  # type: ignore
    images = test_layout(raw=True)
    sdf = SplitDataFlow(train=images, val=[], test=None)
    with pytest.raises(ValueError):
        sdf.build(split=123)


def test_merge_dataset_build_concatenates_datapoints(fintabnet, pubtabnet) -> None:  # type: ignore
    merge = MergeDataset(fintabnet, pubtabnet)
    df = merge.dataflow.build(split="val")
    out: list[Image] = stu.collect_datapoint_from_dataflow(df)
    assert len(out) == 4 + 3  # fintabnet val + pubtabnet val


def test_merge_dataset_categories_union(fintabnet, pubtabnet) -> None:  # type: ignore
    merge = MergeDataset(fintabnet, pubtabnet)
    cats = merge.dataflow.categories.get_categories(as_dict=False, init=True)
    assert "table" in cats
    assert "cell" in cats
    assert "item" in cats
    assert "word" in cats


def test_merge_dataset_explicit_dataflows(fintabnet, pubtabnet) -> None:  # type: ignore
    df_fn = fintabnet.dataflow.build(split="val", max_datapoints=2)
    df_pt = pubtabnet.dataflow.build(
        split="train", max_datapoints=1
    )  # there is no train split in for this test setting
    merge = MergeDataset(fintabnet, pubtabnet)
    merge.explicit_dataflows(df_fn, df_pt)
    df = merge.dataflow.build()
    out: list[Image] = stu.collect_datapoint_from_dataflow(df)
    assert len(out) == 2


def test_merge_dataset_buffer_and_split_datasets(  # type: ignore
    monkeypatch: pytest.MonkeyPatch, fintabnet, pubtabnet
) -> None:
    # Deterministic split for 7 datapoints -> train:5, val:1, test:1
    monkeypatch.setattr(
        np.random,
        "binomial",
        lambda n, p, size: np.array([0, 0, 0, 0, 1, 1, 0]),  # 7 samples: zeros->train, ones->val/test
    )
    merge = MergeDataset(fintabnet, pubtabnet)
    merge.buffer_datasets(split="val")
    merge.split_datasets(ratio=0.3, add_test=True)
    split_ids = merge.get_ids_by_split()
    assert len(split_ids["train"]) == 5
    assert len(split_ids["val"]) == 1
    assert len(split_ids["test"]) == 1


def test_merge_dataset_create_split_by_id_reproduces(  # type: ignore
    monkeypatch: pytest.MonkeyPatch, fintabnet, pubtabnet
) -> None:
    monkeypatch.setattr(np.random, "binomial", lambda n, p, size: np.array([0, 0, 0, 0, 1, 1, 0]))
    merge = MergeDataset(fintabnet, pubtabnet)
    merge.buffer_datasets(split="val")
    merge.split_datasets(ratio=0.3, add_test=True)
    original_split = merge.get_ids_by_split()

    # Reproduce
    merge2 = MergeDataset(fintabnet, pubtabnet)
    merge2.create_split_by_id(original_split, split="val")
    reproduced_split = merge2.get_ids_by_split()

    assert {k: len(v) for k, v in reproduced_split.items()} == {k: len(v) for k, v in original_split.items()}


def test_merge_dataset_explicit_dataflows_warning(fintabnet, pubtabnet, caplog) -> None:  # type: ignore
    df_fn = fintabnet.dataflow.build(split="val", max_datapoints=1)
    df_pt = pubtabnet.dataflow.build(split="val", max_datapoints=1)
    merge = MergeDataset(fintabnet, pubtabnet)
    merge.explicit_dataflows(df_fn, df_pt)
    caplog.clear()
    _ = stu.collect_datapoint_from_dataflow(merge.dataflow.build())
    # Expect info log about using explicit dataflows
    assert any("explicitly passed configuration" in rec.getMessage() for rec in caplog.records)
