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


import pytest

import shared_test_utils as stu
from dd_datasets.base import SplitDataFlow


def test_splitdataflow_default_train_split(test_layout):
    images = test_layout(raw=True)
    sdf = SplitDataFlow(train=images, val=[], test=None)
    df = sdf.build()
    collected = stu.collect_datapoint_from_dataflow(df)
    assert collected == images
    assert len(collected) == len(images)


def test_splitdataflow_val_split_max_datapoints_str(test_layout):
    images = test_layout(raw=False)
    sdf = SplitDataFlow(train=[], val=images, test=None)
    df = sdf.build(split="val", max_datapoints=1)
    collected = stu.collect_datapoint_from_dataflow(df)
    assert len(collected) == 1
    assert collected[0] in images


def test_splitdataflow_invalid_split_type_raises(test_layout):
    images = test_layout(raw=True)
    sdf = SplitDataFlow(train=images, val=[], test=None)
    with pytest.raises(ValueError):
        sdf.build(split=123)
