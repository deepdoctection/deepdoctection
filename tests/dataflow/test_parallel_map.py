# -*- coding: utf-8 -*-
# File: test_parallel_map.py

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
Testing module dataflow.parallel_map
"""
from typing import no_type_check

import numpy as np
from pytest import mark

from deepdoctection.dataflow import DataFlow, FakeData, MultiThreadMapData

from ..test_utils import collect_datapoint_from_dataflow


@no_type_check
def map_to_one(dp):
    """map to one"""
    return np.ones(dp[0].shape)


@mark.basic
def test_multithread_map_data_non_strict() -> None:
    """Test MultiThreadMapData non strict"""

    # Arrange
    df: DataFlow
    df = FakeData(shapes=[[4, 7, 3]], domain=(0, 1), size=10)

    # Act
    df = MultiThreadMapData(df, num_thread=4, map_func=map_to_one, buffer_size=10)
    output = collect_datapoint_from_dataflow(df, max_datapoints=20)

    # Assert
    assert len(output) == 20


@mark.basic
def test_multithread_map_data_strict() -> None:
    """Test MultiThreadMapData strict"""

    # Arrange
    df: DataFlow
    df = FakeData(shapes=[[4, 7, 3]], domain=(0, 1), size=10)

    # Act
    df = MultiThreadMapData(df, num_thread=4, map_func=map_to_one, buffer_size=11, strict=True)
    output = collect_datapoint_from_dataflow(df, max_datapoints=20)

    # Assert
    assert len(output) == 10
