# -*- coding: utf-8 -*-
# File: test_parallel_map.py

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

"""
Testing module dataflow.parallel_map
"""
from typing import Any, Dict, List

import pytest

import shared_test_utils as stu
from dd_core.dataflow import DataFromList, MultiProcessMapData, MultiThreadMapData
from dd_core.utils import file_utils as fu


@pytest.mark.skipif(not fu.pyzmq_available(), reason="Pyzmq is not installed")
def test_multi_thread_map_data_applies_mapping_function(simple_dict_dataflow: DataFromList) -> None:
    """
    Test MultiThreadMapData applies a mapping function correctly across multiple threads in non-strict mode
    """

    # Arrange
    def double_key2(dp: Dict[str, Any]) -> Dict[str, Any]:
        dp["key2"] = dp["key2"] * 2
        return dp

    df = MultiThreadMapData(simple_dict_dataflow, num_thread=2, map_func=double_key2, buffer_size=5, strict=True)

    # Act
    output:list[dict[str,Any]] = stu.collect_datapoint_from_dataflow(df=df, max_datapoints=3)

    # Assert
    assert len(output) == 3
    # Check that all values have been doubled (order may vary due to parallelism)
    key2_values = sorted([dp["key2"] for dp in output])
    assert key2_values == [2, 4, 6]
    # Check that key1 values are preserved
    key1_values = sorted([dp["key1"] for dp in output])
    assert key1_values == ["a", "b", "c"]


@pytest.mark.skipif(not fu.pyzmq_available(), reason="Pyzmq is not installed")
def test_multi_process_map_data_applies_mapping_function(simple_list_dataflow: DataFromList) -> None:
    """
    Test MultiProcessMapData applies a mapping function correctly across multiple processes in strict mode
    """

    # Arrange
    def reverse_list(dp: List[str]) -> List[str]:
        return dp[::-1]

    df = MultiProcessMapData(simple_list_dataflow, num_proc=2, map_func=reverse_list, buffer_size=5, strict=True)

    # Act
    output:list[dict[str,Any]] = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 3
    # Check that all lists have been reversed (order may vary due to parallelism)
    output_sorted = sorted(output)
    assert output_sorted == [["b", "a"], ["d", "c"], ["f", "e"]]
