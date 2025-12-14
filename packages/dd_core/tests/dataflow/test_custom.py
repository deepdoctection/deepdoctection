# -*- coding: utf-8 -*-
# File: test_custom.py

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
Testing the module dataflow.custom
"""
from typing import Any

from pytest import mark

import shared_test_utils as stu
from dd_core.dataflow import CacheData, CustomDataFromIterable, CustomDataFromList


def rebalance_remove_first(lst: list[Any]) -> list[Any]:
    """
    Helper function to remove first element from list
    """
    return lst[1:]


def test_cache_data_caches_dataflow(simple_list_dataflow: Any) -> None:
    """
    Test that CacheData properly caches a dataflow after first iteration
    """
    # Arrange
    df = CacheData(simple_list_dataflow)

    # Act
    first_pass: list[list[str]] = stu.collect_datapoint_from_dataflow(df)
    second_pass: list[list[str]] = stu.collect_datapoint_from_dataflow(df)

    # Assert
    assert first_pass == second_pass
    assert len(first_pass) == 3
    assert first_pass == [["a", "b"], ["c", "d"], ["e", "f"]]


def test_cache_data_get_cache(simple_dict_dataflow: Any) -> None:
    """
    Test CacheData get_cache method returns complete list
    """
    # Arrange
    df = CacheData(simple_dict_dataflow)

    # Act
    cached_list = df.get_cache()

    # Assert
    assert len(cached_list) == 3
    assert cached_list[0] == {"key1": "a", "key2": 1}
    assert cached_list[1] == {"key1": "b", "key2": 2}
    assert cached_list[2] == {"key1": "c", "key2": 3}


def test_cache_data_with_shuffle(simple_list_dataflow: Any) -> None:
    """
    Test CacheData shuffle option
    """
    # Arrange
    df = CacheData(simple_list_dataflow, shuffle=True)

    # Act
    first_pass: list[list[str]] = stu.collect_datapoint_from_dataflow(df)

    # Assert - check all elements are present (order may vary)
    assert len(first_pass) == 3
    assert set(map(tuple, first_pass)) == {("a", "b"), ("c", "d"), ("e", "f")}


def test_custom_data_from_list_with_max_datapoints(simple_list: list[str]) -> None:
    """
    Test CustomDataFromList respects max_datapoints limit
    """
    # Arrange
    df = CustomDataFromList(simple_list, max_datapoints=3)

    # Act
    result: list[list[str]] = stu.collect_datapoint_from_dataflow(df)

    # Assert
    assert len(result) == 3
    assert result == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert len(df) == 3


def test_custom_data_from_list_with_rebalance_func(simple_dict_list: list[dict[str, Any]]) -> None:
    """
    Test CustomDataFromList with rebalance function
    """
    # Arrange
    df = CustomDataFromList(simple_dict_list, rebalance_func=rebalance_remove_first)

    # Act
    result: list[dict[str,Any]] = stu.collect_datapoint_from_dataflow(df)

    # Assert
    assert len(result) == 2
    assert result[0]["key2"] == 2
    assert result[1]["key2"] == 3


def test_custom_data_from_iterable(simple_list: list[str]) -> None:
    """
    Test CustomDataFromIterable with max_datapoints
    """
    # Arrange
    iterable = iter(simple_list)
    df = CustomDataFromIterable(iterable, max_datapoints=2)

    # Act
    result:list[list[str]] = stu.collect_datapoint_from_dataflow(df)

    # Assert
    assert len(result) == 2
    assert result == [["a", "b"], ["c", "d"]]
