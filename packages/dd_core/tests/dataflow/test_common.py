# -*- coding: utf-8 -*-
# File: test_common.py

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
Testing the module dataflow.common
"""

from typing import Any, Dict

import shared_test_utils as stu
from dd_core.dataflow import DataFromList
from dd_core.dataflow.common import (
    BatchData,
    ConcatData,
    FlattenData,
    JoinData,
    MapData,
    MapDataComponent,
    RepeatedData,
    TestDataSpeed,
)


def test_test_data_speed(simple_list_dataflow: DataFromList) -> None:
    """
    Test TestDataSpeed runs and produces the same datapoints
    """
    # Arrange
    df = TestDataSpeed(simple_list_dataflow, size=2, warmup=0)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 3
    assert output[0] == ["a", "b"]
    assert output[1] == ["c", "d"]
    assert output[2] == ["e", "f"]


def test_flatten_data_with_lists(simple_list_dataflow: DataFromList) -> None:
    """
    Test FlattenData flattens list elements correctly
    """
    # Arrange
    df = FlattenData(simple_list_dataflow)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 6
    assert output[0] == ["a"]
    assert output[1] == ["b"]
    assert output[2] == ["c"]
    assert output[3] == ["d"]
    assert output[4] == ["e"]
    assert output[5] == ["f"]


def test_map_data_adds_key_value(simple_dict_dataflow: DataFromList) -> None:
    """
    Test MapData applies a simple function that adds a key-value pair
    """

    # Arrange
    def add_new_key(dp: Dict[str, Any]) -> Dict[str, Any]:
        dp["new_key"] = "new_value"
        return dp

    df = MapData(simple_dict_dataflow, add_new_key)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 3
    assert output[0]["new_key"] == "new_value"
    assert output[0]["key1"] == "a"
    assert output[1]["new_key"] == "new_value"
    assert output[2]["new_key"] == "new_value"


def test_map_data_component_with_dict_index(simple_dict_dataflow: DataFromList) -> None:
    """
    Test MapDataComponent modifies a specific component of dict datapoints using string index
    """

    # Arrange
    def multiply_by_10(val: int) -> int:
        return val * 10

    df = MapDataComponent(simple_dict_dataflow, multiply_by_10, index="key2")

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 3
    assert output[0]["key2"] == 10
    assert output[0]["key1"] == "a"
    assert output[1]["key2"] == 20
    assert output[2]["key2"] == 30


def test_map_data_component_with_list_index(simple_list_dataflow: DataFromList) -> None:
    """
    Test MapDataComponent modifies a specific component of list datapoints using int index
    """

    # Arrange
    def uppercase(val: str) -> str:
        return val.upper()

    df = MapDataComponent(simple_list_dataflow, uppercase, index=0)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 3
    assert output[0][0] == "A"
    assert output[0][1] == "b"
    assert output[1][0] == "C"
    assert output[2][0] == "E"


def test_repeated_data_finite(simple_list_dataflow: DataFromList) -> None:
    """
    Test RepeatedData repeats dataflow a fixed number of times
    """
    # Arrange
    df = RepeatedData(simple_list_dataflow, num=3)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 9
    assert len(df) == 9


def test_repeated_data_infinite(simple_list_dataflow: DataFromList) -> None:
    """
    Test RepeatedData with num=-1 repeats infinitely (collect limited number)
    """
    # Arrange
    df = RepeatedData(simple_list_dataflow, num=-1)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df, max_datapoints=10)

    # Assert
    assert len(output) == 10
    # Verify it cycles through the data
    assert output[0] == ["a", "b"]
    assert output[3] == ["a", "b"]  # After cycling once
    assert output[6] == ["a", "b"]  # After cycling twice


def test_concat_data_combines_dataflows(simple_list_dataflow: DataFromList, simple_dict_dataflow: DataFromList) -> None:
    """
    Test ConcatData concatenates multiple dataflows sequentially
    """
    # Arrange
    df1 = DataFromList([["x"], ["y"]], shuffle=False)
    df2 = DataFromList([["z"], ["w"]], shuffle=False)
    df = ConcatData([df1, df2])

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 4
    assert len(df) == 4
    assert output[0] == ["x"]
    assert output[1] == ["y"]
    assert output[2] == ["z"]
    assert output[3] == ["w"]


def test_concat_data_preserves_order() -> None:
    """
    Test ConcatData preserves the order of dataflows
    """
    # Arrange
    df1 = DataFromList([{"id": 1}, {"id": 2}], shuffle=False)
    df2 = DataFromList([{"id": 3}, {"id": 4}], shuffle=False)
    df3 = DataFromList([{"id": 5}], shuffle=False)
    df = ConcatData([df1, df2, df3])

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 5
    assert [dp["id"] for dp in output] == [1, 2, 3, 4, 5]


def test_join_data_with_lists() -> None:
    """
    Test JoinData joins list-based dataflows
    """
    # Arrange
    df1 = DataFromList([["a"], ["b"], ["c"]], shuffle=False)
    df2 = DataFromList([["x"], ["y"], ["z"]], shuffle=False)
    df = JoinData([df1, df2])

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 3
    assert len(df) == 3
    assert output[0] == ["a", "x"]
    assert output[1] == ["b", "y"]
    assert output[2] == ["c", "z"]


def test_join_data_with_dicts() -> None:
    """
    Test JoinData joins dict-based dataflows
    """
    # Arrange
    df1 = DataFromList([{"key1": "a"}, {"key1": "b"}], shuffle=False)
    df2 = DataFromList([{"key2": "x"}, {"key2": "y"}], shuffle=False)
    df = JoinData([df1, df2])

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 2
    assert output[0] == {"key1": "a", "key2": "x"}
    assert output[1] == {"key1": "b", "key2": "y"}


def test_batch_data_full_batches(simple_list_dataflow: DataFromList) -> None:
    """
    Test BatchData creates full batches without remainder
    """
    # Arrange
    df = DataFromList([["a"], ["b"], ["c"], ["d"], ["e"], ["f"]], shuffle=False)
    df_batch = BatchData(df, batch_size=2, remainder=False)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df_batch)

    # Assert
    assert len(output) == 3
    assert len(df_batch) == 3
    assert output[0] == [["a"], ["b"]]
    assert output[1] == [["c"], ["d"]]
    assert output[2] == [["e"], ["f"]]


def test_batch_data_with_remainder() -> None:
    """
    Test BatchData creates batches with remainder
    """
    # Arrange
    df = DataFromList([["a"], ["b"], ["c"], ["d"], ["e"]], shuffle=False)
    df_batch = BatchData(df, batch_size=2, remainder=True)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df_batch)

    # Assert
    assert len(output) == 3
    assert len(df_batch) == 3
    assert output[0] == [["a"], ["b"]]
    assert output[1] == [["c"], ["d"]]
    assert output[2] == [["e"]]  # Remainder batch


def test_batch_data_dict_elements() -> None:
    """
    Test BatchData works with dict elements
    """
    # Arrange
    df = DataFromList([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}], shuffle=False)
    df_batch = BatchData(df, batch_size=2, remainder=False)

    # Act
    output = stu.collect_datapoint_from_dataflow(df=df_batch)

    # Assert
    assert len(output) == 2
    assert output[0] == [{"id": 1}, {"id": 2}]
    assert output[1] == [{"id": 3}, {"id": 4}]
