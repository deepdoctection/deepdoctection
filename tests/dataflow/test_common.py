# -*- coding: utf-8 -*-
# File: test_common.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
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

from typing import List, no_type_check

import numpy as np
from numpy.testing import assert_array_equal
from pytest import mark

from deepdoctection.dataflow import (
    ConcatData,
    CustomDataFromList,
    DataFlow,
    DataFromList,
    FakeData,
    FlattenData,
    JoinData,
    MapData,
    MapDataComponent,
    RepeatedData,
)

from ..test_utils import collect_datapoint_from_dataflow


@mark.basic
def test_flatten_data(dataset_three_dim: List[List[float]], dataset_flatten: List[List[List[float]]]) -> None:
    """
    Test the flattening of a dataflow
    """

    # Arrange
    df: DataFlow
    df = CustomDataFromList(dataset_three_dim)

    # Act
    df = FlattenData(df)
    output = collect_datapoint_from_dataflow(df=df)

    # Assert
    assert output == dataset_flatten


@mark.basic
def test_map_data() -> None:
    """Test MapData"""

    # Arrange
    @no_type_check
    def map_to_one(dp):
        return np.ones(dp[0].shape)

    df: DataFlow
    df = FakeData(shapes=[[4, 7, 3]], domain=(0, 1), size=1)

    # Act
    df = MapData(df, map_to_one)
    output = collect_datapoint_from_dataflow(df=df)

    # Assert
    assert_array_equal(output[0], np.ones((4, 7, 3)))


@mark.basic
def test_map_data_component() -> None:
    """Test MapDataComponent"""

    # Arrange
    dataflow_list = [{"foo": "1", "bak": "a"}, {"foo": "3", "bak": "c"}]
    df: DataFlow
    df = DataFromList(dataflow_list, shuffle=False)

    # Act
    df = MapDataComponent(df, lambda dp: "4", "foo")
    output = collect_datapoint_from_dataflow(df=df)

    # Assert
    assert output[0]["foo"] == "4"
    assert output[0]["bak"] == "a"
    assert output[1]["foo"] == "4"
    assert output[1]["bak"] == "c"


@mark.basic
def test_repeated_data() -> None:
    """Test RepeatedData"""

    # Arrange
    dataflow_list = [{"foo": "1", "bak": "a"}, {"foo": "3", "bak": "c"}]
    df: DataFlow
    df = DataFromList(dataflow_list, shuffle=False)

    # Act
    df = RepeatedData(df, num=2)
    output = collect_datapoint_from_dataflow(df=df)

    # Arrange
    assert len(output) == 4


@mark.basic
def test_repeated_data_yields_infinitely_many_datapoints() -> None:
    """Test RepeatedData produces infinitely many datapoints"""
    # Arrange
    dataflow_list = [{"foo": "1", "bak": "a"}, {"foo": "3", "bak": "c"}]
    df: DataFlow
    df = DataFromList(dataflow_list, shuffle=False)

    # Act
    df = RepeatedData(df, -1)
    output = collect_datapoint_from_dataflow(df=df, max_datapoints=100)

    # Assert
    assert len(output) == 100


@mark.basic
def test_concat_data() -> None:
    """Test ConcatData"""

    # Arrange
    dataflow_list_1 = [{"foo": "1", "bak": "a"}, {"foo": "3", "bak": "c"}]
    dataflow_list_2 = [{"foo": "2", "bak": "a"}, {"foo": "4", "bak": "c"}]
    df_1 = DataFromList(dataflow_list_1, shuffle=False)
    df_2 = DataFromList(dataflow_list_2, shuffle=False)

    # Act
    df: DataFlow
    df = ConcatData([df_1, df_2])
    output = collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 4


@mark.basic
def test_join_data() -> None:
    """Test JoinData"""

    # Arrange
    dataflow_list_1 = [{"foo": "1", "bak": "a"}, {"foo": "3", "bak": "c"}]
    dataflow_list_2 = [{"baz": "2", "bal": "a"}, {"baz": "4", "bal": "c"}]
    df_1 = DataFromList(dataflow_list_1, shuffle=False)
    df_2 = DataFromList(dataflow_list_2, shuffle=False)

    # Act
    df: DataFlow
    df = JoinData([df_1, df_2])
    output = collect_datapoint_from_dataflow(df=df)

    # Assert
    assert len(output) == 2
    assert "foo" in output[0] and "bak" in output[0] and "baz" in output[0] and "bal" in output[0]
