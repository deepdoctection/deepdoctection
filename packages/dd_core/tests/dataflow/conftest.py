# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixtures for dataflow package testing
"""

from pathlib import Path
from typing import Any

from pytest import fixture

import shared_test_utils as stu

from dd_core.dataflow import DataFromList

@fixture(name="simple_list")
def ficture_simple_list() -> list[list[str]]:
    """
    Simple list fixture
    """
    return [["a", "b"], ["c", "d"], ["e", "f"]]


@fixture(name="simple_list_dataflow")
def fixture_simple_list_dataflow(simple_list) -> DataFromList:
    """
    Simple dataflow with list elements
    """
    return DataFromList(simple_list, shuffle=False)

@fixture(name="simple_dict_list")
def ficture_simple_dict() -> list[dict[str, Any]]:
    """
    Simple list fixture
    """
    return [{"key1": "a", "key2": 1}, {"key1": "b", "key2": 2}, {"key1": "c", "key2": 3}]

@fixture(name="simple_dict_dataflow")
def fixture_simple_dict_dataflow(simple_dict_list) -> DataFromList:
    """
    Simple dataflow with dict elements
    """
    return DataFromList(simple_dict_list, shuffle=False)


@fixture(name="nested_list_dataflow")
def fixture_nested_list_dataflow() -> DataFromList:
    """
    Dataflow with nested lists for FlattenData testing
    """
    data = [["item1", "item2"], ["item3", "item4"]]
    return DataFromList(data, shuffle=False)

@fixture(name="coco_file_path")
def fixture_coco_file_path() -> Path:
    """Provide path to a sample page json file."""
    return stu.asset_path("coco_like")


@fixture(name="text_file")
def fixture_test_text_file_path() -> Path:
    """Provide path to a sample page json file."""
    return stu.asset_path("text_file")


@fixture(name="numerical_dataset")
def fixture_numerical_dataset() -> list[list[list[float]]]:
    """
    Numerical dataset of shape (2,2,3) for stats testing
    """
    return [[[1.0, 0.0, 2.0], [0.5, 1.0, 1.0]], [[2.0, 0.0, 4.0], [0.0, 1.0, 1.0]]]


@fixture(name="numerical_dataflow")
def fixture_numerical_dataflow(numerical_dataset) -> DataFromList:
    """
    Numerical dataflow for stats testing
    """
    return DataFromList(numerical_dataset, shuffle=False)


@fixture(name="expected_mean_axis_zero")
def fixture_expected_mean_axis_zero() -> list[list[float]]:
    """
    Expected mean along axis 0 for numerical_dataset
    """
    return [[1.5, 0.0, 3.0], [0.25, 1.0, 1.0]]


@fixture(name="expected_mean_all_axes")
def fixture_expected_mean_all_axes() -> float:
    """
    Expected mean along all axes for numerical_dataset
    """
    return 1.125


@fixture(name="expected_std_axis_zero")
def fixture_expected_std_axis_zero() -> list[list[float]]:
    """
    Expected std along axis 0 for numerical_dataset
    """
    return [[0.70710678, 0.0, 1.41421356], [0.35355339, 0.0, 0.0]]


@fixture(name="expected_std_all_axes")
def fixture_expected_std_all_axes() -> float:
    """
    Expected std along all axes for numerical_dataset
    """
    return 0.29462782549439476

