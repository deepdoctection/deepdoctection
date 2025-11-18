# -*- coding: utf-8 -*-
# File: conftest.py

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
Fixtures for dataflow package testing
"""

from dataclasses import dataclass
from typing import List

from pytest import fixture


@fixture(name="datapoint_list")
def fixture_datapoint_list() -> List[str]:
    """
    List fixture
    """
    return ["a", "b", "c", "d"]


@dataclass
class DatasetThreeDim:
    """
    Dataset of shape (2,2,3)
    """

    dataset = [[[1.0, 0.0, 2.0], [0.5, 1.0, 1.0]], [[2.0, 0.0, 4.0], [0.0, 1.0, 1.0]]]
    mean_axis_zero = [[1.5, 0.0, 3.0], [0.25, 1.0, 1.0]]
    mean_all_axes = 1.125
    std_axis_zero = [[0.70710678, 0.0, 1.41421356], [0.35355339, 0.0, 0.0]]
    std_all_axes = 0.29462782549439476
    dataset_flatten = [[[1.0, 0.0, 2.0]], [[0.5, 1.0, 1.0]], [[2.0, 0.0, 4.0]], [[0.0, 1.0, 1.0]]]


@fixture(name="dataset_three_dim")
def fixture_dataset_three_dim() -> List[List[List[float]]]:
    """
    Dataset fixture
    """
    return DatasetThreeDim().dataset


@fixture(name="mean_axis_zero")
def fixture_mean_axis_zero() -> List[List[float]]:
    """
    Mean fixture
    """
    return DatasetThreeDim().mean_axis_zero


@fixture(name="mean_all_axes")
def fixture_mean_all_axes() -> float:
    """
    Mean fixture
    """
    return DatasetThreeDim().mean_all_axes


@fixture(name="std_axis_zero")
def fixture_std_axis_zero() -> List[List[float]]:
    """
    Std fixture
    """
    return DatasetThreeDim().std_axis_zero


@fixture(name="std_all_axes")
def fixture_std_all_axes() -> float:
    """
    Std fixture
    """
    return DatasetThreeDim().std_all_axes


@fixture(name="dataset_flatten")
def fixture_dataset_flatten() -> List[List[List[float]]]:
    """
    Dataset flatten
    """
    return DatasetThreeDim().dataset_flatten
