# -*- coding: utf-8 -*-
# File: test_stats.py

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
Testing the module dataflow.stats
"""

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from dd_core.dataflow import DataFromList, MeanFromDataFlow, StdFromDataFlow


def test_mean_from_dataflow_along_one_dimension(
    numerical_dataflow: DataFromList, expected_mean_axis_zero: list[list[float]]
) -> None:
    """
    Test MeanFromDataFlow calculates mean along the 0th dimension correctly
    """
    # Arrange
    df = MeanFromDataFlow(numerical_dataflow, axis=0)

    # Act
    mean = df.start()

    # Assert
    assert_array_equal(mean, np.array(expected_mean_axis_zero))


def test_mean_from_dataflow_along_all_dimensions(
    numerical_dataflow: DataFromList, expected_mean_all_axes: float
) -> None:
    """
    Test MeanFromDataFlow calculates mean along all dimensions correctly
    """
    # Arrange
    df = MeanFromDataFlow(numerical_dataflow)

    # Act
    mean = df.start()

    # Assert
    assert_array_equal(mean, np.array(expected_mean_all_axes))


def test_std_from_dataflow_along_one_dimension(
    numerical_dataflow: DataFromList, expected_std_axis_zero: list[list[float]]
) -> None:
    """
    Test StdFromDataFlow calculates standard deviation along the 0th dimension correctly
    """
    # Arrange
    df = StdFromDataFlow(numerical_dataflow, axis=0)

    # Act
    std = df.start()

    # Assert
    assert_array_almost_equal(std, np.array(expected_std_axis_zero), decimal=4)


def test_std_from_dataflow_along_all_dimensions(numerical_dataflow: DataFromList, expected_std_all_axes: float) -> None:
    """
    Test StdFromDataFlow calculates standard deviation along all dimensions correctly
    """
    # Arrange
    df = StdFromDataFlow(numerical_dataflow)

    # Act
    std = df.start()

    # Assert
    assert_array_almost_equal(std, np.array(expected_std_all_axes), decimal=4)
