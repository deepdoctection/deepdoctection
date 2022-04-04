# -*- coding: utf-8 -*-
# File: test_stats.py

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
Testing the module dataflow.stats
"""
from typing import List

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from deepdoctection.dataflow import CustomDataFromList, MeanFromDataFlow, StdFromDataFlow


class TestMeanFromDataFlow:  # pylint: disable=too-few-public-methods
    """
    Testing MeanFromDataFlow along various dimensions
    """

    @staticmethod
    def test_mean_along_one_dimension(dataset_three_dim: List[List[float]], mean_axis_zero: List[float]) -> None:
        """
        Test mean along the 0th dimension.
        """

        # Arrange
        df = CustomDataFromList(dataset_three_dim)

        # Act
        mean = MeanFromDataFlow(df, axis=0).start()

        # Assert
        assert_array_equal(mean, np.array(mean_axis_zero))

    @staticmethod
    def test_mean_along_all_dimension(dataset_three_dim: List[List[float]], mean_all_axes: List[float]) -> None:
        """
        Test mean along all dimensions.
        """

        # Arrange
        df = CustomDataFromList(dataset_three_dim)

        # Act
        mean = MeanFromDataFlow(df).start()

        # Assert
        assert_array_equal(mean, np.array(mean_all_axes))


class TestStdFromDataFlow:  # pylint: disable=too-few-public-methods
    """
    Testing StdFromDataFlow along various dimensions
    """

    @staticmethod
    def test_std_along_one_dimension(dataset_three_dim: List[List[float]], std_axis_zero: List[float]) -> None:
        """
        Test std along the 0th dimension.
        """

        # Arrange
        df = CustomDataFromList(dataset_three_dim)

        # Act
        std = StdFromDataFlow(df, axis=0).start()

        # Assert
        assert_array_almost_equal(std, np.array(std_axis_zero), decimal=4)

    @staticmethod
    def test_std_along_all_dimension(dataset_three_dim: List[List[float]], std_all_axes: List[float]) -> None:
        """
        Test std along all dimensions.
        """

        # Arrange
        df = CustomDataFromList(dataset_three_dim)

        # Act
        std = StdFromDataFlow(df).start()

        # Assert
        assert_array_almost_equal(std, np.array(std_all_axes), decimal=4)
