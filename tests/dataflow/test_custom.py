# -*- coding: utf-8 -*-
# File: test_custom.py

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
Testing the module dataflow.custom
"""
from typing import Any, List

from deepdoctection.dataflow import CacheData, CustomDataFromList


def test_dataflow_cached_in_list(datapoint_list: List[Any]) -> None:
    """
    Testing CacheData get_cache method.
    """
    # Arrange
    df = CustomDataFromList(datapoint_list)

    # Act
    df = CacheData(df)
    df.reset_state()
    df_list = df.get_cache()

    # Assert
    assert set(df_list) == set(datapoint_list)


def test_dataflow_from_list_with_max_datapoint(datapoint_list: List[Any]) -> None:
    """
    Testing CustomDataFromList max_datapoint argument
    """
    # Act
    df = CustomDataFromList(datapoint_list, max_datapoints=3)
    df = CacheData(df)
    df.reset_state()
    df_list = df.get_cache()

    # Assert
    assert len(df) == 3
    assert len(df_list) == 3
