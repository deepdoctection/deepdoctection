# -*- coding: utf-8 -*-
# File: xxx.py

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
Module for methods that might be helpful for testing
"""

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union
from unittest.mock import MagicMock

from deepdoctection.dataflow import DataFlow
from deepdoctection.datapoint import Annotation


def get_test_path() -> Path:
    """
    get path to test objects
    """
    return Path(os.path.split(__file__)[0]) / "test_objects"


def get_integration_test_path() -> Path:
    """
    fixture integration test path
    """
    return get_test_path() / "sample_2"


def collect_datapoint_from_dataflow(
    df: Union[DataFlow, Iterator[Any]], max_datapoints: Optional[int] = None
) -> List[Any]:
    """
    Calls the reset_state method of a dataflow and collects all datapoints to an output list
    :param df: A Dataflow
    :param max_datapoints: The maximum number of datapoints to yield from
    :return: A list of datapoints of df
    """

    output: List[Any] = []
    if hasattr(df, "reset_state"):
        df.reset_state()

    for idx, dp in enumerate(df):
        if max_datapoints is not None:
            if idx >= max_datapoints:
                break

        output.append(dp)

    return output


def anns_to_ids(annotations: Union[Iterable[Annotation], List[Annotation]]) -> List[Optional[str]]:
    """
    For a list of annotations return the list of annotation ids.
    :param annotations: A list of Annotations
    :return: A list of corresponding annotation ids
    """

    return [ann.annotation_id for ann in annotations]


def set_num_gpu_to_one() -> int:
    """
    set gpu number to one
    """
    return 1


def get_mock_patch(name: str) -> MagicMock:
    """Generating a mock object with a specific name"""
    mock = MagicMock()
    mock.__class__.__name__ = name
    return mock
