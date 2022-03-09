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
from typing import Any, Iterable, List, Optional, Union

from dataflow import DataFlow  # type: ignore

from deepdoctection.datapoint import Annotation
from deepdoctection.utils import sub_path
from deepdoctection.utils.systools import get_package_path


def get_test_path() -> str:
    """
    get path to test objects
    """
    return sub_path(os.path.split(__file__)[0], "tests/test_objects")


def get_integration_test_path() -> str:
    """
    fixture integration test path
    """
    return os.path.join(get_package_path(), "notebooks/pics/samples/sample_2")


def collect_datapoint_from_dataflow(df: DataFlow) -> List[Any]:
    """
    Calls the reset_state method of a dataflow and collects all datapoints to an output list
    :param df: A Dataflow
    :return: A list of datapoints of df
    """
    output: List[Any] = []
    df.reset_state()
    for dp in df:
        output.append(dp)

    return output


def anns_to_ids(annotations: Union[Iterable[Annotation], List[Annotation]]) -> List[Optional[str]]:
    """
    For a list of annotations return the list of annotation ids.
    :param annotations: A list of Annotations
    :return: A list of corresponding annotation ids
    """

    return [ann.annotation_id for ann in annotations]
