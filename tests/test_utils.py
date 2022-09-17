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

from deepdoctection.dataflow import DataFlow
from deepdoctection.datapoint import Annotation
from deepdoctection.utils.systools import get_package_path
from deepdoctection.utils.settings import object_types_registry, ObjectTypes


def get_test_path() -> Path:
    """
    get path to test objects
    """
    return Path(os.path.split(__file__)[0]) / "test_objects"


def get_integration_test_path() -> Path:
    """
    fixture integration test path
    """
    return get_package_path() / "notebooks" / "pics" / "samples" / "sample_2"


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
        df.reset_state()  # type: ignore

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


@object_types_registry.register("TestType")
class TestType(ObjectTypes):
    foo = "foo"
    FOO = "FOO"
    bak = "bak"
    BAK = "BAK"
    BAK_1 = "BAK_1"
    BAK_11 = "BAK_11"
    BAK_12 = "BAK_12"
    BAK_21 = "BAK_21"
    BAK_22 = "BAK_22"
    cat = "cat"
    FOO_1 = "FOO_1"
    FOO_2 = "FOO_2"
    FOO_3 = "FOO_3"
    FOOBAK = "FOOBAK"
    TEST_SUMMARY = "TEST_SUMMARY"
    baz = "baz"
    BAZ = "BAZ"
    b_foo = "B-FOO"
    i_foo = "I-FOO"
    o = "O"
    sub = "sub"
    sub_2 = "sub_2"
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"


