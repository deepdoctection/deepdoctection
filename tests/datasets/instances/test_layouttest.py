# -*- coding: utf-8 -*-
# File: test_layouttest.py

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
Testing module datasets.instances.layouttest
"""


from deepdoctection.datasets import LayoutTest

from ...test_utils import collect_datapoint_from_dataflow, get_test_path


def test_dataset_layouttest_returns_image() -> None:
    """
    test dataset layouttest returns image
    """

    # Arrange
    layouttest = LayoutTest()
    layouttest.dataflow.get_workdir = get_test_path  # type: ignore
    layouttest.dataflow.splits = {"test": ""}
    layouttest.dataflow.annotation_files = {"test": "test_layout.jsonl"}
    df = layouttest.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 2


def test_dataset_layouttest_with_load_image_returns_image() -> None:
    """
    test dataset layouttest returns image
    """

    # Arrange
    layouttest = LayoutTest()
    layouttest.dataflow.get_workdir = get_test_path  # type: ignore
    layouttest.dataflow.splits = {"test": ""}
    layouttest.dataflow.annotation_files = {"test": "test_layout.jsonl"}
    df = layouttest.dataflow.build(load_image=True)

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 2
    assert df_list[0].image is not None
