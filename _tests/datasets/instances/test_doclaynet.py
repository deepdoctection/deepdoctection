# -*- coding: utf-8 -*-
# File: test_doclaynet.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Testing module datasets.instances.doclaynet
"""

from pytest import mark

from deepdoctection.datasets import DocLayNet

from ...test_utils import collect_datapoint_from_dataflow, get_test_path


@mark.basic
def test_dataset_doclaynet_returns_image() -> None:
    """
    test dataset publaynet returns image
    """

    # Arrange
    doclaynet = DocLayNet()
    doclaynet.dataflow.get_workdir = get_test_path  # type: ignore
    doclaynet.dataflow.annotation_files = {"val": "test_file_doclay.json"}
    df = doclaynet.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 2
