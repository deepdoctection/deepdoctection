# -*- coding: utf-8 -*-
# File: test_iiitar13k.py

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
Testing module datasets.instances.iiitar13k
"""

from deepdoctection.datasets import IIITar13K

from ...test_utils import collect_datapoint_from_dataflow, get_test_path


def test_dataset_iiitar13k_returns_image() -> None:
    """
    test dataset iiitar13k returns image
    """

    # Arrange
    iiitar13k = IIITar13K()
    iiitar13k.dataflow.get_workdir = get_test_path  # type: ignore
    iiitar13k.dataflow.annotation_files = {"val": ""}
    df = iiitar13k.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 1
