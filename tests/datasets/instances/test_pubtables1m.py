# -*- coding: utf-8 -*-
# File: test_pubtables1m.py

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
Testing module datasets.instances.pubtables1m
"""

from pathlib import Path

from pytest import mark

from deepdoctection.datasets import Pubtables1MDet, Pubtables1MStruct

from ...test_utils import collect_datapoint_from_dataflow, get_test_path


@mark.basic
def test_dataset_pubtables1m_det_returns_image() -> None:
    """
    test dataset pubtables1m_det return image
    """

    # Arrange
    pubtables = Pubtables1MDet()
    pubtables.dataflow.get_workdir = get_test_path  # type: ignore
    pubtables.dataflow.annotation_files = {"val": ""}
    df = pubtables.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 1


@mark.basic
def test_dataset_pubtables1m_struct_returns_image() -> None:
    """
    test dataset pubtables1m_struct return image
    """

    def get_pubtab1m_struct_test_path() -> Path:
        test_path = get_test_path() / "pubtable1m_struct"
        return test_path

    # Arrange
    pubtables = Pubtables1MStruct()
    pubtables.dataflow.get_workdir = get_pubtab1m_struct_test_path  # type: ignore
    pubtables.dataflow.annotation_files = {"val": ""}
    df = pubtables.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 1
