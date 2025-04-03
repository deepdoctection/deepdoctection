# -*- coding: utf-8 -*-
# File: test_funsd.py

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
Testing module datasets.instances.funsd
"""

from unittest.mock import MagicMock, patch

from pytest import mark

from deepdoctection.datasets import Funsd
from deepdoctection.utils.settings import LayoutType, WordType

from ...test_utils import collect_datapoint_from_dataflow, get_test_path
from .conftest import get_white_image


@mark.basic
@patch("deepdoctection.mapper.xfundstruct.load_image_from_file", MagicMock(side_effect=get_white_image))
def test_dataset_funsd_returns_image_and_annotations() -> None:
    """
    test dataset funsd returns image and annotations
    """

    # Arrange
    funsd = Funsd()
    funsd.dataflow.get_workdir = get_test_path  # type: ignore
    funsd.dataflow.splits = {"test": ""}
    funsd.dataflow.annotation_files = {"test": ""}
    df = funsd.dataflow.build()

    # Act
    df_list = collect_datapoint_from_dataflow(df)
    assert len(df_list) == 6  # the first four images coming from files not related to funsd data
    dp = [dp for dp in df_list if dp.file_name == "test_file_funsd.png"][0]
    word = dp.get_annotation(category_names=LayoutType.WORD)[0]
    assert word.get_sub_category(WordType.TOKEN_CLASS) is not None
    assert word.get_sub_category(WordType.CHARACTERS) is not None
    assert word.get_sub_category(WordType.TAG) is not None
    assert word.get_sub_category(WordType.TOKEN_TAG) is not None

    text = dp.get_annotation(category_names=LayoutType.TEXT)[0]
    assert text.get_sub_category(WordType.TOKEN_CLASS) is not None
