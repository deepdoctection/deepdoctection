# -*- coding: utf-8 -*-
# File: conftest.py

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
Module for fixtures
"""

from typing import Dict
from pathlib import Path

import numpy as np

from pytest import fixture

from deepdoctection.utils.settings import LayoutType, ObjectTypes
from deepdoctection.utils.types import PixelValues

from tests.test_utils import get_test_path

@fixture(name="path_to_d2_frcnn_yaml")
def fixture_path_to_d2_frcnn_yaml() -> Path:
    """
    path to d2 frcnn yaml file
    """
    return get_test_path() / "configs/d2/CASCADE_RCNN_R_50_FPN_GN.yaml"


@fixture(name="categories")
def fixture_categories() -> Dict[int, ObjectTypes]:
    """
    Categories as Dict
    """
    return {1: LayoutType.TEXT, 2: LayoutType.TITLE, 3: LayoutType.TABLE,
            4: LayoutType.FIGURE, 5: LayoutType.LIST}


@fixture(name="np_image")
def fixture_np_image() -> PixelValues:
    """
    np_array image
    """
    return np.ones([4, 6, 3], dtype=np.float32)
