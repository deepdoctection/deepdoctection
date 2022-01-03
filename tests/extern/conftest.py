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
Fixtures for extern package testing
"""
from typing import List

from pytest import fixture

from deep_doctection.utils.detection_types import JsonDict

from ..data import get_textract_response
from ..mapper.data import DatapointXfund


@fixture(name="layoutlm_input")
def fixture_layoutlm_input() -> JsonDict:
    """
    Layoutlm input
    """
    return DatapointXfund().get_layout_input()


@fixture(name="categories_semantics")
def fixture_categories_semantics() -> List[str]:
    """
    Categories semantics
    """
    return DatapointXfund().get_categories_semantics()


@fixture(name="categories_bio")
def fixture_categories_bio() -> List[str]:
    """
    Categories semantics
    """
    return DatapointXfund().get_categories_bio()


@fixture(name="token_class_names")
def fixture_token_class_names() -> List[str]:
    """
    Categories semantics
    """
    return DatapointXfund().get_token_class_names()


@fixture(name="textract_response")
def fixture_textract_response() -> JsonDict:
    """fixture textract_response"""
    return get_textract_response()
