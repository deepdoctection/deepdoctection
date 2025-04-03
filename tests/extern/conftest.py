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
from typing import Mapping, Sequence

from pytest import fixture

from deepdoctection.extern.base import DetectionResult
from deepdoctection.utils.settings import ObjectTypes
from deepdoctection.utils.types import JsonDict

from ..data import get_textract_response
from ..mapper.data import DatapointXfund
from .data import ANGLE_RESULT, PDF_BYTES, PDF_BYTES_2, get_detr_categories


@fixture(name="layoutlm_input_for_predictor")
def fixture_layoutlm_input_for_predictor() -> JsonDict:
    """
    Layoutlm input
    """
    return DatapointXfund().get_layout_input()


@fixture(name="layoutlm_v2_input")
def fixture_layoutlm_input() -> JsonDict:
    """
    Layoutlm_v2 input
    """
    return DatapointXfund().get_layout_v2_input()


@fixture(name="categories_semantics")
def fixture_categories_semantics() -> Sequence[str]:
    """
    Categories semantics
    """
    return DatapointXfund().get_categories_semantics()


@fixture(name="categories_bio")
def fixture_categories_bio() -> Sequence[str]:
    """
    Categories semantics
    """
    return DatapointXfund().get_categories_bio()


@fixture(name="token_class_names")
def fixture_token_class_names() -> Sequence[str]:
    """
    Categories semantics
    """
    return DatapointXfund().get_token_class_names()


@fixture(name="textract_response")
def fixture_textract_response() -> JsonDict:
    """fixture textract_response"""
    return get_textract_response()


@fixture(name="pdf_bytes")
def fixture_pdf_bytes() -> bytes:
    """
    fixture pdf bytes
    """
    return PDF_BYTES


@fixture(name="pdf_bytes_page_2")
def fixture_pdf_bytes_page_2() -> bytes:
    """
    fixture pdf bytes
    """
    return PDF_BYTES_2


@fixture(name="angle_detection_result")
def fixture_angle_detection_result() -> DetectionResult:
    """
    fixture detection result for running rotation image transformation
    """
    return ANGLE_RESULT


@fixture(name="detr_categories")
def fixture_detr_categories() -> Mapping[int, ObjectTypes]:
    """
    fixture object types
    """
    return get_detr_categories()
