# -*- coding: utf-8 -*-
# File: test_base.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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

import json
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

import shared_test_utils as stu
from dd_core.dataflow.custom_serialize import SerializerPdfDoc
from dd_core.utils.object_types import get_type
from dd_core.utils.transform import BaseTransform
from dd_core.utils.types import PixelValues
from deepdoctection.extern.base import (
    DetectionResult,
    DeterministicImageTransformer,
    ModelCategories,
    NerModelCategories,
)


@pytest.fixture
def model_categories():
    init_categories = {1: "word", 2: "line", 3: "table", 4: "figure", 5: "header", 6: "footnote"}
    return ModelCategories(init_categories=init_categories)


@pytest.fixture
def ner_semantics():
    return ("question", "answer")


@pytest.fixture
def ner_bio():
    return ("B", "I")


@pytest.fixture
def ner_model_categories(ner_semantics, ner_bio):
    return NerModelCategories(
        init_categories=None,
        categories_semantics=ner_semantics,
        categories_bio=ner_bio,
    )


@pytest.fixture
def mock_base_transform():
    mock = MagicMock(spec=BaseTransform)
    mock.get_init_args.return_value = ["angle"]
    mock.get_category_names.return_value = (get_type("text"),)
    mock.angle = 90
    mock.apply_image.return_value = np.ones((10, 10, 3))
    mock.apply_coords.return_value = np.array([[10, 10, 20, 20], [30, 30, 40, 40]])
    mock.inverse_apply_coords.return_value = np.array([[5, 5, 15, 15], [25, 25, 35, 35]])
    return mock


@pytest.fixture
def transformer(mock_base_transform):
    return DeterministicImageTransformer(mock_base_transform)


@pytest.fixture
def detection_results():
    dr1 = DetectionResult(
        box=[1, 1, 2, 2],
        class_id=1,
        class_name=get_type("report_date"),
        score=0.9,
        absolute_coords=True,
        uuid=str(uuid.uuid4()),
    )
    dr2 = DetectionResult(
        box=[3, 3, 4, 4],
        class_id=2,
        class_name=get_type("umbrella"),
        score=0.8,
        absolute_coords=True,
        uuid=str(uuid.uuid4()),
    )
    return [dr1, dr2]


@pytest.fixture
def sample_np_img() -> PixelValues:
    # Small dummy image
    return np.zeros((10, 20, 3), dtype=np.uint8)


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    # Load first page bytes of the two-page test PDF
    df = SerializerPdfDoc.load(stu.asset_path("pdf_file_two_pages"))
    df.reset_state()
    dp = next(iter(df))
    return dp["pdf_bytes"]


@pytest.fixture
def textract_json() -> dict:
    with open(stu.asset_path("textract_sample"), "r") as f:
        return json.load(f)
