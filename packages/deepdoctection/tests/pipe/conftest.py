# -*- coding: utf-8 -*-
# File: xxx.py

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

import os
import numpy as np

import pytest

from dd_core.datapoint.image import Image
import shared_test_utils as stu

@pytest.fixture(name="dp_image")
def fixture_dp_image() -> Image:
    """fixture Image datapoint"""
    img = Image(location="/test/to/path", file_name="test_name")
    img.image = np.ones([400, 600, 3], dtype=np.float32)
    return img


@pytest.fixture
def pdf_path():
    # `stu.asset_path('pdf_file_two_pages')` points to a pdf file
    return stu.asset_path("pdf_file_two_pages")


@pytest.fixture
def image_dir_and_file():
    # `stu.asset_path('sample_image')` points to a png file; we need its directory for dir test
    img_path = stu.asset_path("sample_image")
    img_dir = os.path.dirname(img_path)
    return img_dir, img_path


@pytest.fixture
def image_bytes(image_dir_and_file):
    # Build bytes for the single-image test
    _, img_path = image_dir_and_file
    with open(img_path, "rb") as f:
        return f.read()


@pytest.fixture
def identity_pipe():
    # No pipeline components
    return DoctectionPipe(pipeline_component_list=[])