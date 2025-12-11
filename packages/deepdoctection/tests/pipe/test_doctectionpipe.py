# -*- coding: utf-8 -*-
# File: test_doctectionpipe.py

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


import pytest

import shared_test_utils as stu
from dd_core.dataflow import DataFromList
from dd_core.datapoint.image import Image
from dd_core.datapoint.view import Page
from deepdoctection.pipe.doctectionpipe import DoctectionPipe


@pytest.fixture(autouse=True)
def _set_env_dpi(monkeypatch):
    monkeypatch.setenv("DPI", "72")
    monkeypatch.delenv("IMAGE_WIDTH", raising=False)
    monkeypatch.delenv("IMAGE_HEIGHT", raising=False)


def test_analyze_path_pdf_pages(pdf_path):
    identity_pipe = DoctectionPipe(pipeline_component_list=[])
    df = identity_pipe.analyze(path=pdf_path, output="page")
    items = stu.collect_datapoint_from_dataflow(df)
    assert len(items) >= 1
    assert all(isinstance(p, Page) for p in items)


def test_analyze_path_dir_png_identity_image(image_dir_and_file):
    img_dir, _ = image_dir_and_file
    identity_pipe = DoctectionPipe(pipeline_component_list=[])
    df = identity_pipe.analyze(path=img_dir, file_type=".png", output="image")
    items = stu.collect_datapoint_from_dataflow(df)
    assert len(items) == 1
    assert all(isinstance(img, Image) for img in items)


def test_analyze_bytes_single_image(image_dir_and_file, image_bytes):
    identity_pipe = DoctectionPipe(pipeline_component_list=[])
    _, img_path = image_dir_and_file
    df = identity_pipe.analyze(path=img_path, bytes=image_bytes, file_type=".png", output="image")
    items = stu.collect_datapoint_from_dataflow(df)
    assert len(items) == 1
    assert isinstance(items[0], Image)


def test_analyze_dataset_dataflow_image(dp_image):
    identity_pipe = DoctectionPipe(pipeline_component_list=[])
    dp_list = [dp_image]
    dataset_df = DataFromList(lst=dp_list)
    df = identity_pipe.analyze(dataset_dataflow=dataset_df, output="image")
    items = stu.collect_datapoint_from_dataflow(df)
    assert len(items) == 1
    assert isinstance(items[0], Image)


def test_analyze_path_pdf_dict(pdf_path):
    identity_pipe = DoctectionPipe(pipeline_component_list=[])
    df = identity_pipe.analyze(path=pdf_path, output="dict")
    items = stu.collect_datapoint_from_dataflow(df)
    assert len(items) >= 1
    assert all(isinstance(d, dict) for d in items)
