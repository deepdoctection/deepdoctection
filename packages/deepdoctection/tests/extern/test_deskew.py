# -*- coding: utf-8 -*-
# File: test_deskew.py

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

"""
Unit tests for deskew functionality using OpenCV and Pillow backends.

This module contains test cases for verifying the functionality of the deskewing
operation provided by the `Jdeskewer` class. It validates both the prediction
and transformation steps against expected outputs using images processed
with OpenCV and Pillow backends. The tests are skipped if the `jdeskew` library
is not available.
"""

import pytest
from numpy.testing import assert_array_equal

import shared_test_utils as stu
from dd_core.utils.env_info import SETTINGS
from dd_core.utils.file_utils import jdeskew_available
from dd_core.utils.fs import load_image_from_file
from dd_core.utils.viz import viz_handler
from deepdoctection.extern.deskew import Jdeskewer


@pytest.mark.skipif(not jdeskew_available(), reason="Requires jdeskew to be installed")
def test_jdeskew_predict_and_transform_opencv() -> None:
    """test jdeskew predict and transform with OpenCV backend"""
    # Use OpenCV backend
    SETTINGS.USE_DD_OPENCV = True
    SETTINGS.USE_DD_PILLOW = False
    SETTINGS.export_to_environ()
    viz_handler.refresh()

    img_in = load_image_from_file(stu.asset_path("skewed_input"))
    img_gt = load_image_from_file(stu.asset_path("deskewed_gt_opencv"))

    skewer = Jdeskewer()
    det = skewer.predict(img_in)  # type: ignore
    assert det.angle == 4.5326

    img_out = skewer.transform_image(img_in, det)  # type: ignore
    assert_array_equal(img_gt, img_out)


@pytest.mark.skipif(not jdeskew_available(), reason="Requires jdeskew to be installed")
def test_jdeskew_predict_and_transform_pillow() -> None:
    """test jdeskew predict and transform with Pillow backend"""
    SETTINGS.USE_DD_OPENCV = False
    SETTINGS.USE_DD_PILLOW = True
    SETTINGS.export_to_environ()
    viz_handler.refresh()

    img_in = load_image_from_file(stu.asset_path("skewed_input"))
    img_gt = load_image_from_file(stu.asset_path("deskewed_gt_pil"))

    skewer = Jdeskewer()
    det = skewer.predict(img_in)  # type: ignore
    assert det.angle == 4.5326

    img_out = skewer.transform_image(img_in, det)  # type: ignore
    assert_array_equal(img_gt, img_out)
