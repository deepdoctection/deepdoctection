# -*- coding: utf-8 -*-
# File: test_deskew.py

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
Testing module extern.deskew
"""

from numpy.testing import assert_array_equal
from pytest import mark

from deepdoctection.extern.deskew import Jdeskewer
from deepdoctection.utils.fs import load_image_from_file
from tests.test_utils import get_test_path


class TestJdeskewer:
    """
    Test Jdeskewer
    """

    @staticmethod
    @mark.requires_tf_or_pt
    def test_deskewer_transforms_image() -> None:
        """
        Detector deskews image and rotates it accordingly
        """

        # Arrange
        test_path_input_image = get_test_path() / "skewed_input.png"
        test_path_gt_image = get_test_path() / "skewed_gt.png"

        image = load_image_from_file(test_path_input_image)
        image_gt = load_image_from_file(test_path_gt_image)

        deskewer = Jdeskewer()

        # Act
        assert image is not None
        output_image = deskewer.transform(image)

        # Assert
        assert image_gt is not None
        assert_array_equal(image_gt, output_image)
