# -*- coding: utf-8 -*-
# File: test_image_pixel_operations.py

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
Testing Image pixel data operations (setting/getting images)
"""

import numpy as np
from numpy.testing import assert_array_equal
from pytest import raises

from dd_core.datapoint import Image, convert_np_array_to_b64
from dd_core.utils.error import ImageError

from ..conftest import WhiteImage

class TestImagePixelOperations:
    """Test Image pixel data handling"""

    def test_image_accepts_numpy_array(self, white_image: WhiteImage):
        """Image accepts and stores numpy array"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.image = white_image.image

        assert img.image is not None
        assert isinstance(img.image, np.ndarray)
        assert_array_equal(img.image, white_image.image)

    def test_image_converts_to_uint8(self):
        """Image converts numpy array to uint8"""
        img = Image(file_name="test.png")
        float_array = np.ones([10, 10, 3], dtype=np.float32) * 255
        img.image = float_array

        assert img.image.dtype == np.uint8
        assert_array_equal(img.image, np.ones([10, 10, 3], dtype=np.uint8) * 255)

    def test_image_accepts_b64_string(self, white_image: WhiteImage):
        """Image accepts and converts base64 string"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        b64_str = convert_np_array_to_b64(white_image.image)
        img.image = b64_str

        assert img.image is not None
        assert_array_equal(img.image, white_image.image)

    def test_image_get_image_to_np_array(self, white_image: WhiteImage):
        """get_image().to_np_array() returns numpy array"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image

        result = img.get_image().to_np_array()
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, white_image.image)

    def test_image_get_image_to_b64(self, white_image: WhiteImage):
        """get_image().to_b64() returns base64 string"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image

        result = img.get_image().to_b64()
        assert isinstance(result, str)
        assert result == convert_np_array_to_b64(white_image.image)

    def test_image_get_image_to_b64_none_when_no_image(self):
        """get_image().to_b64() returns None when image is None"""
        img = Image(file_name="test.png")

        result = img.get_image().to_b64()
        assert result is None

    def test_image_rejects_invalid_type(self):
        """Image setter rejects unsupported types"""
        img = Image(file_name="test.png")

        with raises(ImageError, match="Cannot load image. Unsupported type"):
            img.image = [1, 2, 3]  # type: ignore

    def test_image_clear_image_preserves_bbox_by_default(self, white_image: WhiteImage):
        """clear_image() preserves bbox by default"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image
        original_bbox = img._bbox

        img.clear_image(clear_bbox=False)
        assert img._bbox == original_bbox

    def test_image_clear_image_can_clear_bbox(self, white_image: WhiteImage):
        """clear_image(clear_bbox=True) clears bbox"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image

        img.clear_image(clear_bbox=True)
        assert img._bbox is None

    def test_image_clear_bbox_removes_self_embedding(self, white_image: WhiteImage):
        """clear_image(clear_bbox=True) removes self embedding"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image
        image_id = img.image_id
        assert image_id in img.embeddings

        img.clear_image(clear_bbox=True)
        assert image_id not in img.embeddings

    def test_setting_image_creates_self_embedding(self, white_image: WhiteImage):
        """Setting image creates self embedding"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image

        assert img.image_id in img.embeddings
        bbox = img.embeddings[img.image_id]
        assert bbox.width == white_image.image.shape[1]
        assert bbox.height == white_image.image.shape[0]

