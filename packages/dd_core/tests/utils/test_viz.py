# -*- coding: utf-8 -*-
# File: test_viz.py

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
Testing the module utils.viz
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from numpy import uint8
from numpy.typing import NDArray

from dd_core.utils.viz import viz_handler


class TestVizHandlerReadWrite:
    """Test VizPackageHandler read/write operations"""

    @staticmethod
    def test_read_write_image(np_image: NDArray[np.uint8]) -> None:
        """Test reading and writing images"""

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "test_image.png"
            viz_handler.write_image(tmp_path, np_image)
            assert tmp_path.exists()
            loaded_image = viz_handler.read_image(tmp_path)
            assert loaded_image.shape == np_image.shape


class TestVizHandlerEncode:
    """Test VizPackageHandler encode operations"""

    @staticmethod
    @pytest.mark.parametrize("image_dtype", [uint8])
    def test_encode(np_image: NDArray[np.uint8], image_dtype: type) -> None:
        """Test encoding image to bytes"""

        result = viz_handler.encode(np_image)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestVizHandlerConvertNpToB64:
    """Test VizPackageHandler convert_np_to_b64"""

    @staticmethod
    def test_convert_np_to_b64(np_image: NDArray[np.uint8]) -> None:
        """Test converting numpy array to base64 string"""

        result = viz_handler.convert_np_to_b64(np_image)
        assert isinstance(result, str)
        assert len(result) > 0


class TestVizHandlerConvertB64ToNp:
    """Test VizPackageHandler convert_b64_to_np"""

    @staticmethod
    def test_convert_b64_to_np(np_image: NDArray[np.uint8]) -> None:
        """Test converting base64 string to numpy array"""

        b64_str = viz_handler.convert_np_to_b64(np_image)
        result = viz_handler.convert_b64_to_np(b64_str)
        assert isinstance(result, np.ndarray)
        assert result.shape == np_image.shape
        assert np.array_equal(result, np_image)


class TestVizHandlerConvertBytesToNp:
    """Test VizPackageHandler convert_bytes_to_np"""

    @staticmethod
    def test_convert_bytes_to_np(np_image: NDArray[np.uint8]) -> None:
        """Test converting bytes to numpy array"""

        image_bytes = viz_handler.encode(np_image)
        result = viz_handler.convert_bytes_to_np(image_bytes)
        assert isinstance(result, np.ndarray)
        assert result.shape == np_image.shape
        assert np.array_equal(result, np_image)


class TestVizHandlerResize:
    """Test VizPackageHandler resize"""

    @staticmethod
    @pytest.mark.parametrize("width,height,interpolation", [(75, 50, "VIZ"), (200, 300, "VIZ")])
    def test_resize(np_image: NDArray[np.uint8], width: int, height: int, interpolation: str) -> None:
        """Test resizing image"""

        result = viz_handler.resize(np_image, width, height, interpolation)
        assert result.shape[:2] == (height, width)


class TestVizHandlerGetTextSize:
    """Test VizPackageHandler get_text_size"""

    @staticmethod
    @pytest.mark.parametrize("text,font_scale", [("test", 1.0), ("longer text", 2.0)])
    def test_get_text_size(text: str, font_scale: float) -> None:
        """Test getting text size"""

        width, height = viz_handler.get_text_size(text, font_scale)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert width > 0
        assert height > 0


class TestVizHandlerRotateImage:
    """Test VizPackageHandler rotate_image"""

    @staticmethod
    @pytest.mark.parametrize("angle", [90, 180])
    def test_rotate_image(np_image: NDArray[np.uint8], angle: float) -> None:
        """Test rotating image"""
        result = viz_handler.rotate_image(np_image, angle)
        assert isinstance(result, np.ndarray)
