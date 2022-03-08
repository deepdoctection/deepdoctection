# -*- coding: utf-8 -*-
# File: test_misc.py

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
Testing the module mapper.misc
"""
from typing import Dict, Union
from unittest.mock import MagicMock, patch

from numpy import float32, ones
from numpy.testing import assert_array_equal
from pytest import mark

from deepdoctection.datapoint import Image
from deepdoctection.mapper.misc import to_image

_TEST_IMAGE = Image(file_name="test_image.png", location="test/to/path/test_image.png")
_TEST_IMAGE.image = ones((4, 3, 3), dtype=float32)

_TEST_IMAGE_2 = Image(file_name="test_image.pdf", location="test/to/path/test_image.pdf")
_TEST_IMAGE_2.image = ones((4, 3, 3), dtype=float32)


@mark.parametrize(
    "datapoint,expected_image",
    [
        ("test/to/path/test_image.png", _TEST_IMAGE),
        ({"file_name": "test_image.png", "location": "test/to/path/test_image.png"}, _TEST_IMAGE),
        ({"file_name": "test_image.png", "path": "test/to/path"}, _TEST_IMAGE),
        ({"file_name": "test_image.pdf", "path": "test/to/path", "pdf_bytes": b"some_bytes"}, _TEST_IMAGE_2),
    ],
)
@patch("deepdoctection.mapper.misc.load_image_from_file", MagicMock(return_value=ones((4, 3, 3))))
@patch("deepdoctection.mapper.misc.convert_pdf_bytes_to_np_array_v2", MagicMock(return_value=ones((4, 3, 3))))
def test_to_image(datapoint: Union[str, Dict[str, Union[str, bytes]]], expected_image: Image) -> None:
    """
    Image is properly constructed
    :param datapoint: input to testing method
    :param expected_image: expected Image output
    """

    # Act

    dp = to_image(datapoint)

    # Assert
    assert dp is not None
    assert dp.image_id == expected_image.image_id
    assert dp.file_name == expected_image.file_name
    assert dp.location == expected_image.location
    assert_array_equal(dp.get_image("np"), expected_image.image)  # type: ignore

    if dp.file_name.endswith("pdf"):
        assert dp.pdf_bytes == datapoint["pdf_bytes"]  # type: ignore
