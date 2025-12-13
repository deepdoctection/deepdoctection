# -*- coding: utf-8 -*-
# File: test_image_initialization.py

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
Testing Image initialization and basic properties
"""

from pytest import mark, raises

from dd_core.datapoint import Image
from dd_core.utils import get_uuid
from dd_core.utils.error import ImageError

from ..conftest import WhiteImage


class TestImageInitialization:
    """Test Image initialization and basic properties"""

    def test_image_can_be_created_with_minimal_params(self) -> None:
        """Image can be created with just file_name"""
        img = Image(file_name="test.png")

        assert img.file_name == "test.png"
        assert img.location == ""
        assert img.image_id is not None

    def test_image_with_location_and_filename(self) -> None:
        """Image stores location and filename correctly"""
        img = Image(file_name="test.png", location="/path/to/file")

        assert img.file_name == "test.png"
        assert img.location == "/path/to/file"
        assert img.image_id == get_uuid("/path/to/file", "test.png")

    def test_image_with_external_id(self) -> None:
        """Image with external_id generates correct image_id"""
        img = Image(file_name="test.png", external_id="custom_id_123")

        assert img.external_id == "custom_id_123"
        assert img.image_id == get_uuid("custom_id_123")

    def test_image_with_uuid_external_id(self) -> None:
        """Image with UUID-like external_id uses it directly as image_id"""
        uuid_str = "90c05f37-0000-0000-0000-b84f9d14ff44"
        img = Image(file_name="test.png", external_id=uuid_str)

        assert img.image_id == uuid_str

    def test_image_document_id_defaults_to_image_id(self) -> None:
        """When document_id is not provided, it defaults to image_id"""
        img = Image(file_name="test.png", location="/path")

        assert img.document_id == img.image_id

    def test_image_id_cannot_be_reassigned_once_set(self, white_image: WhiteImage) -> None:
        """image_id cannot be reassigned after initialization"""
        img = Image(file_name=white_image.file_name, location=white_image.location)

        with raises(ImageError, match="image_id cannot be reassigned"):
            img.image_id = "new-uuid-value"
