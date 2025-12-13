# -*- coding: utf-8 -*-
# File: test_image_properties.py

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
Property-based testing for Image class using hypothesis
"""

import numpy as np
from hypothesis import assume, given # type: ignore
from hypothesis import strategies as st

from dd_core.datapoint import BoundingBox, Image
from dd_core.utils import get_uuid


class TestImageProperties:
    """Property-based tests for Image class"""

    @staticmethod
    @given(
        file_name=st.text(min_size=1, max_size=100).filter(lambda x: len(x.strip()) > 0), location=st.text(max_size=200)
    )
    def test_image_id_deterministic(file_name: str, location: str) -> None:
        """image_id is deterministic for same file_name and location"""
        img1 = Image(file_name=file_name, location=location)
        img2 = Image(file_name=file_name, location=location)

        assert img1.image_id == img2.image_id

    @staticmethod
    @given(
        file_name=st.text(min_size=1, max_size=100).filter(lambda x: len(x.strip()) > 0),
        external_id=st.text(min_size=1, max_size=100),
    )
    def test_external_id_generates_consistent_image_id(file_name: str, external_id: str) -> None:
        """external_id generates consistent image_id"""
        img1 = Image(file_name=file_name, external_id=external_id)
        img2 = Image(file_name=file_name, external_id=external_id)

        assert img1.image_id == img2.image_id
        assert img1.image_id == get_uuid(external_id)

    @staticmethod
    @given(
        width=st.integers(min_value=1, max_value=1000),
        height=st.integers(min_value=1, max_value=1000),
        channels=st.just(3),
    )
    def test_image_dimensions_preserved(width: int, height: int, channels: int) -> None:
        """Image dimensions are correctly preserved"""
        img = Image(file_name="test.png")
        arr = np.ones([height, width, channels], dtype=np.uint8)

        img.image = arr

        assert img.width == width
        assert img.height == height
        assert img.image.shape == (height, width, channels)

    @staticmethod
    @given(
        width=st.floats(min_value=10, max_value=5000, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=10, max_value=5000, allow_nan=False, allow_infinity=False),
    )
    def test_set_width_height_accepts_various_values(width: float, height: float) -> None:
        """set_width_height accepts various numeric values"""
        img = Image(file_name="test.png")

        img.set_width_height(width, height)

        assert abs(img.width - width) < 1
        assert abs(img.width - width) < 1

    @staticmethod
    @given(page_number=st.integers(min_value=0, max_value=10000))
    def test_page_number_stored_correctly(page_number: int) -> None:
        """page_number is stored correctly"""
        img = Image(file_name="test.png", page_number=page_number)

        assert img.page_number == page_number

    @staticmethod
    @given(n_embeddings=st.integers(min_value=1, max_value=20))
    def test_multiple_embeddings_stored(n_embeddings: int) -> None:
        """Multiple embeddings can be stored"""
        img = Image(file_name="test.png")

        for i in range(n_embeddings):
            bbox = BoundingBox(ulx=i, uly=i, width=10 + i, height=10 + i, absolute_coords=True)
            img.set_embedding(f"parent_{i}", bbox)

        assert len(img.embeddings) == n_embeddings

    @staticmethod
    @given(
        ulx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        uly=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        width=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
    )
    def test_embedding_bbox_values_preserved(ulx: float, uly: float, width: float, height: float) -> None:
        """Embedding bounding box values are preserved"""
        img = Image(file_name="test.png")
        bbox = BoundingBox(ulx=ulx, uly=uly, width=width, height=height, absolute_coords=True)

        img.set_embedding("parent", bbox)
        retrieved = img.get_embedding("parent")

        assert abs(retrieved.ulx - ulx) < 1
        assert abs(retrieved.uly - uly) < 1
        assert abs(retrieved.width - width) < 1
        assert abs(retrieved.height - height) < 1
