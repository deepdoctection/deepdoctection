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
from hypothesis import given, strategies as st, assume
from pytest import mark

from dd_datapoint.datapoint import BoundingBox, Image, ImageAnnotation
from dd_datapoint.utils import get_uuid


class TestImageProperties:
    """Property-based tests for Image class"""

    @staticmethod
    @mark.property
    @given(
        file_name=st.text(min_size=1, max_size=100).filter(lambda x: len(x.strip()) > 0),
        location=st.text(max_size=200)
    )
    def test_image_id_deterministic(file_name: str, location: str):
        """image_id is deterministic for same file_name and location"""
        img1 = Image(file_name=file_name, location=location)
        img2 = Image(file_name=file_name, location=location)

        assert img1.image_id == img2.image_id

    @staticmethod
    @mark.property
    @given(
        file_name=st.text(min_size=1, max_size=100).filter(lambda x: len(x.strip()) > 0),
        external_id=st.text(min_size=1, max_size=100)
    )
    def test_external_id_generates_consistent_image_id(file_name: str, external_id: str):
        """external_id generates consistent image_id"""
        img1 = Image(file_name=file_name, external_id=external_id)
        img2 = Image(file_name=file_name, external_id=external_id)

        assert img1.image_id == img2.image_id
        assert img1.image_id == get_uuid(external_id)

    @staticmethod
    @mark.property
    @given(
        width=st.integers(min_value=1, max_value=1000),
        height=st.integers(min_value=1, max_value=1000),
        channels=st.just(3)
    )
    def test_image_dimensions_preserved(width: int, height: int, channels: int):
        """Image dimensions are correctly preserved"""
        img = Image(file_name="test.png")
        arr = np.ones([height, width, channels], dtype=np.uint8)

        img.image = arr

        assert img.width == width
        assert img.height == height
        assert img.image.shape == (height, width, channels)

    @staticmethod
    @mark.property
    @given(
        width=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
    )
    def test_set_width_height_accepts_various_values(width: float, height: float):
        """set_width_height accepts various numeric values"""
        img = Image(file_name="test.png")

        img.set_width_height(width, height)

        assert img.width == width
        assert img.height == height

    @staticmethod
    @mark.property
    @given(
        page_number=st.integers(min_value=0, max_value=10000)
    )
    def test_page_number_stored_correctly(page_number: int):
        """page_number is stored correctly"""
        img = Image(file_name="test.png", page_number=page_number)

        assert img.page_number == page_number

    @staticmethod
    @mark.property
    @given(
        n_annotations=st.integers(min_value=0, max_value=20)
    )
    def test_multiple_annotations_unique_ids(n_annotations: int):
        """Each annotation gets unique ID"""
        img = Image(file_name="test.png")
        img.set_width_height(100, 100)

        annotation_ids = []
        for i in range(n_annotations):
            ann = ImageAnnotation(
                category_name=f"CAT_{i}",
                bounding_box=BoundingBox(ulx=i, uly=i, width=10, height=10, absolute_coords=True)
            )
            img.dump(ann)
            annotation_ids.append(ann.annotation_id)

        # All IDs should be unique
        assert len(set(annotation_ids)) == n_annotations

    @staticmethod
    @mark.property
    @given(
        n_embeddings=st.integers(min_value=1, max_value=20)
    )
    def test_multiple_embeddings_stored(n_embeddings: int):
        """Multiple embeddings can be stored"""
        img = Image(file_name="test.png")

        for i in range(n_embeddings):
            bbox = BoundingBox(ulx=i, uly=i, width=10+i, height=10+i, absolute_coords=True)
            img.set_embedding(f"parent_{i}", bbox)

        assert len(img.embeddings) == n_embeddings

    @staticmethod
    @mark.property
    @given(
        ulx=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        uly=st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        width=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False),
        height=st.floats(min_value=1, max_value=100, allow_nan=False, allow_infinity=False)
    )
    def test_embedding_bbox_values_preserved(ulx: float, uly: float, width: float, height: float):
        """Embedding bounding box values are preserved"""
        img = Image(file_name="test.png")
        bbox = BoundingBox(ulx=ulx, uly=uly, width=width, height=height, absolute_coords=True)

        img.set_embedding("parent", bbox)
        retrieved = img.get_embedding("parent")

        assert retrieved.ulx == ulx
        assert retrieved.uly == uly
        assert retrieved.width == width
        assert retrieved.height == height

    @staticmethod
    @mark.property
    @given(
        data=st.data(),
        n_changes=st.integers(min_value=1, max_value=10)
    )
    def test_state_id_changes_on_modifications(data, n_changes: int):
        """state_id changes when image is modified"""
        img = Image(file_name="test.png")
        state_ids = [img.state_id]

        for _ in range(n_changes):
            choice = data.draw(st.integers(min_value=0, max_value=2))
            if choice == 0:
                # Add annotation
                ann = ImageAnnotation(
                    category_name="TEST",
                    bounding_box=BoundingBox(
                        ulx=data.draw(st.floats(0, 100, allow_nan=False, allow_infinity=False)),
                        uly=data.draw(st.floats(0, 100, allow_nan=False, allow_infinity=False)),
                        width=data.draw(st.floats(1, 50, allow_nan=False, allow_infinity=False)),
                        height=data.draw(st.floats(1, 50, allow_nan=False, allow_infinity=False)),
                        absolute_coords=True
                    )
                )
                img.dump(ann)
            elif choice == 1:
                # Add embedding
                bbox = BoundingBox(
                    ulx=data.draw(st.floats(0, 100, allow_nan=False, allow_infinity=False)),
                    uly=data.draw(st.floats(0, 100, allow_nan=False, allow_infinity=False)),
                    width=data.draw(st.floats(1, 50, allow_nan=False, allow_infinity=False)),
                    height=data.draw(st.floats(1, 50, allow_nan=False, allow_infinity=False)),
                    absolute_coords=True
                )
                img.set_embedding(f"parent_{len(state_ids)}", bbox)
            else:
                # Set dimensions
                img.set_width_height(
                    data.draw(st.floats(1, 1000, allow_nan=False, allow_infinity=False)),
                    data.draw(st.floats(1, 1000, allow_nan=False, allow_infinity=False))
                )

            state_ids.append(img.state_id)

        # All state IDs should be different (modifications change state)
        assert len(set(state_ids)) == len(state_ids)

    @staticmethod
    @mark.property
    @given(
        file_name=st.text(min_size=1, max_size=50).filter(lambda x: len(x.strip()) > 0),
        location=st.text(max_size=100)
    )
    def test_roundtrip_serialization_preserves_metadata(file_name: str, location: str):
        """Roundtrip serialization preserves metadata"""
        img1 = Image(file_name=file_name, location=location)

        data = img1.as_dict()
        img2 = Image(**data)

        assert img2.file_name == img1.file_name
        assert img2.location == img1.location
        assert img2.image_id == img1.image_id

    @staticmethod
    @mark.property
    @given(
        doc_id=st.text(min_size=1, max_size=100)
    )
    def test_document_id_stored(doc_id: str):
        """document_id is stored correctly"""
        img = Image(file_name="test.png", document_id=doc_id)

        assert img.document_id == doc_id

