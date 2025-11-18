# -*- coding: utf-8 -*-
# File: test_image_state.py

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
Testing Image state_id and state management
"""

import numpy as np
from pytest import mark

from dd_core.datapoint import BoundingBox, Image, ImageAnnotation

import shared_test_utils as stu


class TestImageState:
    """Test Image state_id and state management"""

    @staticmethod
    def test_state_id_is_deterministic():
        """state_id is deterministic for same image state"""
        img1 = Image(file_name="test.png", location="/path")
        img2 = Image(file_name="test.png", location="/path")

        assert img1.state_id == img2.state_id


    @staticmethod
    def test_state_id_changes_when_annotation_added(white_image: stu.WhiteImage):
        """state_id changes when annotation is added"""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        initial_state_id = img.state_id

        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann)

        assert img.state_id != initial_state_id

    @staticmethod
    def test_state_id_changes_when_image_added(white_image: stu.WhiteImage):
        """state_id changes when image pixels are set"""
        img = Image(file_name=white_image.file_name)
        initial_state_id = img.state_id

        img.image = white_image.image

        assert img.state_id != initial_state_id

    @staticmethod
    def test_state_id_changes_when_embedding_added():
        """state_id changes when embedding is added"""
        img = Image(file_name="test.png")
        img.set_width_height(100, 100)
        initial_state_id = img.state_id

        bbox = BoundingBox(ulx=10, uly=10, width=50, height=50, absolute_coords=True)
        img.set_embedding(img.image_id, bbox)

        assert img.state_id != initial_state_id

    @staticmethod
    def test_state_id_changes_when_summary_accessed():
        """state_id changes when summary is accessed (and created)"""
        img = Image(file_name="test.png")
        initial_state_id = img.state_id

        _ = img.summary

        assert img.state_id != initial_state_id

    @staticmethod
    def test_state_attributes_list():
        """get_state_attributes returns correct attributes"""
        attrs = Image.get_state_attributes()

        assert "annotations" in attrs
        assert "embeddings" in attrs
        assert "_image" in attrs
        assert "_summary" in attrs


    @staticmethod
    def test_state_id_with_multiple_annotations(white_image: stu.WhiteImage):
        """state_id changes with each annotation added"""
        img = Image(file_name=white_image.file_name)
        state_ids = [img.state_id]

        for i in range(1,3):
            ann = ImageAnnotation(
                category_name=f"test_cat_{i}",
                bounding_box=BoundingBox(ulx=i*10, uly=i*10, width=20, height=20, absolute_coords=True)
            )
            img.dump(ann)
            state_ids.append(img.state_id)

        # All state_ids should be different
        assert len(set(state_ids)) == len(state_ids)

    @staticmethod
    def test_state_id_includes_image_content(white_image: stu.WhiteImage):
        """state_id reflects image content changes"""
        img = Image(file_name=white_image.file_name)

        img.image = np.ones([10, 10, 3], dtype=np.uint8)
        state_id1 = img.state_id

        img._image = np.zeros([10, 10, 3], dtype=np.uint8)
        state_id2 = img.state_id

        assert state_id1 != state_id2

    @staticmethod
    def test_state_id_reflects_embedding_changes():
        """state_id changes when embeddings are modified"""
        img = Image(file_name="test.png")
        img.set_width_height(100, 100)
        state_id1 = img.state_id

        bbox1 = BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        img.set_embedding(state_id1, bbox1)
        state_id2 = img.state_id

        bbox2 = BoundingBox(ulx=30, uly=30, width=25, height=25, absolute_coords=True)
        img.set_embedding(state_id2, bbox2)
        state_id3 = img.state_id

        assert state_id1 != state_id2
        assert state_id2 != state_id3
        assert state_id1 != state_id3


    @staticmethod
    def test_state_id_different_for_different_states():
        """Two images with different states have different state_ids"""
        img1 = Image(file_name="test.png", location="/path")
        img2 = Image(file_name="test.png", location="/path")

        # Add different annotations
        ann1 = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img1.dump(ann1)

        ann2 = ImageAnnotation(
            category_name="test_cat_2",
            bounding_box=BoundingBox(ulx=30, uly=30, width=25, height=25, absolute_coords=True)
        )
        img2.dump(ann2)

        assert img1.state_id != img2.state_id

