# -*- coding: utf-8 -*-
# File: test_ann_misc.py

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

from dd_core.datapoint.annotation import (
    CategoryAnnotation,
    ContainerAnnotation,
    ImageAnnotation,
)
from dd_core.datapoint.box import BoundingBox
from dd_core.utils.error import UUIDError
from dd_core.utils.object_types import get_type


class TestAnnotationSerialization:
    """Tests for annotation serialization and deserialization"""

    def test_category_annotation_roundtrip(self):
        """Test serializing and deserializing CategoryAnnotation"""
        cat = CategoryAnnotation(
            category_name="test_cat_1",
            category_id=1,
            score=0.95,
            external_id="test_id"
        )
        data = cat.as_dict()
        restored = CategoryAnnotation.from_dict(**data)
        assert restored.category_name == cat.category_name
        assert restored.category_id == cat.category_id
        assert restored.score == cat.score
        assert restored.annotation_id == cat.annotation_id
        assert cat == restored

    def test_image_annotation_roundtrip(self):
        """Test serializing and deserializing ImageAnnotation"""
        bbox = BoundingBox(ulx=10.0, uly=20.0, width=30.0, height=40.0, absolute_coords=True)
        img_ann = ImageAnnotation(
            category_name="test_cat_1",
            category_id=1,
            bounding_box=bbox,
            external_id="test_id"
        )
        data = img_ann.as_dict()
        restored = ImageAnnotation.from_dict(**data)
        assert restored.category_name == img_ann.category_name
        assert restored.bounding_box == img_ann.bounding_box
        assert restored.annotation_id == img_ann.annotation_id
        assert img_ann == restored

    def test_container_annotation_roundtrip(self):
        """Test serializing and deserializing ContainerAnnotation"""
        container = ContainerAnnotation(
            category_name="test_cat_2",
            value="test_value",
            external_id="test_id"
        )
        data = container.as_dict()
        restored = ContainerAnnotation.from_dict(**data)
        assert restored.category_name == container.category_name
        assert restored.value == container.value
        assert restored.annotation_id == container.annotation_id
        assert container == restored

    def test_annotation_with_sub_categories_roundtrip(self):
        """Test serializing annotation with sub-categories"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="test_cat_2", category_id=2, external_id="child_id")
        cat.dump_sub_category("sub_cat_1", sub_cat)
        data = cat.as_dict()
        restored = CategoryAnnotation.from_dict(**data)
        assert get_type("sub_cat_1") in restored.sub_categories
        retrieved_sub = restored.get_sub_category(get_type("sub_cat_1"))
        assert retrieved_sub.category_name == get_type("test_cat_2")
        assert cat == restored

    def test_annotation_with_relationships_roundtrip(self):
        """Test serializing annotation with relationships"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat = CategoryAnnotation(category_name="test_cat_1", external_id="test_id")
        cat.dump_relationship("relationship_1", uuid)
        data = cat.as_dict()
        restored = CategoryAnnotation.from_dict(**data)
        relationships = restored.get_relationship(get_type("relationship_1"))
        assert uuid in relationships
        assert cat == restored









class TestAnnotationValidation:
    """Tests for validation and error handling"""

    def test_relationships_validator_rejects_invalid_uuid(self):
        """Test that relationships validator rejects invalid UUIDs"""
        with pytest.raises(UUIDError):
            CategoryAnnotation(
                category_name="test_cat_1",
                relationships={"relationship_1": ["not_a_uuid"]}
            )

    def test_relationships_validator_accepts_valid_uuids(self):
        """Test that relationships validator accepts valid UUIDs"""
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        cat = CategoryAnnotation(
            category_name="test_cat_1",
            relationships={"relationship_1": [uuid1, uuid2]}
        )
        assert uuid1 in cat.relationships[get_type("relationship_1")]
        assert uuid2 in cat.relationships[get_type("relationship_1")]


    def test_sub_categories_validator_handles_string_keys(self):
        """Test that sub_categories validator converts string keys to ObjectTypes"""
        data = {
            "category_name": "test_cat_1",
            "sub_categories": {
                "sub_cat_1": {
                    "category_name": "test_cat_2",
                    "category_id": 1
                }
            }
        }

        cat = CategoryAnnotation(**data)
        # String key should be converted to ObjectTypes
        assert get_type("sub_cat_1") in cat.sub_categories



