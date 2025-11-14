# -*- coding: utf-8 -*-
# File: test_annotation.py

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
Unit tests for the dd_datapoint.datapoint.annotation module
"""

import pytest

from pydantic import ValidationError

from dd_datapoint.datapoint.annotation import (
    CategoryAnnotation,
    ContainerAnnotation,
    ImageAnnotation,
)
from dd_datapoint.datapoint.box import BoundingBox
from dd_datapoint.utils.error import AnnotationError, UUIDError
from dd_datapoint.utils.identifier import get_uuid, is_uuid_like
from dd_datapoint.utils.object_types import DefaultType, get_type


class TestCategoryAnnotationBasics:
    """Tests for basic CategoryAnnotation functionality"""

    def test_category_annotation_creation_defaults(self):
        """Test CategoryAnnotation creation with default values"""
        cat = CategoryAnnotation()
        assert cat.category_name == DefaultType.DEFAULT_TYPE
        assert cat.category_id == -1
        assert cat.score is None
        assert cat.active is True
        assert len(cat.sub_categories) == 0
        assert len(cat.relationships) == 0

    def test_category_annotation_creation_with_values(self):
        """Test CategoryAnnotation creation with explicit values"""
        cat = CategoryAnnotation(category_name="person", category_id=1, score=0.95)
        assert cat.category_name == get_type("person")
        assert cat.category_id == 1
        assert cat.score == 0.95

    def test_category_name_string_conversion(self):
        """Test that category_name string is converted to ObjectTypes"""
        cat = CategoryAnnotation(category_name="foo")
        assert cat.category_name == get_type("foo")

    def test_category_id_coercion(self):
        """Test category_id coercion from various inputs"""
        # String to int
        cat1 = CategoryAnnotation(category_id="5")
        assert cat1.category_id == 5

        # None to default
        cat2 = CategoryAnnotation(category_id=None)
        assert cat2.category_id == -1

        # Empty string to default
        cat3 = CategoryAnnotation(category_id="")
        assert cat3.category_id == -1

    def test_deactivate(self):
        """Test deactivating an annotation"""
        cat = CategoryAnnotation(category_name="TEST")
        assert cat.active is True
        cat.deactivate()
        assert cat.active is False

    def test_as_dict(self):
        """Test converting annotation to dict"""
        cat = CategoryAnnotation(category_name="TEST", category_id=1, score=0.9)
        result = cat.as_dict()
        assert isinstance(result, dict)
        assert result["category_id"] == 1
        assert result["score"] == 0.9

    def test_from_dict(self):
        """Test creating annotation from dict"""
        data = {"category_name": "TEST", "category_id": 2, "score": 0.85}
        cat = CategoryAnnotation.from_dict(**data)
        assert cat.category_name == get_type("TEST")
        assert cat.category_id == 2
        assert cat.score == 0.85


class TestAnnotationIdGeneration:
    """Tests for annotation_id generation and determinism"""

    def test_annotation_id_without_external_id_raises_error(self):
        """Test that accessing annotation_id without dumping raises error"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1)
        with pytest.raises(AnnotationError):
            _ = cat.annotation_id

    def test_annotation_id_from_external_id_string(self):
        """Test annotation_id generation from external_id string"""
        external_id = "my_external_id"
        cat = CategoryAnnotation(category_name="FOO", external_id=external_id)
        expected_id = get_uuid(external_id)
        assert cat.annotation_id == expected_id
        assert is_uuid_like(cat.annotation_id)

    def test_annotation_id_from_external_id_uuid(self):
        """Test annotation_id when external_id is already a UUID"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat = CategoryAnnotation(category_name="FOO", external_id=uuid)
        assert cat.annotation_id == uuid


    def test_annotation_id_deterministic(self):
        """Test that annotation_id is deterministic based on defining attributes"""
        cat1 = CategoryAnnotation(category_name="FOO", category_id=1)
        cat2 = CategoryAnnotation(category_name="FOO", category_id=1)
        
        # Set annotation_id using the same logic
        ann_id_1 = CategoryAnnotation.set_annotation_id(cat1)
        ann_id_2 = CategoryAnnotation.set_annotation_id(cat2)
        
        assert ann_id_1 == ann_id_2

    def test_annotation_id_different_for_different_attributes(self):
        """Test that different defining attributes generate different annotation_ids"""
        cat1 = CategoryAnnotation(category_name="FOO", category_id=1)
        cat2 = CategoryAnnotation(category_name="BAR", category_id=1)
        
        ann_id_1 = CategoryAnnotation.set_annotation_id(cat1)
        ann_id_2 = CategoryAnnotation.set_annotation_id(cat2)
        
        assert ann_id_1 != ann_id_2

    def test_annotation_id_setter_valid_uuid(self):
        """Test setting annotation_id with valid UUID"""
        cat = CategoryAnnotation(category_name="FOO")
        valid_uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat.annotation_id = valid_uuid
        assert cat.annotation_id == valid_uuid

    def test_annotation_id_setter_invalid_uuid(self):
        """Test that setting annotation_id with invalid UUID raises error"""
        cat = CategoryAnnotation(category_name="FOO")
        with pytest.raises(AnnotationError):
            cat.annotation_id = "not_a_uuid"

    def test_annotation_id_cannot_be_reset(self):
        """Test that annotation_id cannot be changed once set"""
        cat = CategoryAnnotation(category_name="FOO", external_id="test_id")
        with pytest.raises(AnnotationError):
            cat.annotation_id = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"



class TestSubCategories:
    """Tests for sub-category management"""

    def test_dump_sub_category_basic(self):
        """Test dumping a sub-category to an annotation"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1)
        sub_cat = CategoryAnnotation(category_name="BAR", category_id=2)
        
        cat.dump_sub_category("sub_category_name", sub_cat)
        
        retrieved = cat.get_sub_category(get_type("sub_category_name"))
        assert retrieved.category_name == get_type("BAR")
        assert retrieved.category_id == 2

    def test_dump_sub_category_generates_annotation_id(self):
        """Test that dumping sub-category generates annotation_id"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="BAR", category_id=2)
        
        cat.dump_sub_category("sub_category_name", sub_cat)
        
        retrieved = cat.get_sub_category(get_type("sub_category_name"))
        assert is_uuid_like(retrieved.annotation_id)

    def test_dump_sub_category_with_context(self):
        """Test that sub-category annotation_id includes context"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="BAR", category_id=2)
        context_id = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        
        cat.dump_sub_category("sub_category_name", sub_cat, context_id)
        
        retrieved = cat.get_sub_category(get_type("sub_category_name"))
        # The annotation_id should be deterministic based on attributes and context
        assert is_uuid_like(retrieved.annotation_id)

    def test_dump_sub_category_duplicate_raises_error(self):
        """Test that dumping duplicate sub-category raises error"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="parent_id")
        sub_cat_1 = CategoryAnnotation(category_name="BAR", category_id=2)
        sub_cat_2 = CategoryAnnotation(category_name="BAZ", category_id=3)
        
        cat.dump_sub_category("sub_category_name", sub_cat_1)
        
        with pytest.raises(AnnotationError):
            cat.dump_sub_category("sub_category_name", sub_cat_2)

    def test_dump_sub_category_with_external_id(self):
        """Test that sub-category with external_id keeps its annotation_id"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="parent_id")
        external_uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        sub_cat = CategoryAnnotation(category_name="BAR", category_id=2, external_id=external_uuid)
        cat.dump_sub_category("sub_category_name", sub_cat)
        retrieved = cat.get_sub_category(get_type("sub_category_name"))
        assert retrieved.annotation_id == external_uuid

    def test_get_sub_category_nonexistent_raises_error(self):
        """Test that getting nonexistent sub-category raises error"""
        cat = CategoryAnnotation(category_name="FOO")
        with pytest.raises(KeyError):
            cat.get_sub_category(get_type("nonexistent"))

    def test_remove_sub_category(self):
        """Test removing a sub-category"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="BAR", category_id=2)
        
        cat.dump_sub_category("sub_category_name", sub_cat)
        assert get_type("sub_category_name") in cat.sub_categories
        
        cat.remove_sub_category(get_type("sub_category_name"))
        assert get_type("sub_category_name") not in cat.sub_categories

    def test_remove_nonexistent_sub_category_no_error(self):
        """Test that removing nonexistent sub-category doesn't raise error"""
        cat = CategoryAnnotation(category_name="FOO")
        # Should not raise
        cat.remove_sub_category(get_type("nonexistent"))

    def test_sub_categories_from_dict(self):
        """Test creating annotation with sub_categories from dict"""
        data = {
            "category_name": "FOO",
            "category_id": 1,
            "sub_categories": {
                "sub_category_name": {
                    "category_name": "BAR",
                    "category_id": 2
                }
            }
        }
        cat = CategoryAnnotation(**data)
        assert get_type("sub_category_name") in cat.sub_categories
        retrieved = cat.get_sub_category(get_type("sub_category_name"))
        assert retrieved.category_name == get_type("BAR")


class TestRelationships:
    """Tests for relationship management"""

    def test_dump_relationship_basic(self):
        """Test dumping a relationship"""
        cat = CategoryAnnotation(category_name="FOO")
        rel_uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        
        cat.dump_relationship("relationship_name", rel_uuid)
        
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert rel_uuid in relationships

    def test_dump_relationship_invalid_uuid_raises_error(self):
        """Test that dumping relationship with invalid UUID raises error"""
        cat = CategoryAnnotation(category_name="FOO")
        
        with pytest.raises(UUIDError):
            cat.dump_relationship("relationship_name", "not_a_uuid")

    def test_dump_relationship_multiple_to_same_key(self):
        """Test dumping multiple relationships to same key"""
        cat = CategoryAnnotation(category_name="FOO")
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        
        cat.dump_relationship("relationship_name", uuid1)
        cat.dump_relationship("relationship_name", uuid2)
        
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert uuid1 in relationships
        assert uuid2 in relationships
        assert len(relationships) == 2

    def test_dump_relationship_no_duplicates(self):
        """Test that duplicate relationship ids are not added"""
        cat = CategoryAnnotation(category_name="FOO")
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        
        cat.dump_relationship("relationship_name", uuid)
        cat.dump_relationship("relationship_name", uuid)
        
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert len(relationships) == 1


    def test_remove_relationship_specific_id(self):
        """Test removing specific relationship id"""
        cat = CategoryAnnotation(category_name="FOO")
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        
        cat.dump_relationship("relationship_name", uuid1)
        cat.dump_relationship("relationship_name", uuid2)
        
        cat.remove_relationship(get_type("relationship_name"), uuid1)
        
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert uuid1 not in relationships
        assert uuid2 in relationships

    def test_remove_relationship_all_for_key(self):
        """Test removing all relationships for a key"""
        cat = CategoryAnnotation(category_name="FOO")
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        
        cat.dump_relationship("relationship_name", uuid1)
        cat.dump_relationship("relationship_name", uuid2)
        
        cat.remove_relationship(get_type("relationship_name"))
        
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert len(relationships) == 0

    def test_relationships_from_dict(self):
        """Test creating annotation with relationships from dict"""
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        data = {
            "category_name": "FOO",
            "relationships": {
                "relationship_name": [uuid1, uuid2]
            }
        }
        cat = CategoryAnnotation(**data)
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert uuid1 in relationships
        assert uuid2 in relationships

    def test_relationships_from_dict_deduplication(self):
        """Test that duplicate relationship ids are deduplicated from dict"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        data = {
            "category_name": "FOO",
            "relationships": {
                "relationship_name": [uuid, uuid]
            }
        }
        cat = CategoryAnnotation(**data)
        relationships = cat.get_relationship(get_type("relationship_name"))
        assert len(relationships) == 1


class TestStateId:
    """Tests for state_id generation"""

    def test_state_id_generation_basic(self):
        """Test basic state_id generation"""
        cat = CategoryAnnotation(
            category_name="FOO", 
            category_id=1, 
            external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        state_id = cat.state_id
        assert is_uuid_like(state_id)

    def test_state_id_includes_sub_categories(self):
        """Test that state_id changes with sub-categories"""
        cat1 = CategoryAnnotation(
            category_name="FOO", 
            category_id=1, 
            external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        state_id_1 = cat1.state_id
        
        cat2 = CategoryAnnotation(
            category_name="FOO", 
            category_id=1, 
            external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        sub_cat = CategoryAnnotation(
            category_name="BAR", 
            category_id=2, 
            external_id="d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        )
        cat2.dump_sub_category("sub_category_name", sub_cat)
        state_id_2 = cat2.state_id
        
        assert state_id_1 != state_id_2

    def test_state_id_includes_active_status(self):
        """Test that state_id changes with active status"""
        cat1 = CategoryAnnotation(
            category_name="FOO", 
            category_id=1, 
            external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        state_id_1 = cat1.state_id
        
        cat2 = CategoryAnnotation(
            category_name="FOO", 
            category_id=1, 
            external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        cat2.deactivate()
        state_id_2 = cat2.state_id
        
        assert state_id_1 != state_id_2


class TestImageAnnotation:
    """Tests for ImageAnnotation class"""

    def test_image_annotation_creation_basic(self):
        """Test basic ImageAnnotation creation"""
        img_ann = ImageAnnotation(
            category_name="PERSON",
            category_id=1,
            bounding_box=BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        )
        assert img_ann.category_name == get_type("PERSON")
        assert img_ann.bounding_box is not None

    def test_image_annotation_bounding_box_from_dict(self):
        """Test creating ImageAnnotation with bounding_box from dict"""
        data = {
            "category_name": "PERSON",
            "bounding_box": {
                "ulx": 10,
                "uly": 20,
                "width": 30,
                "height": 40,
                "absolute_coords": True
            }
        }
        img_ann = ImageAnnotation(**data)
        assert img_ann.bounding_box.ulx == 10.0
        assert img_ann.bounding_box.uly == 20.0

    def test_image_annotation_get_bounding_box_basic(self):
        """Test getting bounding box from ImageAnnotation"""
        bbox = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        img_ann = ImageAnnotation(category_name="PERSON", bounding_box=bbox)
        
        retrieved_bbox = img_ann.get_bounding_box()
        assert retrieved_bbox == bbox

    def test_image_annotation_get_bounding_box_none_raises_error(self):
        """Test that getting bounding box when None raises error"""
        img_ann = ImageAnnotation(category_name="PERSON", external_id="test")
        
        with pytest.raises(AnnotationError):
            img_ann.get_bounding_box()


    def test_image_annotation_annotation_id_determinism(self):
        """Test that ImageAnnotation annotation_id is deterministic based on bounding_box"""
        bbox1 = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        bbox2 = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        
        img_ann1 = ImageAnnotation(category_name="PERSON", bounding_box=bbox1)
        img_ann2 = ImageAnnotation(category_name="PERSON", bounding_box=bbox2)
        
        ann_id_1 = ImageAnnotation.set_annotation_id(img_ann1)
        ann_id_2 = ImageAnnotation.set_annotation_id(img_ann2)
        
        # Same bounding box should produce same annotation_id
        assert ann_id_1 == ann_id_2

    def test_image_annotation_annotation_id_different_boxes(self):
        """Test that different bounding boxes produce different annotation_ids"""
        bbox1 = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        bbox2 = BoundingBox(ulx=15, uly=25, width=35, height=45, absolute_coords=True)
        
        img_ann1 = ImageAnnotation(category_name="PERSON", bounding_box=bbox1)
        img_ann2 = ImageAnnotation(category_name="PERSON", bounding_box=bbox2)
        
        ann_id_1 = ImageAnnotation.set_annotation_id(img_ann1)
        ann_id_2 = ImageAnnotation.set_annotation_id(img_ann2)
        
        assert ann_id_1 != ann_id_2


class TestContainerAnnotation:
    """Tests for ContainerAnnotation class with type-aware value handling"""

    def test_container_annotation_creation_basic(self):
        container = ContainerAnnotation(category_name="TEXT", value="Hello World")
        assert container.value == "Hello World"
        assert isinstance(container.value, str)
        assert container.value_type == "str"

    def test_container_annotation_value_coercion_int_untyped(self):
        container = ContainerAnnotation(category_name="NUMBER", value=42)
        assert container.value == 42
        assert isinstance(container.value, int)
        assert container.value_type == "int"

    def test_container_annotation_set_type_str(self):
        container = ContainerAnnotation(category_name="TEXT")
        container.set_type("str")
        container.value = "test"
        assert container.value == "test"

    def test_container_annotation_set_type_int_assignment_preserved(self):
        container = ContainerAnnotation(category_name="NUMBER")
        container.set_type("int")
        container.value = 42
        assert container.value == 42

    def test_container_annotation_set_type_float(self):
        container = ContainerAnnotation(category_name="NUMBER")
        container.set_type("float")
        container.value = 3.14
        assert container.value == 3.14
        assert isinstance(container.value, float)

    def test_container_annotation_set_type_list_str(self):
        container = ContainerAnnotation(category_name="TEXT", value=["a", "b"])
        assert container.value == ["a", "b"]
        container.set_type("list[str]")

    def test_container_annotation_list_str_rejects_non_str(self):
        with pytest.raises(ValidationError):
            _ = ContainerAnnotation(category_name="TEXT", value=["a", 2])

    def test_container_annotation_set_type_none_disables_validation_and_coerces(self):
        container = ContainerAnnotation(category_name="NUMBER", value=42)
        container.value = 42
        container.set_type("str")
        assert container.value == "42"
        assert isinstance(container.value, str)

    def test_container_annotation_set_type_invalid_raises_error(self):
        container = ContainerAnnotation(category_name="TEXT")
        with pytest.raises(ValueError):
            container.set_type("invalid_type")

    def test_container_validates_and_converts_value(self):
        container = ContainerAnnotation(category_name="TEXT")
        container.set_type("str")
        container.value = 456
        assert container.value == "456"

    def test_container_value_type_raises_value_error_when_None(self):
        container = ContainerAnnotation(category_name="NUMBER", value=5)
        with pytest.raises(ValueError, match="type cannot be None"):
            container.set_type(None)




class TestAnnotationSerialization:
    """Tests for annotation serialization and deserialization"""

    def test_category_annotation_roundtrip(self):
        """Test serializing and deserializing CategoryAnnotation"""
        cat = CategoryAnnotation(
            category_name="FOO",
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
            category_name="PERSON",
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
            category_name="TEXT",
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
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="BAR", category_id=2, external_id="child_id")
        cat.dump_sub_category("sub_category_name", sub_cat)
        
        data = cat.as_dict()
        restored = CategoryAnnotation.from_dict(**data)
        
        assert get_type("sub_category_name") in restored.sub_categories
        retrieved_sub = restored.get_sub_category(get_type("sub_category_name"))
        assert retrieved_sub.category_name == get_type("BAR")
        assert cat == restored

    def test_annotation_with_relationships_roundtrip(self):
        """Test serializing annotation with relationships"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat = CategoryAnnotation(category_name="FOO", external_id="test_id")
        cat.dump_relationship("relationship_name", uuid)
        
        data = cat.as_dict()
        restored = CategoryAnnotation.from_dict(**data)
        
        relationships = restored.get_relationship(get_type("relationship_name"))
        assert uuid in relationships
        assert cat == restored


class TestAnnotationEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_service_and_model_id_fields(self):
        """Test that service_id, model_id, session_id fields work correctly"""
        cat = CategoryAnnotation(
            category_name="FOO",
            service_id="text_detector",
            model_id="model_v1",
            session_id="session_123"
        )
        assert cat.service_id == "text_detector"
        assert cat.model_id == "model_v1"
        assert cat.session_id == "session_123"

    def test_annotation_repr(self):
        """Test that __repr__ works correctly"""
        cat = CategoryAnnotation(category_name="FOO", category_id=1, external_id="test")
        repr_str = repr(cat)
        assert "CategoryAnnotation" in repr_str
        assert "foo" in repr_str # category_name is converted to lowercase ObjectType

    def test_image_annotation_repr(self):
        """Test that ImageAnnotation __repr__ works"""
        bbox = BoundingBox(ulx=10.0, uly=20.0, width=30.0, height=40.0, absolute_coords=True)
        img_ann = ImageAnnotation(category_name="PERSON", bounding_box=bbox, external_id="test")
        repr_str = repr(img_ann)
        assert "ImageAnnotation" in repr_str

    def test_container_annotation_repr(self):
        """Test that ContainerAnnotation __repr__ works"""
        container = ContainerAnnotation(category_name="TEXT", value="test")
        repr_str = repr(container)
        assert "ContainerAnnotation" in repr_str

    def test_multiple_sub_category_types(self):
        """Test annotation with multiple different sub-categories"""
        cat = CategoryAnnotation(category_name="FOO", external_id="parent")
        sub1 = CategoryAnnotation(category_name="SUB1", category_id=1)
        sub2 = CategoryAnnotation(category_name="SUB2", category_id=2)
        container = ContainerAnnotation(category_name="TEXT", value="test")
        
        cat.dump_sub_category("sub_category_name", sub1)
        cat.dump_sub_category("sub_category_name_2", sub2)
        cat.dump_sub_category("sub_category_name_3", container)
        
        assert len(cat.sub_categories) == 3
        assert isinstance(cat.get_sub_category(get_type("sub_category_name_3")), ContainerAnnotation)

    def test_complex_nested_structure(self):
        """Test complex nested annotation structure"""
        # Create parent with sub-categories and relationships
        parent = CategoryAnnotation(category_name="FOO", external_id="parent_id")
        
        # Add sub-category with its own sub-category
        sub = CategoryAnnotation(category_name="BAR", category_id=1)
        sub_sub = CategoryAnnotation(category_name="TEST", category_id=2, external_id="subsub_id")
        sub.dump_sub_category("sub_category_name_2", sub_sub)
        parent.dump_sub_category("sub_category_name", sub)
        
        # Add relationship
        parent.dump_relationship("relationship_name", "c822f8c3-1148-30c4-90eb-cb4896b1ebe5")
        
        # Verify structure
        assert len(parent.sub_categories) == 1
        child = parent.get_sub_category(get_type("sub_category_name"))
        assert len(child.sub_categories) == 1
        nested = child.get_sub_category(get_type("sub_category_name_2"))



class TestAnnotationIdContextPropagation:
    """Tests for annotation_id context propagation through nested structures"""

    def test_sub_category_annotation_id_includes_parent_context(self):
        """Test that sub-category annotation_id depends on parent annotation_id"""
        parent1 = CategoryAnnotation(category_name="person", category_id=1, external_id="parent1")
        sub1 = CategoryAnnotation(category_name="number", category_id=2)
        parent1.dump_sub_category("sub_category_name", sub1)
        sub1_id = parent1.get_sub_category(get_type("sub_category_name")).annotation_id

        parent2 = CategoryAnnotation(category_name="person", category_id=1, external_id="parent2")
        sub2 = CategoryAnnotation(category_name="number", category_id=2)
        parent2.dump_sub_category("sub_category_name", sub2)
        sub2_id = parent2.get_sub_category(get_type("sub_category_name")).annotation_id

        # Same sub-category attributes but different parent -> different annotation_ids
        assert sub1_id != sub2_id

    def test_sub_category_deterministic_with_same_parent(self):
        """Test that sub-category annotation_id is deterministic with same parent"""
        parent = CategoryAnnotation(category_name="person", category_id=1, external_id="parent")

        sub1 = CategoryAnnotation(category_name="number", category_id=2)
        parent.dump_sub_category("sub_category_name", sub1)

        # Get annotation_id
        first_id = parent.get_sub_category(get_type("sub_category_name")).annotation_id

        # Remove and re-add same sub-category
        parent.remove_sub_category(get_type("sub_category_name"))
        sub2 = CategoryAnnotation(category_name="number", category_id=2)
        parent.dump_sub_category("sub_category_name", sub2)
        second_id = parent.get_sub_category(get_type("sub_category_name")).annotation_id

        # Should be deterministic
        assert first_id == second_id



class TestContainerAnnotationAdvanced:
    """Advanced tests for ContainerAnnotation"""

    def test_container_annotation_from_dict_with_value_field(self):
        """Test that ContainerAnnotation is correctly identified from dict with value field"""
        data = {
            "category_name": "TEXT",
            "category_id": 1,
            "value": "test_text"
        }

        # When loading as sub_category, should be recognized as ContainerAnnotation
        parent = CategoryAnnotation(category_name="person", external_id="parent_id")
        parent.dump_sub_category(get_type("text"), ContainerAnnotation(**data))

        retrieved = parent.get_sub_category(get_type("text"))
        assert isinstance(retrieved, ContainerAnnotation)
        assert retrieved.value == "test_text"

    def test_container_annotation_value_list_coercion(self):
        """Test that list values are coerced to list of strings"""
        container = ContainerAnnotation(category_name="NUMBERS", value=[1, 2, 3])
        assert container.value == ["1", "2", "3"]
        assert all(isinstance(v, str) for v in container.value)

    def test_container_annotation_set_type_validates_existing_value(self):
        """Test that set_type validates existing value immediately"""
        container = ContainerAnnotation(category_name="TEXT", value="string_value")

        # Setting correct type should work
        container.set_type("str")

        # But if we have wrong value and try to set type, should fail
        container2 = ContainerAnnotation(category_name="TEXT", value="string_value")
        with pytest.raises(TypeError):
            container2.set_type("int")

    def test_container_annotation_type_validation_list_str(self):
        """Test list[str] type validation catches incorrect lists"""
        container = ContainerAnnotation(category_name="MIXED", value=[1, "two", 3])
        # Value will be coerced to ["1", "two", "3"] first
        container.set_type("list[str]")
        assert container.value == ["1", "two", "3"]


class TestAnnotationValidation:
    """Tests for validation and error handling"""

    def test_relationships_validator_rejects_invalid_uuid(self):
        """Test that relationships validator rejects invalid UUIDs"""
        with pytest.raises(UUIDError):
            CategoryAnnotation(
                category_name="FOO",
                relationships={"related": ["not_a_uuid"]}
            )

    def test_relationships_validator_accepts_valid_uuids(self):
        """Test that relationships validator accepts valid UUIDs"""
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"

        cat = CategoryAnnotation(
            category_name="FOO",
            relationships={"related": [uuid1, uuid2]}
        )

        assert uuid1 in cat.relationships[get_type("related")]
        assert uuid2 in cat.relationships[get_type("related")]

    def test_relationships_must_be_dict(self):
        """Test that relationships must be a dict"""
        with pytest.raises(TypeError):
            CategoryAnnotation(category_name="FOO", relationships="not_a_dict")

    def test_relationships_values_must_be_list(self):
        """Test that relationship values must be lists"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        with pytest.raises(TypeError):
            CategoryAnnotation(
                category_name="FOO",
                relationships={"related": uuid}  # Should be a list
            )

    def test_sub_categories_must_be_dict(self):
        """Test that sub_categories must be a dict"""
        with pytest.raises(TypeError):
            CategoryAnnotation(category_name="FOO", sub_categories="not_a_dict")

    def test_sub_categories_validator_handles_string_keys(self):
        """Test that sub_categories validator converts string keys to ObjectTypes"""
        data = {
            "category_name": "PARENT",
            "sub_categories": {
                "child": {
                    "category_name": "CHILD",
                    "category_id": 1
                }
            }
        }

        cat = CategoryAnnotation(**data)
        # String key should be converted to ObjectTypes
        assert get_type("child") in cat.sub_categories


class TestBoundingBoxIntegration:
    """Tests for BoundingBox integration with ImageAnnotation"""

    def test_bounding_box_none_allowed(self):
        """Test that ImageAnnotation can have None bounding_box"""
        img_ann = ImageAnnotation(category_name="PERSON", bounding_box=None)
        assert img_ann.bounding_box is None

    def test_bounding_box_from_dict_coercion(self):
        """Test BoundingBox coercion from dict in ImageAnnotation"""
        bbox_dict = {
            "ulx": 5.0,
            "uly": 10.0,
            "width": 20.0,
            "height": 30.0,
            "absolute_coords": True
        }

        img_ann = ImageAnnotation(
            category_name="PERSON",
            bounding_box=bbox_dict
        )

        assert isinstance(img_ann.bounding_box, BoundingBox)
        assert img_ann.bounding_box.ulx == 5.0
        assert img_ann.bounding_box.uly == 10.0

    def test_bounding_box_invalid_type_raises_error(self):
        """Test that invalid bounding_box type raises error"""
        with pytest.raises(TypeError):
            ImageAnnotation(
                category_name="PERSON",
                bounding_box="not_a_bbox"
            )
