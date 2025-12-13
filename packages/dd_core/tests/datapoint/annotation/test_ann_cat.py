# -*- coding: utf-8 -*-
# File: test_cat_ann.py

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
)
from dd_core.utils.error import AnnotationError, UUIDError
from dd_core.utils.identifier import is_uuid_like
from dd_core.utils.object_types import DefaultType, get_type


class TestCategoryAnnotationBasics:
    """Tests for basic CategoryAnnotation functionality"""

    def test_category_annotation_creation_defaults(self) -> None:
        """Test CategoryAnnotation creation with default values"""
        cat = CategoryAnnotation()
        assert cat.category_name == DefaultType.DEFAULT_TYPE
        assert cat.category_id == -1
        assert cat.score is None
        assert cat.active is True
        assert len(cat.sub_categories) == 0
        assert len(cat.relationships) == 0

    def test_category_annotation_creation_with_values(self) -> None:
        """Test CategoryAnnotation creation with explicit values"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, score=0.95)
        assert cat.category_name == get_type("test_cat_1")
        assert cat.category_id == 1
        assert cat.score == 0.95

    def test_category_id_coercion(self) -> None:
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

    def test_as_dict(self) -> None:
        """Test converting annotation to dict"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, score=0.9)
        result = cat.as_dict()
        assert isinstance(result, dict)
        assert result["category_id"] == 1
        assert result["score"] == 0.9

    def test_from_dict(self) -> None:
        """Test creating annotation from dict"""
        data = {"category_name": "test_cat_1", "category_id": 2, "score": 0.85}
        cat = CategoryAnnotation.from_dict(**data)
        assert cat.category_name == get_type("test_cat_1")
        assert cat.category_id == 2
        assert cat.score == 0.85


class TestSubCategories:
    """Tests for sub-category management"""

    def test_dump_sub_category_basic(self) -> None:
        """Test dumping a sub-category to an annotation"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1)
        sub_cat = CategoryAnnotation(category_name="test_cat_2", category_id=2)
        cat.dump_sub_category("sub_cat_1", sub_cat)
        retrieved = cat.get_sub_category(get_type("sub_cat_1"))
        assert retrieved.category_name == get_type("test_cat_2")
        assert retrieved.category_id == 2

    def test_dump_sub_category_generates_annotation_id(self) -> None:
        """Test that dumping sub-category generates annotation_id"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="test_cat_2", category_id=2)
        cat.dump_sub_category("sub_cat_1", sub_cat)
        retrieved = cat.get_sub_category(get_type("sub_cat_1"))
        assert is_uuid_like(retrieved.annotation_id)

    def test_dump_sub_category_duplicate_raises_error(self) -> None:
        """Test that dumping duplicate sub-category raises error"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, external_id="parent_id")
        sub_cat_1 = CategoryAnnotation(category_name="test_cat_2", category_id=2)
        sub_cat_2 = CategoryAnnotation(category_name="test_cat_3", category_id=3)
        cat.dump_sub_category("sub_cat_1", sub_cat_1)

        with pytest.raises(AnnotationError):
            cat.dump_sub_category("sub_cat_1", sub_cat_2)

    def test_dump_sub_category_with_external_id(self) -> None:
        """Test that sub-category with external_id keeps its annotation_id"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, external_id="parent_id")
        external_uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        sub_cat = CategoryAnnotation(category_name="test_cat_2", category_id=2, external_id=external_uuid)
        cat.dump_sub_category("sub_cat_1", sub_cat)
        retrieved = cat.get_sub_category(get_type("sub_cat_1"))
        assert retrieved.annotation_id == external_uuid

    def test_get_sub_category_nonexistent_raises_error(self) -> None:
        """Test that getting nonexistent sub-category raises error"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        with pytest.raises(KeyError):
            cat.get_sub_category(get_type("nonexistent"))

    def test_remove_sub_category(self) -> None:
        """Test removing a sub-category"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1, external_id="parent_id")
        sub_cat = CategoryAnnotation(category_name="test_cat_2", category_id=2)
        cat.dump_sub_category("sub_cat_1", sub_cat)
        assert get_type("sub_cat_1") in cat.sub_categories
        cat.remove_sub_category(get_type("sub_cat_1"))
        assert get_type("sub_cat_1") not in cat.sub_categories

    def test_remove_nonexistent_sub_category_no_error(self) -> None:
        """Test that removing nonexistent sub-category doesn't raise error"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        cat.remove_sub_category(get_type("non_existent"))

    def test_sub_categories_from_dict(self) -> None:
        """Test creating annotation with sub_categories from dict"""
        data = {
            "category_name": "test_cat_1",
            "category_id": 1,
            "sub_categories": {"sub_cat_1": {"category_name": "test_cat_2", "category_id": 2}},
        }
        cat = CategoryAnnotation(**data)
        assert get_type("sub_cat_1") in cat.sub_categories
        retrieved = cat.get_sub_category(get_type("sub_cat_1"))
        assert retrieved.category_name == get_type("test_cat_2")
        assert retrieved.category_id == 2


class TestRelationships:
    """Tests for relationship management"""

    def test_dump_relationship_basic(self) -> None:
        """Test dumping a relationship"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        rel_uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat.dump_relationship("relationship_1", rel_uuid)
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert rel_uuid in relationships

    def test_dump_relationship_invalid_uuid_raises_error(self) -> None:
        """Test that dumping relationship with invalid UUID raises error"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        with pytest.raises(UUIDError):
            cat.dump_relationship("relationship_1", "not_a_uuid")

    def test_dump_relationship_multiple_to_same_key(self) -> None:
        """Test dumping multiple relationships to same key"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        cat.dump_relationship("relationship_1", uuid1)
        cat.dump_relationship("relationship_1", uuid2)
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert uuid1 in relationships
        assert uuid2 in relationships
        assert len(relationships) == 2

    def test_dump_relationship_no_duplicates(self) -> None:
        """Test that duplicate relationship ids are not added"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat.dump_relationship("relationship_1", uuid)
        cat.dump_relationship("relationship_1", uuid)
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert len(relationships) == 1

    def test_remove_relationship_specific_id(self) -> None:
        """Test removing specific relationship id"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        cat.dump_relationship("relationship_1", uuid1)
        cat.dump_relationship("relationship_1", uuid2)
        cat.remove_relationship(get_type("relationship_1"), uuid1)
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert uuid1 not in relationships
        assert uuid2 in relationships

    def test_remove_relationship_all_for_key(self) -> None:
        """Test removing all relationships for a key"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        cat.dump_relationship("relationship_1", uuid1)
        cat.dump_relationship("relationship_1", uuid2)
        cat.remove_relationship(get_type("relationship_1"))
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert len(relationships) == 0

    def test_relationships_from_dict(self) -> None:
        """Test creating annotation with relationships from dict"""
        uuid1 = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        uuid2 = "d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        data = {"category_name": "test_cat_1", "relationships": {"relationship_1": [uuid1, uuid2]}}
        cat = CategoryAnnotation(**data)
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert uuid1 in relationships
        assert uuid2 in relationships

    def test_relationships_from_dict_deduplication(self) -> None:
        """Test that duplicate relationship ids are deduplicated from dict"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        data = {"category_name": "test_cat_1", "relationships": {"relationship_1": [uuid, uuid]}}
        cat = CategoryAnnotation(**data)
        relationships = cat.get_relationship(get_type("relationship_1"))
        assert len(relationships) == 1


class TestAnnotationEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_service_and_model_id_fields(self) -> None:
        """Test that service_id, model_id, session_id fields work correctly"""
        cat = CategoryAnnotation(
            category_name="test_cat_1", service_id="text_detector", model_id="model_v1", session_id="session_123"
        )
        assert cat.service_id == "text_detector"
        assert cat.model_id == "model_v1"
        assert cat.session_id == "session_123"

    def test_multiple_sub_category_types(self) -> None:
        """Test annotation with multiple different sub-categories"""
        cat = CategoryAnnotation(category_name="test_cat_1", external_id="parent")
        sub1 = CategoryAnnotation(category_name="test_cat_2", category_id=1)
        sub2 = CategoryAnnotation(category_name="test_cat_3", category_id=2)
        container = ContainerAnnotation(category_name="test_cat_4", value="test")
        cat.dump_sub_category("sub_cat_1", sub1)
        cat.dump_sub_category("sub_cat_2", sub2)
        cat.dump_sub_category("sub_cat_3", container)
        assert len(cat.sub_categories) == 3
        assert isinstance(cat.get_sub_category(get_type("sub_cat_3")), ContainerAnnotation)

    def test_complex_nested_structure(self) -> None:
        """Test complex nested annotation structure"""
        parent = CategoryAnnotation(category_name="test_cat_1", external_id="parent_id")
        sub = CategoryAnnotation(category_name="test_cat_2", category_id=1)
        sub_sub = CategoryAnnotation(category_name="test_cat_3", category_id=2, external_id="subsub_id")
        sub.dump_sub_category("sub_cat_2", sub_sub)
        parent.dump_sub_category("sub_cat_1", sub)
        parent.dump_relationship("relationship_1", "c822f8c3-1148-30c4-90eb-cb4896b1ebe5")
        assert len(parent.sub_categories) == 1
        child = parent.get_sub_category(get_type("sub_cat_1"))
        assert len(child.sub_categories) == 1
        nested = child.get_sub_category(get_type("sub_cat_2"))
        assert nested.category_name == get_type("test_cat_3")
        relationship = parent.get_relationship("relationship_1")
        assert "c822f8c3-1148-30c4-90eb-cb4896b1ebe5" in relationship
