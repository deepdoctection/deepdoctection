# -*- coding: utf-8 -*-
# File: test_ann_ids.py

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
)
from dd_core.utils.error import AnnotationError
from dd_core.utils.identifier import get_uuid, is_uuid_like
from dd_core.utils.object_types import get_type


class TestAnnotationIdGeneration:
    """Tests for annotation_id generation and determinism"""

    def test_annotation_id_without_external_id_raises_error(self):
        """Test that accessing annotation_id without dumping raises error"""
        cat = CategoryAnnotation(category_name="test_cat_1", category_id=1)
        with pytest.raises(AnnotationError):
            _ = cat.annotation_id

    def test_annotation_id_from_external_id_string(self):
        """Test annotation_id generation from external_id string"""
        external_id = "my_external_id"
        cat = CategoryAnnotation(category_name="test_cat_1", external_id=external_id)
        expected_id = get_uuid(external_id)
        assert cat.annotation_id == expected_id
        assert is_uuid_like(cat.annotation_id)

    def test_annotation_id_from_external_id_uuid(self):
        """Test annotation_id when external_id is already a UUID"""
        uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat = CategoryAnnotation(category_name="test_cat_1", external_id=uuid)
        assert cat.annotation_id == uuid

    def test_annotation_id_deterministic(self):
        """Test that annotation_id is deterministic based on defining attributes"""
        cat1 = CategoryAnnotation(category_name="test_cat_1", category_id=1)
        cat2 = CategoryAnnotation(category_name="test_cat_1", category_id=1)

        # Set annotation_id using the same logic
        ann_id_1 = CategoryAnnotation.set_annotation_id(cat1)
        ann_id_2 = CategoryAnnotation.set_annotation_id(cat2)

        assert ann_id_1 == ann_id_2

    def test_annotation_id_different_for_different_attributes(self):
        """Test that different defining attributes generate different annotation_ids"""
        cat1 = CategoryAnnotation(category_name="test_cat_1", category_id=1)
        cat2 = CategoryAnnotation(category_name="test_cat_2", category_id=1)

        ann_id_1 = CategoryAnnotation.set_annotation_id(cat1)
        ann_id_2 = CategoryAnnotation.set_annotation_id(cat2)

        assert ann_id_1 != ann_id_2

    def test_annotation_id_setter_valid_uuid(self):
        """Test setting annotation_id with valid UUID"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        valid_uuid = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        cat.annotation_id = valid_uuid
        assert cat.annotation_id == valid_uuid

    def test_annotation_id_setter_invalid_uuid(self):
        """Test that setting annotation_id with invalid UUID raises error"""
        cat = CategoryAnnotation(category_name="test_cat_1")
        with pytest.raises(AnnotationError):
            cat.annotation_id = "not_a_uuid"

    def test_annotation_id_cannot_be_reset(self):
        """Test that annotation_id cannot be changed once set"""
        cat = CategoryAnnotation(category_name="test_cat_1", external_id="test_id")
        with pytest.raises(AnnotationError):
            cat.annotation_id = "c822f8c3-1148-30c4-90eb-cb4896b1ebe5"


class TestStateId:
    """Tests for state_id generation"""

    def test_state_id_generation_basic(self):
        """Test basic state_id generation"""
        cat = CategoryAnnotation(
            category_name="test_cat_1", category_id=1, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        state_id = cat.state_id
        assert is_uuid_like(state_id)

    def test_state_id_includes_sub_categories(self):
        """Test that state_id changes with sub-categories"""
        cat1 = CategoryAnnotation(
            category_name="test_cat_1", category_id=1, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        state_id_1 = cat1.state_id
        cat2 = CategoryAnnotation(
            category_name="test_cat_1", category_id=1, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        sub_cat = CategoryAnnotation(
            category_name="test_cat_2", category_id=2, external_id="d822f8c3-1148-30c4-90eb-cb4896b1ebe6"
        )
        cat2.dump_sub_category("sub_cat_1", sub_cat)
        state_id_2 = cat2.state_id
        assert state_id_1 != state_id_2

    def test_state_id_includes_active_status(self):
        """Test that state_id changes with active status"""
        cat1 = CategoryAnnotation(
            category_name="test_cat_1", category_id=1, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        state_id_1 = cat1.state_id
        cat2 = CategoryAnnotation(
            category_name="test_cat_1", category_id=1, external_id="c822f8c3-1148-30c4-90eb-cb4896b1ebe5"
        )
        cat2.deactivate()
        state_id_2 = cat2.state_id
        assert state_id_1 != state_id_2


class TestAnnotationIdContextPropagation:
    """Tests for annotation_id context propagation through nested structures"""

    def test_sub_category_annotation_id_includes_parent_context(self):
        """Test that sub-category annotation_id depends on parent annotation_id"""
        parent1 = CategoryAnnotation(category_name="test_cat_3", category_id=1, external_id="parent1")
        sub1 = CategoryAnnotation(category_name="test_cat_4", category_id=2)
        parent1.dump_sub_category("sub_cat_1", sub1)
        sub1_id = parent1.get_sub_category(get_type("sub_cat_1")).annotation_id
        parent2 = CategoryAnnotation(category_name="test_cat_3", category_id=1, external_id="parent2")
        sub2 = CategoryAnnotation(category_name="test_cat_4", category_id=2)
        parent2.dump_sub_category("sub_cat_1", sub2)
        sub2_id = parent2.get_sub_category(get_type("sub_cat_1")).annotation_id
        assert sub1_id != sub2_id

    def test_sub_category_deterministic_with_same_parent(self):
        """Test that sub-category annotation_id is deterministic with same parent"""
        parent = CategoryAnnotation(category_name="test_cat_3", category_id=1, external_id="parent")
        sub1 = CategoryAnnotation(category_name="test_cat_4", category_id=2)
        parent.dump_sub_category("sub_cat_1", sub1)
        first_id = parent.get_sub_category(get_type("sub_cat_1")).annotation_id
        parent.remove_sub_category(get_type("sub_cat_1"))
        sub2 = CategoryAnnotation(category_name="test_cat_4", category_id=2)
        parent.dump_sub_category("sub_cat_1", sub2)
        second_id = parent.get_sub_category(get_type("sub_cat_1")).annotation_id
        assert first_id == second_id
