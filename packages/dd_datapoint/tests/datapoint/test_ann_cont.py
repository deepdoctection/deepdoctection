# -*- coding: utf-8 -*-
# File: test_cont_ann.py

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
from pydantic import ValidationError

from dd_datapoint.datapoint.annotation import (
    CategoryAnnotation,
    ContainerAnnotation,
)
from dd_datapoint.utils.object_types import get_type


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



class TestContainerAnnotationAdvanced:
    """Advanced tests for ContainerAnnotation"""

    def test_container_annotation_from_dict_with_value_field(self):
        """Test that ContainerAnnotation is correctly identified from dict with value field"""
        data = {
            "category_name": "TEXT",
            "category_id": 1,
            "value": "test_text"
        }
        parent = CategoryAnnotation(category_name="person", external_id="parent_id")
        parent.dump_sub_category(get_type("text"), ContainerAnnotation(**data))
        retrieved = parent.get_sub_category(get_type("text"))
        assert isinstance(retrieved, ContainerAnnotation)
        assert retrieved.value == "test_text"

    def test_container_annotation_set_type_validates_existing_value(self):
        """Test that set_type validates existing value immediately"""
        container = ContainerAnnotation(category_name="TEXT", value="string_value")
        container.set_type("str")

        container2 = ContainerAnnotation(category_name="TEXT", value="string_value")
        with pytest.raises(TypeError):
            container2.set_type("int")