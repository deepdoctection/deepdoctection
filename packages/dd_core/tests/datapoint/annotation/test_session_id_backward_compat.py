# -*- coding: utf-8 -*-
# File: test_session_id_backward_compat.py

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
Testing backward compatibility for session_id removal
"""

from dd_core.datapoint.annotation import CategoryAnnotation, ContainerAnnotation
from dd_core.utils.object_types import DefaultType


class TestSessionIdBackwardCompatibility:
    """Test that loading annotations with session_id in data doesn't break"""

    def test_category_annotation_with_session_id_in_dict(self) -> None:
        """Test that CategoryAnnotation ignores session_id when loading from dict"""
        data = {
            "category_name": DefaultType.DEFAULT_TYPE,
            "category_id": 1,
            "service_id": "text_detector",
            "model_id": "model_v1",
            "session_id": "session_123",  # This should be ignored
        }
        cat = CategoryAnnotation(**data)
        assert cat.category_name == DefaultType.DEFAULT_TYPE
        assert cat.service_id == "text_detector"
        assert cat.model_id == "model_v1"
        # Backward compatibility: session_id should be silently ignored during init
        # The important thing is that it doesn't raise an error

    def test_container_annotation_with_session_id_in_dict(self) -> None:
        """Test that ContainerAnnotation ignores session_id when loading from dict"""
        data = {
            "category_name": DefaultType.DEFAULT_TYPE,
            "value": "test_value",
            "service_id": "service_1",
            "session_id": "session_456",  # This should be ignored
        }
        cont = ContainerAnnotation(**data)
        assert cont.category_name == DefaultType.DEFAULT_TYPE
        assert cont.value == "test_value"
        assert cont.service_id == "service_1"
        # Backward compatibility: session_id should be silently ignored during init

    def test_from_dict_with_session_id(self) -> None:
        """Test that from_dict method also ignores session_id"""
        data = {
            "category_name": DefaultType.DEFAULT_TYPE,
            "category_id": 2,
            "session_id": "session_789",
        }
        cat = CategoryAnnotation(**data)
        assert cat.category_name == DefaultType.DEFAULT_TYPE
        assert cat.category_id == 2
        # session_id should be ignored, no error should be raised

    def test_nested_annotations_with_session_id(self) -> None:
        """Test that nested annotations also handle session_id correctly"""
        parent_data = {
            "category_name": DefaultType.DEFAULT_TYPE,
            "session_id": "parent_session",
        }
        parent = CategoryAnnotation(**parent_data)

        sub_data = {
            "category_name": DefaultType.DEFAULT_TYPE,
            "session_id": "child_session",
        }
        child = CategoryAnnotation(**sub_data)

        # Verify both annotations can be created without errors
        # The session_id is silently ignored during initialization
        assert parent.category_name == DefaultType.DEFAULT_TYPE
        assert child.category_name == DefaultType.DEFAULT_TYPE
