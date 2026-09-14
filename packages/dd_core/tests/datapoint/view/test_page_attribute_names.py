# -*- coding: utf-8 -*-
# File: test_page_attribute_names.py

# Copyright 2026 Dr. Janis Meyer. All rights reserved.
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
Testing Page.get_attribute_names/__getattr__ relaxation towards summary sub categories
"""

import pytest

from dd_core.datapoint.annotation import CategoryAnnotation, ContainerAnnotation
from dd_core.datapoint.view import Page

from ...conftest import ObjectTestType


class TestPageAttributeNames:
    """Test Page.get_attribute_names and __getattr__"""

    def test_page_summary_and_page_mapping_are_registered(self, page: Page) -> None:
        """page_summary/page_mapping are part of the registered attribute names"""
        attr_names = page.get_attribute_names()
        assert "page_summary" in attr_names
        assert "page_mapping" in attr_names

    def test_document_summary_and_document_mapping_are_not_registered(self, page: Page) -> None:
        """document_summary/document_mapping are no longer Page attributes"""
        attr_names = page.get_attribute_names()
        assert "document_summary" not in attr_names
        assert "document_mapping" not in attr_names

    def test_all_sensible_property_methods_are_registered(self, page: Page) -> None:
        """All property methods that expose plain values are part of get_attribute_names"""
        attr_names = page.get_attribute_names()
        for name in (
            "image_id",
            "width",
            "height",
            "image",
            "file_name",
            "location",
            "document_id",
            "page_number",
            "b64_image",
            "layouts",
            "words",
            "tables",
            "figures",
            "residual_layouts",
            "text",
            "text_no_line_break",
            "chunks",
        ):
            assert name in attr_names

    def test_summary_and_base_image_are_not_registered(self, page: Page) -> None:
        """summary/base_image return complex domain objects and must be skipped"""
        attr_names = page.get_attribute_names()
        assert "summary" not in attr_names
        assert "base_image" not in attr_names

    def test_get_attribute_names_includes_custom_summary_sub_category(self, page: Page) -> None:
        """A custom key dumped into page.summary is picked up by get_attribute_names"""
        page.summary.dump_sub_category(
            ObjectTestType.SUMMARY_1, CategoryAnnotation(category_name=ObjectTestType.SUMMARY_1, category_id=1)
        )
        assert ObjectTestType.SUMMARY_1.value in page.get_attribute_names()

    def test_custom_summary_sub_category_is_accessible_via_getattr(self, page: Page) -> None:
        """page.my_custom_key resolves to the category_id of a custom summary sub category"""
        page.summary.dump_sub_category(
            ObjectTestType.SUMMARY_1, CategoryAnnotation(category_name=ObjectTestType.SUMMARY_1, category_id=1)
        )
        assert page.summary_1 == 1

    def test_custom_summary_sub_category_container_value_is_accessible_via_getattr(self, page: Page) -> None:
        """page.my_custom_key resolves to the value of a custom summary ContainerAnnotation"""
        page.summary.dump_sub_category(
            ObjectTestType.SUMMARY_1,
            ContainerAnnotation(category_name=ObjectTestType.SUMMARY_1, value="my_value"),
        )
        assert page.summary_1 == "my_value"

    def test_getattr_still_raises_for_unregistered_attribute(self, page: Page) -> None:
        """Attributes that are neither properties nor summary sub categories still raise"""
        with pytest.raises(AttributeError):
            _ = page.some_completely_unknown_attribute
