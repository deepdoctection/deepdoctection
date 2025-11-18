# -*- coding: utf-8 -*-
# File: test_page_properties.py
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
Testing Page class basic properties and pass-through attributes
"""
from dd_core.datapoint.view import Page


class TestPageProperties:
    """Test Page class basic properties"""

    def test_page_has_image_id(self, page: Page):
        """Page instance has image_id property"""
        assert hasattr(page, "image_id")
        assert isinstance(page.image_id, str)
        assert len(page.image_id) > 0

    def test_page_has_width(self, page: Page):
        """Page instance has width property"""
        assert hasattr(page, "width")
        assert isinstance(page.width, int)
        assert page.width > 0

    def test_page_has_height(self, page: Page):
        """Page instance has height property"""
        assert hasattr(page, "height")
        assert isinstance(page.height, int)
        assert page.height > 0

    def test_page_has_file_name(self, page: Page):
        """Page instance has file_name property"""
        assert hasattr(page, "file_name")
        assert isinstance(page.file_name, str)

    def test_page_has_location(self, page: Page):
        """Page instance has location property"""
        assert hasattr(page, "location")
        assert isinstance(page.location, str)

    def test_page_has_document_id(self, page: Page):
        """Page instance has document_id property"""
        assert hasattr(page, "document_id")
        assert isinstance(page.document_id, str)

    def test_page_has_page_number(self, page: Page):
        """Page instance has page_number property"""
        assert hasattr(page, "page_number")
        assert isinstance(page.page_number, int)

    def test_page_has_ann_base_view(self, page: Page):
        """Page instance has ann_base_view list"""
        assert hasattr(page, "ann_base_view")
        assert isinstance(page.ann_base_view, list)
