# -*- coding: utf-8 -*-
# File: test_page_factory.py

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
Testing Page factory methods and initialization
"""

from pathlib import Path

from dd_datapoint.datapoint.view import Page
from dd_datapoint.datapoint.image import Image
from dd_datapoint.utils.object_types import LayoutType


class TestPageFactory:
    """Test Page factory methods"""

    def test_from_file_creates_page(self, page_json_path: Path):
        """Page.from_file() creates a Page instance"""
        page = Page.from_file(file_path=str(page_json_path))
        assert isinstance(page, Page)


    def test_from_image_creates_page(self, page_json_path: Path):
        """Page.from_image() creates a Page instance"""
        image = Image.from_file(str(page_json_path))
        page = Page.from_image(image)
        assert isinstance(page, Page)

    def test_page_init_creates_instance(self, page_json_path: Path):
        """Page() constructor creates instance"""
        image = Image.from_file(str(page_json_path))
        page = Page(base_image=image)
        assert isinstance(page, Page)

    def test_page_copy_creates_new_instance(self, page: Page):
        """Page.__copy__() creates a new Page instance"""
        page_copy = page.__copy__()
        assert isinstance(page_copy, Page)
        assert page_copy is not page
        assert page_copy.image_id == page.image_id

