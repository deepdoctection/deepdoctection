# -*- coding: utf-8 -*-
# File: test_view_relationships.py

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
Testing view relationships and child/parent hierarchies
"""

from dd_core.datapoint.view import Page


class TestViewRelationships:
    """Test view relationship handling"""

    def test_layout_words_relationship(self, page: Page) -> None:
        """Layout.words accesses child words through relationships"""
        layouts = page.layouts
        for layout in layouts:
            words = layout.words
            assert len(words) >= 1 # type: ignore

    def test_table_cells_relationship(self, page: Page) -> None:
        """Table.cells accesses child cells through relationships"""
        tables = page.tables
        for table in tables:
            cells = table.cells
            assert len(cells) >= 1 # type: ignore

    def test_table_rows_relationship(self, page: Page) -> None:
        """Table.rows accesses child rows through relationships"""
        tables = page.tables
        for table in tables:
            rows = table.rows
            assert len(rows) >= 1 # type: ignore

    def test_table_columns_relationship(self, page: Page) -> None:
        """Table.columns accesses child columns through relationships"""
        tables = page.tables
        for table in tables:
            columns = table.columns
            assert len(columns) >= 1 # type: ignore

    def test_relationships_are_resolved_through_base_page(self, page: Page) -> None:
        """Relationships are resolved through base_page reference"""
        layouts = page.layouts
        for layout in layouts:
            if layout.relationships:
                assert layout.base_page == page
