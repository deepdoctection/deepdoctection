# -*- coding: utf-8 -*-
# File: test_table_views.py
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
Testing Table view class and its properties
"""
from dd_core.datapoint.view import Page


class TestTableViews:
    """Test Table class functionality"""

    def test_tables_have_cells_property(self, page: Page) -> None:
        """Table instances have cells property"""

        tables = page.tables
        for table in tables:
            cells = table.cells
            assert isinstance(cells, list)

    def test_tables_have_rows_property(self, page: Page) -> None:
        """Table instances have rows property"""

        tables = page.tables
        for table in tables:
            rows = table.rows
            assert isinstance(rows, list)

    def test_tables_have_columns_property(self, page: Page) -> None:
        """Table instances have columns property"""

        tables = page.tables
        for table in tables:
            columns = table.columns
            assert isinstance(columns, list)

    def test_table_row_method_returns_list(self, page: Page) -> None:
        """Table.row() returns a list of cells"""

        tables = page.tables
        for table in tables:
            row_cells = table.row(1) # type: ignore
            assert isinstance(row_cells, list)

    def test_table_column_method_returns_list(self, page: Page) -> None:
        """Table.column() returns a list of cells"""

        tables = page.tables
        for table in tables:
            col_cells = table.column(1) # type: ignore
            assert isinstance(col_cells, list)

    def test_table_column_header_cells_returns_list(self, page: Page) -> None:
        """Table.column_header_cells returns a list"""

        tables = page.tables
        for table in tables:
            headers = table.column_header_cells
            assert isinstance(headers, list)

    def test_table_html_returns_string(self, page: Page) -> None:
        """Table.html returns a string"""

        tables = page.tables
        for table in tables:
            html = table.html
            assert isinstance(html, str)

    def test_table_csv_returns_list_of_lists(self, page: Page) -> None:
        """Table.csv returns list of lists"""

        tables = page.tables
        for table in tables:
            csv = table.csv
            assert isinstance(csv, list)
            for row in csv:
                assert isinstance(row, list)

    def test_table_csv_underscore_returns_list_of_lists(self, page: Page) -> None:
        """Table.csv_ returns list of lists of lists"""

        tables = page.tables
        for table in tables:
            csv_ = table.csv_
            assert isinstance(csv_, list)

    def test_table_str_returns_string(self, page: Page) -> None:
        """Table __str__ returns a string"""

        tables = page.tables
        for table in tables:
            str_repr = str(table)
            assert isinstance(str_repr, str)

    def test_table_text_returns_string(self, page: Page) -> None:
        """Table.text returns a string"""

        tables = page.tables
        for table in tables:
            text = table.text
            assert isinstance(text, str)

    def test_table_words_returns_list(self, page: Page) -> None:
        """Table.words returns a list"""

        tables = page.tables
        for table in tables:
            words = table.words
            assert isinstance(words, list)

    def test_table_kv_header_rows_returns_mapping(self, page: Page) -> None:
        """Table.kv_header_rows() returns a mapping"""

        tables = page.tables
        for table in tables:
            kv_map = table.kv_header_rows(1) # type: ignore
            assert hasattr(kv_map, "__getitem__")

    def test_table_has_number_of_rows(self, page: Page) -> None:
        """Table has number_of_rows attribute"""

        tables = page.tables
        for table in tables:
            assert table.number_of_rows is not None

    def test_table_has_number_of_columns(self, page: Page) -> None:
        """Table has number_of_columns attribute"""

        tables = page.tables
        for table in tables:
            assert table.number_of_columns is not None
