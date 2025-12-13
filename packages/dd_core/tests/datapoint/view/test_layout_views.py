# -*- coding: utf-8 -*-
# File: test_layout_views.py
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
Testing Layout view class and its properties
"""


from dd_core.datapoint.view import Page, Text_


class TestLayoutViews:
    """Test Layout class functionality"""

    def test_layouts_have_words_property(self, page: Page) -> None:
        """Layout instances have words property"""

        layouts = page.layouts
        for layout in layouts:
            assert isinstance(layout.words, list)

    def test_layouts_have_text_property(self, page: Page) -> None:
        """Layout instances have text property"""

        layouts = page.layouts
        for layout in layouts:
            assert isinstance(layout.text, str)

    def test_layouts_have_text_underscore_property(self, page: Page) -> None:
        """Layout instances have text_ property"""

        layouts = page.layouts
        for layout in layouts:
            assert isinstance(layout.text_, Text_)

    def test_layout_get_ordered_words_returns_list(self, page: Page) -> None:
        """Layout.get_ordered_words() returns a list"""

        layouts = page.layouts
        if layouts:
            ordered_words = layouts[0].get_ordered_words() # type: ignore
            assert isinstance(ordered_words, list)

    def test_layout_get_ordered_words_has_reading_order(self, page: Page) -> None:
        """Layout.get_ordered_words() returns words with reading_order"""

        layouts = page.layouts
        if layouts:
            ordered_words = layouts[0].get_ordered_words() # type: ignore
            for word in ordered_words:
                assert word.reading_order is not None

    def test_layout_len_returns_int(self, page: Page) -> None:
        """Layout __len__ returns integer"""

        layouts = page.layouts
        if layouts:
            length = len(layouts[0]) # type: ignore
            assert isinstance(length, int)
            assert length >= 0

    def test_layout_len_equals_text_length(self, page: Page) -> None:
        """Layout __len__ equals length of text"""

        layouts = page.layouts
        if layouts:
            layout = layouts[0]
            assert len(layout) == len(layout.text) # type: ignore

    def test_layout_bbox_returns_list(self, page: Page) -> None:
        """Layout.bbox returns a list of floats"""

        layouts = page.layouts
        if layouts:
            bbox = layouts[0].bbox
            assert isinstance(bbox, list)
            assert len(bbox) == 4
            assert all(isinstance(coord, (int, float)) for coord in bbox)
