# -*- coding: utf-8 -*-
# File: test_page_methods.py

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
Testing Page class methods
"""

from dd_core.datapoint.view import Page


class TestPageMethods:
    """Test Page class methods"""

    def test_get_layout_context_returns_list(self, page: Page) -> None:
        """get_layout_context() returns a list"""
        layouts = page.layouts
        if layouts and len(layouts) > 0:
            layout = layouts[0]
            if layout.reading_order is not None:
                context = page.get_layout_context(layout.annotation_id, context_size=1)
                assert {context.reading_order for context in context} == {1, 2}

    def test_get_layout_context_includes_target(self, page: Page) -> None:
        """get_layout_context() includes the target annotation"""
        layouts = page.layouts
        ordered = [l for l in layouts if l.reading_order is not None]
        if ordered:
            layout = ordered[0]
            context = page.get_layout_context(layout.annotation_id, context_size=1)
            ann_ids = [c.annotation_id for c in context]
            assert layout.annotation_id in ann_ids

    def test_get_layout_context_respects_context_size(self, page: Page) -> None:
        """get_layout_context() respects context_size parameter"""
        layouts = page.layouts
        ordered = [l for l in layouts if l.reading_order is not None]
        if len(ordered) >= 3:
            # Get middle element
            middle_idx = len(ordered) // 2
            layout = ordered[middle_idx]
            context = page.get_layout_context(layout.annotation_id, context_size=1)
            # Should return at most 3 items (1 before, target, 1 after)
            assert len(context) <= 3

    def test_save_returns_dict_when_dry(self, page: Page) -> None:
        """save() returns dict when dry=True"""
        result = page.save(dry=True)
        assert isinstance(result, dict)
