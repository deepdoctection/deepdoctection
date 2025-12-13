# -*- coding: utf-8 -*-
# File: test_page_annotations.py
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
Testing Page class annotation retrieval and filtering
"""
from dd_core.datapoint.view import Page


class TestPageAnnotations:
    """Test Page annotation retrieval methods"""

    def test_get_annotation_returns_list(self, page: Page) -> None:
        """get_annotation() returns a list"""
        result = page.get_annotation()
        assert isinstance(result, list)

    def test_get_annotation_filters_by_category_name(self, page: Page) -> None:
        """get_annotation() can filter by category_name"""
        all_anns = page.get_annotation()
        if all_anns:
            first_category = all_anns[0].category_name
            filtered = page.get_annotation(category_names=first_category)
            assert all(ann.category_name == first_category for ann in filtered)

    def test_get_annotation_filters_by_annotation_id(self, page: Page) -> None:
        """get_annotation() can filter by annotation_id"""
        all_anns = page.get_annotation()
        if all_anns:
            target_id = all_anns[0].annotation_id
            filtered = page.get_annotation(annotation_ids=target_id)
            assert len(filtered) <= 1
            if filtered:
                assert filtered[0].annotation_id == target_id

    def test_get_annotation_filters_by_annotation_ids_list(self, page: Page) -> None:
        """get_annotation() can filter by list of annotation_ids"""
        all_anns = page.get_annotation()
        if len(all_anns) >= 2:
            target_ids = [all_anns[0].annotation_id, all_anns[1].annotation_id]
            filtered = page.get_annotation(annotation_ids=target_ids)
            assert all(ann.annotation_id in target_ids for ann in filtered)

    def test_layouts_are_floating_text_blocks(self, page: Page) -> None:
        """layouts are filtered by floating_text_block_categories"""
        layouts = page.layouts
        for layout in layouts:
            assert layout.category_name in page.floating_text_block_categories

    def test_words_are_text_containers(self, page: Page) -> None:
        """words are filtered by text_container"""
        words = page.words
        for word in words:
            assert word.category_name == page.text_container

    def test_residual_layouts_filtered_correctly(self, page: Page) -> None:
        """residual_layouts are filtered by residual_text_block_categories"""
        residual = page.residual_layouts
        for res in residual:
            assert res.category_name in page.residual_text_block_categories
