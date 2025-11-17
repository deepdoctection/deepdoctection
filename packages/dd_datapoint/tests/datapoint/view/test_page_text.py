# -*- coding: utf-8 -*-
# File: test_page_text.py
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
Testing Page class text extraction methods
"""
from dd_datapoint.datapoint.view import Page, Text_

class TestPageText:

    """Test Page text extraction"""

    def test_text_returns_string(self, page: Page):
        """text property returns a string"""
        text = page.text
        assert isinstance(text, str)

    def test_text_underscore_returns_text_dataclass(self, page: Page):
        """text_ property returns Text_ dataclass"""
        text_ = page.text_
        assert isinstance(text_, Text_)

    def test_text_underscore_has_text_field(self, page: Page):
        """text_ has text field"""
        text_ = page.text_
        assert isinstance(text_.text, str)

    def test_text_underscore_has_words_field(self, page: Page):
        """text_ has words field"""
        text_ = page.text_
        assert isinstance(text_.words, list)

    def test_text_underscore_has_ann_ids_field(self, page: Page):
        """text_ has ann_ids field"""
        text_ = page.text_
        assert isinstance(text_.ann_ids, list)

    def test_text_underscore_as_dict_returns_dict(self, page: Page):
        """text_.as_dict() returns a dictionary"""
        text_ = page.text_
        as_dict = text_.as_dict()
        assert isinstance(as_dict, dict)

    def test_text_underscore_as_dict_has_required_keys(self, page: Page):
        """text_.as_dict() has required keys"""
        text_ = page.text_
        as_dict = text_.as_dict()
        expected_keys = {
            "text", "words", "ann_ids", "token_classes", 
            "token_class_ann_ids", "token_tags", "token_tag_ann_ids",
            "token_class_ids", "token_tag_ids"
        }
        assert expected_keys.issubset(as_dict.keys())
