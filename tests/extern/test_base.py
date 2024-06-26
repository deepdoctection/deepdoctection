# -*- coding: utf-8 -*-
# File: test_base.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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
Testing module extern.base
"""

from types import MappingProxyType

from pytest import mark

from deepdoctection.extern.base import ModelCategories, NerModelCategories
from deepdoctection.utils.settings import get_type


class TestModelCategories:
    """
    Test ModelCategories
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.init_categories = {1: "word", 2: "line", 3: "table", 4: "figure", 5: "header", 6: "footnote"}
        self.categories = ModelCategories(init_categories=self.init_categories)

    @mark.basic
    def test_get_categories(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType(
            {
                1: get_type("word"),
                2: get_type("line"),
                3: get_type("table"),
                4: get_type("figure"),
                5: get_type("header"),
                6: get_type("footnote"),
            }
        )
        assert categories == expected_categories

    @mark.basic
    def test_get_categories_name_as_keys(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.get_categories(name_as_key=False)

        # Assert
        expected_categories = MappingProxyType(
            {
                1: get_type("word"),
                2: get_type("line"),
                3: get_type("table"),
                4: get_type("figure"),
                5: get_type("header"),
                6: get_type("footnote"),
            }
        )
        assert categories == expected_categories

    @mark.basic
    def test_get_categories_as_tuple(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.get_categories(False)

        # Assert
        expected_categories = (
            get_type("word"),
            get_type("line"),
            get_type("table"),
            get_type("figure"),
            get_type("header"),
            get_type("footnote"),
        )
        assert categories == expected_categories

    @mark.basic
    def test_filter_categories(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        self.categories.filter_categories = (
            get_type("word"),
            get_type("header"),
        )
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType(
            {2: get_type("line"), 3: get_type("table"), 4: get_type("figure"), 6: get_type("footnote")}
        )
        assert categories == expected_categories

    @mark.basic
    def test_shift_category_ids(self) -> None:
        """
        Test ModelCategories
        """

        # Act
        categories = self.categories.shift_category_ids(-1)

        # Assert
        expected_categories = MappingProxyType(
            {
                0: get_type("word"),
                1: get_type("line"),
                2: get_type("table"),
                3: get_type("figure"),
                4: get_type("header"),
                5: get_type("footnote"),
            }
        )
        assert categories == expected_categories


class TestNerModelCategories:
    """TestNerModelCategories"""

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.categories_semantics = (
            "question",
            "answer",
        )
        self.categories_bio = (
            "B",
            "I",
        )

    def test_get_categories(self) -> None:
        """
        Test NerModelCategories
        """
        # Arrange
        self.categories = NerModelCategories(
            init_categories=None, categories_semantics=self.categories_semantics, categories_bio=self.categories_bio
        )

        # Act
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType(
            {
                1: get_type("B-answer"),
                2: get_type("B-question"),
                3: get_type("I-answer"),
                4: get_type("I-question"),
            }
        )
        assert categories == expected_categories

    def test_categories_does_not_overwrite_consolidated_categories(self) -> None:
        """
        Test NerModelCategories
        """
        # Arrange
        self.categories = NerModelCategories(
            init_categories={1: get_type("B-answer"), 2: get_type("B-question")},
            categories_semantics=self.categories_semantics,
            categories_bio=self.categories_bio,
        )

        # Act
        categories = self.categories.get_categories()

        # Assert
        expected_categories = MappingProxyType({1: get_type("B-answer"), 2: get_type("B-question")})
        assert categories == expected_categories
