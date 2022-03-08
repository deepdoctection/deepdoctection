# -*- coding: utf-8 -*-
# File: test_info.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
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
Testing the module datasets.info
"""
from deepdoctection.datasets import DatasetCategories


class TestDatasetCategories:
    """
    Testing DatasetCategories
    """

    @staticmethod
    def setup() -> DatasetCategories:
        """
        Arrange testing setup
        """
        categories = ["FOO", "BAK", "BAZ"]
        sub_categories = {
            "BAK": {"sub": ["BAK_11", "BAK_12"], "sub_2": ["BAK_21", "BAK_22"]},
            "FOO": {"cat": ["FOO_1", "FOO_2", "FOO_3"]},
        }

        return DatasetCategories(init_categories=categories, init_sub_categories=sub_categories)

    @staticmethod
    def test_set_cat_to_subcat_and_check_categories() -> None:
        """
        Categories are dumped and annotation ids are correctly assigned and meth: is_cat_to_sub_cat works properly.
        """

        # Arrange
        cats = TestDatasetCategories.setup()

        # Assert
        assert cats.get_categories() == {"1": "FOO", "2": "BAK", "3": "BAZ"}
        assert not cats.is_cat_to_sub_cat()

        # Act
        cats.set_cat_to_sub_cat({"BAK": "sub_2"})

        # Assert
        assert cats.get_categories(as_dict=False, init=True) == ["FOO", "BAK", "BAZ"]
        assert cats.get_categories(as_dict=False) == ["FOO", "BAK_21", "BAK_22", "BAZ"]
        assert cats.is_cat_to_sub_cat()

        # Act
        cats.set_cat_to_sub_cat({"BAK": "sub", "FOO": "cat"})

        # Assert
        assert cats.get_categories(name_as_key=True) == {
            "FOO_1": "1",
            "FOO_2": "2",
            "FOO_3": "3",
            "BAK_11": "4",
            "BAK_12": "5",
            "BAZ": "6",
        }

    @staticmethod
    def test_filter_and_check_categories() -> None:
        """
        Categories are filtered and meth: is_filtered works properly
        """

        # Arrange
        cats = TestDatasetCategories.setup()

        # Assert
        assert not cats.is_filtered()

        # Act
        cats.set_cat_to_sub_cat({"BAK": "sub", "FOO": "cat"})
        cats.filter_categories(categories=["FOO_1", "BAZ", "BAK_11"])

        # Assert
        assert cats.get_categories(name_as_key=True, filtered=True) == {"FOO_1": "1", "BAZ": "3", "BAK_11": "2"}
        assert cats.is_filtered()

    @staticmethod
    def test_check_sub_categories() -> None:
        """
        get_sub_categories works as expected
        """

        # Arrange
        cats = TestDatasetCategories.setup()

        # Assert
        assert cats.get_sub_categories() == {"BAK": ["sub", "sub_2"], "FOO": ["cat"]}

    @staticmethod
    def test_set_sub_categories_and_check_sub_categories() -> None:
        """
        when categories are replaced with sub categories then get_sub_categories works as expected
        """

        # Arrange
        cats = TestDatasetCategories.setup()
        cats.set_cat_to_sub_cat({"FOO": "cat"})

        # Act
        assert cats.get_sub_categories(categories=["FOO_1", "FOO_2", "FOO_3"]) == {
            "FOO_1": [],
            "FOO_2": [],
            "FOO_3": [],
        }
