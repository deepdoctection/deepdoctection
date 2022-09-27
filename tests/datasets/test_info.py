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

from typing import Tuple

import pytest

from deepdoctection.datasets import DatasetCategories
from deepdoctection.datasets.info import get_merged_categories
from deepdoctection.utils.settings import get_type

from ..data import TestType


class TestDatasetCategories:
    """
    Testing DatasetCategories
    """

    @staticmethod
    def setup() -> DatasetCategories:
        """
        Arrange testing setup
        """
        categories = [get_type("FOO"), get_type("BAK"), get_type("BAZ")]
        sub_categories = {
            get_type("BAK"): {
                get_type("sub"): [get_type("BAK_11"), get_type("BAK_12")],
                get_type("sub_2"): [get_type("BAK_21"), get_type("BAK_22")],
            },
            get_type("FOO"): {get_type("cat"): [get_type("FOO_1"), get_type("FOO_2"), get_type("FOO_3")]},
        }

        return DatasetCategories(init_categories=categories, init_sub_categories=sub_categories)

    @staticmethod
    @pytest.mark.basic
    def test_set_cat_to_subcat_and_check_categories_case_1() -> None:
        """
        Categories are dumped and annotation ids are correctly assigned and meth: is_cat_to_sub_cat works properly.
        """

        # Arrange
        cats = TestDatasetCategories.setup()

        # Assert
        assert cats.get_categories() == {"1": TestType.FOO, "2": TestType.BAK, "3": TestType.BAZ}
        assert not cats.is_cat_to_sub_cat()

        # Act
        cats.set_cat_to_sub_cat({TestType.BAK: TestType.sub_2})

        # Assert
        assert cats.get_categories(as_dict=False, init=True) == [TestType.FOO, TestType.BAK, TestType.BAZ]
        assert cats.get_categories(as_dict=False) == [TestType.FOO, TestType.BAK_21, TestType.BAK_22, TestType.BAZ]
        assert cats.is_cat_to_sub_cat()

    @staticmethod
    @pytest.mark.basic
    def test_set_cat_to_subcat_and_check_categories_case_2() -> None:
        """
        Categories are dumped and annotation ids are correctly assigned and meth: is_cat_to_sub_cat works properly.
        """

        # Arrange
        cats = TestDatasetCategories.setup()

        # Act
        cats.set_cat_to_sub_cat({TestType.BAK: TestType.sub, TestType.FOO: TestType.cat})

        # Assert
        assert cats.get_categories(name_as_key=True) == {
            get_type("FOO_1"): "1",
            get_type("FOO_2"): "2",
            get_type("FOO_3"): "3",
            get_type("BAK_11"): "4",
            get_type("BAK_12"): "5",
            get_type("BAZ"): "6",
        }

    @staticmethod
    @pytest.mark.basic
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
        assert cats.get_categories(name_as_key=True, filtered=True) == {
            TestType.FOO_1: "1",
            TestType.BAZ: "3",
            TestType.BAK_11: "2",
        }
        assert cats.is_filtered()

    @staticmethod
    @pytest.mark.basic
    def test_check_sub_categories() -> None:
        """
        get_sub_categories works as expected
        """

        # Arrange
        cats = TestDatasetCategories.setup()

        # Assert
        assert cats.get_sub_categories() == {TestType.BAK: [TestType.sub, TestType.sub_2], TestType.FOO: [TestType.cat]}

    @staticmethod
    @pytest.mark.basic
    def test_set_sub_categories_and_check_sub_categories() -> None:
        """
        when categories are replaced with sub categories then get_sub_categories works as expected
        """

        # Arrange
        cats = TestDatasetCategories.setup()
        cats.set_cat_to_sub_cat({"FOO": "cat"})

        # Act
        assert cats.get_sub_categories(categories=["FOO_1", "FOO_2", "FOO_3"]) == {
            TestType.FOO_1: [],
            TestType.FOO_2: [],
            TestType.FOO_3: [],
        }


class TestMergeDatasetCategories:
    """
    Testing get_merged_categories
    """

    @staticmethod
    def setup() -> Tuple[DatasetCategories, DatasetCategories]:
        """
        Arrange testing setup
        """
        init_categories_1 = [get_type("FOO"), get_type("BAK")]
        init_categories_2 = [get_type("FOO"), get_type("BAZ")]

        sub_categories_1 = {
            get_type("FOO"): {
                get_type("FOO_1"): [get_type("1"), get_type("2")],
                get_type("FOO_2"): [get_type("3"), get_type("4")],
            }
        }
        sub_categories_2 = {
            get_type("FOO"): {get_type("FOO_1"): [get_type("1"), get_type("3")]},
            get_type("BAK"): {get_type("BAK_1"): [get_type("4"), get_type("5")]},
        }

        return (
            DatasetCategories(init_categories=init_categories_1, init_sub_categories=sub_categories_1),
            DatasetCategories(init_categories=init_categories_2, init_sub_categories=sub_categories_2),
        )

    @staticmethod
    @pytest.mark.basic
    def test_merge_categories_returns_union_categories_and_sub_categories() -> None:
        """
        Merge categories returns union of categories of datasets
        """

        # Arrange
        cat_1, cat_2 = TestMergeDatasetCategories.setup()

        # Act
        merge = get_merged_categories(cat_1, cat_2)

        # Assert
        assert merge.get_categories(init=True, as_dict=False) == [TestType.FOO, TestType.BAK, TestType.BAZ]
        assert merge.get_sub_categories() == {TestType.FOO: [TestType.FOO_1]}

    @staticmethod
    @pytest.mark.basic
    def test_merge_categories_updates_categories_correctly() -> None:
        """
        Merge categories updates categories when categories of datasets are being updated
        """

        # Arrange
        cat_1, cat_2 = TestMergeDatasetCategories.setup()
        cat_1.set_cat_to_sub_cat({"FOO": "FOO_1"})
        cat_2.set_cat_to_sub_cat({"FOO": "FOO_1"})

        # Act
        merge = get_merged_categories(cat_1, cat_2)

        # Assert
        assert merge.get_categories(as_dict=False, init=True) == [TestType.FOO, TestType.BAK, TestType.BAZ]
        assert merge.get_categories(as_dict=False) == [
            TestType.one,
            TestType.two,
            TestType.BAK,
            TestType.three,
            TestType.BAZ,
        ]

    @staticmethod
    @pytest.mark.basic
    def test_merge_categories_updates_and_filters_categories_correctly() -> None:
        """
        Merge categories returns updates and filtered categories correctly
        """

        # Arrange
        cat_1, cat_2 = TestMergeDatasetCategories.setup()
        cat_1.set_cat_to_sub_cat({"FOO": "FOO_1"})
        cat_1.filter_categories(categories=["2"])

        cat_2.set_cat_to_sub_cat({"FOO": "FOO_1"})

        # Act
        merge = get_merged_categories(cat_1, cat_2)

        # Assert
        assert merge.get_categories(as_dict=False, init=False) == [
            TestType.one,
            TestType.two,
            TestType.BAK,
            TestType.three,
            TestType.BAZ,
        ]
        assert merge.get_categories(as_dict=False, init=False, filtered=True) == [
            TestType.two,
            TestType.one,
            TestType.three,
            TestType.BAZ,
        ]

    @staticmethod
    @pytest.mark.basic
    def test_merge_categories_cannot_update_or_filter() -> None:
        """
        Calling :meth::`filter_categories` or :meth::`set_cat_to_sub_cat` is not allowed
        """

        # Arrange
        cat_1, cat_2 = TestMergeDatasetCategories.setup()
        merge = get_merged_categories(cat_1, cat_2)

        # Act & Assert
        with pytest.raises(PermissionError):
            merge.filter_categories(categories="BAZ")
            merge.set_cat_to_sub_cat({"FOO": "FOO_1"})
