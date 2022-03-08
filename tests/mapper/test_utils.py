# -*- coding: utf-8 -*-
# File: test_utils.py

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
Testing the module mapper.utils
"""
from typing import Dict, List, Union
from unittest.mock import MagicMock

import pytest

from deepdoctection.mapper import DefaultMapper, LabelSummarizer


class TestDefaultMapper:  # pylint: disable=R0903
    """
    Testing Class methods of DefaultMapper
    """

    @staticmethod
    def test_func_is_called_with_default_arguments() -> None:
        """
        Function in DefaultMapper is called with first argument of default_mapper meth: __call__ argument and all other
        arguments from DefaultMapper attributes.
        """

        # Arrange
        test_mapper = DefaultMapper(MagicMock(), "foo", "bak", baz="foo_bak")

        # Act
        test_mapper(dp="input")

        # Assert
        test_mapper.func.assert_called_with("input", "foo", "bak", baz="foo_bak")  # type: ignore


class TestLabelSummarizer:  # pylint: disable=R0903
    """
    Testing Class methods of LabelSummarizer
    """

    @staticmethod
    @pytest.mark.parametrize(
        "categories, cat_ids, summary",
        [
            ({"1": "FOO", "2": "BAK", "3": "BAZ"}, ["1", "3", "2", "2", "3"], {"1": 1, "2": 2, "3": 2}),
            ({"1": "FOO", "2": "BAK"}, ["1", "2", ["1", "1", "2"], "1", "2", ["1", "1"]], {"1": 6, "2": 3}),
            ({"1": "FOO", "2": "BAK", "3": "BAZ"}, [1, 3, 2, 2, "3", 1, 1, 1, 1, "1"], {"1": 6, "2": 2, "3": 2}),
        ],
    )
    def test_categories_are_correctly_summarized(
        categories: Dict[str, str], cat_ids: List[Union[List[Union[str, int]], str]], summary: Dict[str, int]
    ) -> None:
        """
        Testing Summarizer input with various dumped category id representations.
        """
        # Arrange
        summarizer = LabelSummarizer(categories)

        # Act
        for element in cat_ids:
            summarizer.dump(element)

        # Assert
        assert summarizer.get_summary() == summary
