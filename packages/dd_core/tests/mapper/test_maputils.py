# -*- coding: utf-8 -*-
# File: test_maputils.py

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

from typing import Union

import numpy as np
import pytest

from dd_core.mapper.maputils import (
    DefaultMapper,
    LabelSummarizer,
    MappingContextManager,
    curry,
    maybe_get_fake_score,
)
from dd_core.utils.object_types import ObjectTypes, get_type


def test_mapping_context_manager_suppresses_known_exception_and_logs(monkeypatch)-> None: # type: ignore
    recorded = {}

    def fake_warning(arg): # type: ignore
        # capture the logging record passed
        recorded["arg"] = arg

    monkeypatch.setattr("dd_core.mapper.maputils.logger.warning", fake_warning)

    with MappingContextManager(dp_name="dp1", filter_level="image") as m:
        # raising a known exception should be suppressed by the context manager
        raise KeyError("missing key")

    assert m.context_error is True
    assert "MappingContextManager error" in str(recorded["arg"])


def test_mapping_context_manager_propagates_unknown_exception()-> None:
    with pytest.raises(RuntimeError):
        with MappingContextManager():
            raise RuntimeError("boom")


def test_mapping_context_manager_no_exception_sets_flag_false()-> None:
    with MappingContextManager():
        pass
    m = MappingContextManager()
    with m:
        pass
    assert m.context_error is False


def test_default_mapper_and_curry_behavior()-> None:
    def my_map(dp, a, b=0): # type: ignore
        return (dp, a, b)

    dm = DefaultMapper(my_map, 2, b=3)
    assert dm("datum") == ("datum", 2, 3)

    @curry
    def decorated(dp, x, y): # type: ignore
        return (dp, x, y)

    wrapped = decorated(5, 6)
    assert isinstance(wrapped, DefaultMapper)
    assert wrapped("Z") == ("Z", 5, 6)


def test_maybe_get_fake_score_mocked(monkeypatch)-> None:  # type: ignore
    # monkeypatch the RNG in the module to return a known value
    monkeypatch.setattr("dd_core.mapper.maputils.np.random.uniform", lambda a, b, c: np.array([0.123456]))
    val = maybe_get_fake_score(True)
    assert isinstance(val, float)
    assert abs(val - 0.123456) < 1e-9

    assert maybe_get_fake_score(False) is None


def test_label_summarizer_dump_get_summary_and_print(monkeypatch)-> None:  # type: ignore
    # simple category objects with .value attribute expected by LabelSummarizer
    categories = {
        1: get_type("test_cat_1"),
        2: get_type("test_cat_2"),
    }
    summarizer = LabelSummarizer(categories)

    # dump single id and multiple ids
    summarizer.dump(1)
    summarizer.dump([2, 2, 2])

    summary = summarizer.get_summary()
    assert summary == {1: 1, 2: 3}

    # capture logger.info called by print_summary_histogram
    captured = {}

    def fake_info(arg): # type: ignore
        captured["arg"] = arg

    monkeypatch.setattr("dd_core.mapper.maputils.logger.info", fake_info)

    # should not raise and should call logger.info with a LoggingRecord containing the table text
    summarizer.print_summary_histogram(dd_logic=True)
    assert "Ground-Truth category distribution" in str(captured["arg"])


class TestLabelSummarizer:
    """
    Testing Class methods of LabelSummarizer
    """

    @staticmethod
    @pytest.mark.parametrize(
        "categories, cat_ids, summary",
        [
            ({1: "FOO", 2: "BAK", 3: "BAZ"}, [1, 3, 2, 2, 3], {1: 1, 2: 2, 3: 2}),
            ({1: "FOO", 2: "BAK"}, [1, 2, [1, 1, 2], 1, 2, [1, 1]], {1: 6, 2: 3}),
            ({1: "FOO", 2: "BAK", 3: "BAZ"}, [1, 3, 2, 2, 3, 1, 1, 1, 1, 1], {1: 6, 2: 2, 3: 2}),
        ],
    )
    def test_categories_are_correctly_summarized(
        categories: dict[int, ObjectTypes], cat_ids: list[Union[list[int], int]], summary: dict[int, int]
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
