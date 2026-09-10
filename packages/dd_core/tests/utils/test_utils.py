# -*- coding: utf-8 -*-
# File: test_utils.py

# Copyright 2026 Dr. Janis Meyer. All rights reserved.
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
Testing the record flattening helpers of the module utils.utils
"""

from collections import OrderedDict
from typing import Any

import pytest

from dd_core.utils.utils import as_string, flatten, is_leaf

BOOKING_GT = {
    "bookingDate": ["04.01"],
    "valueDate": ["04.01.2016"],
    "description": [["Zahlungseingang", "Michael", "Feick", "bekannt"]],
    "amount": ["197,66-"],
    "postings": [
        {"postingType": ["Umsatzsteuer"], "postingAmount": ["31,55-"]},
        {"postingType": ["Nettobetrag"], "postingAmount": ["166,11-"]},
    ],
}


class TestAsString:
    """Test as_string"""

    @staticmethod
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, ""),
            ("", ""),
            ([], ""),
            ((), ""),
            ([None, "", "   "], ""),
            ("plain", "plain"),
            ("  leading and trailing  ", "leading and trailing"),
            ("collapse\n\tinner   blanks", "collapse inner blanks"),
            (["DE95", "5135"], "DE95 5135"),
            (("DE95", "5135"), "DE95 5135"),
            (["DE95", ["5135", "0000"]], "DE95 5135 0000"),
            (["DE95", None, "", "5135"], "DE95 5135"),
            (7, "7"),
            (3.5, "3.5"),
            (True, "True"),
        ],
    )
    def test_as_string_returns_single_normalized_string(value: Any, expected: str) -> None:
        """Test as_string maps a leaf to exactly one whitespace normalized string"""
        assert as_string(value) == expected

    @staticmethod
    def test_as_string_keeps_a_token_list_as_one_unit() -> None:
        """Test as_string joins a token list instead of splitting it into several values"""

        # Act
        result = as_string(["DE95", "5135"])

        # Assert
        assert result == "DE95 5135"


class TestIsLeaf:
    """Test is_leaf"""

    @staticmethod
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("some string", True),
            (None, True),
            (7, True),
            ([], True),
            (["DE95", "5135"], True),
            (("DE95", "5135"), True),
            ([["DE95"], ["5135"]], True),
            ({}, False),
            ({"postingType": ["tax"]}, False),
            (OrderedDict(postingType=["tax"]), False),
            ([{"postingType": ["tax"]}], False),
            (({"postingType": ["tax"]},), False),
            (["a token", {"postingType": ["tax"]}], False),
        ],
    )
    def test_is_leaf_detects_arrays_of_records(value: Any, expected: bool) -> None:
        """Test is_leaf returns False exactly for mappings and sequences holding a mapping"""
        assert is_leaf(value) is expected


class TestFlatten:
    """Test flatten"""

    @staticmethod
    def test_flatten_joins_nested_paths_with_a_dot() -> None:
        """Test flatten builds dot separated field paths"""

        # Act
        result = flatten({"a": {"b": {"c": "value"}}})

        # Assert
        assert result == {"a.b.c": ["value"]}

    @staticmethod
    def test_flatten_discards_list_indices() -> None:
        """Test flatten collects records of the same path into one list, so the index is dropped"""

        # Act
        result = flatten(BOOKING_GT)

        # Assert
        assert result == {
            "bookingDate": ["04.01"],
            "valueDate": ["04.01.2016"],
            "description": ["Zahlungseingang Michael Feick bekannt"],
            "amount": ["197,66-"],
            "postings.postingType": ["Umsatzsteuer", "Nettobetrag"],
            "postings.postingAmount": ["31,55-", "166,11-"],
        }

    @staticmethod
    @pytest.mark.parametrize(
        "record",
        [
            {},
            [],
            None,
            "",
            {"valueDate": None, "postings": []},
            {"a": {"b": ["", "   ", None]}},
        ],
    )
    def test_flatten_discards_empty_leaves(record: Any) -> None:
        """Test flatten drops empty leaves so that empty on both sides is not a match"""
        assert not flatten(record)

    @staticmethod
    def test_flatten_unfolds_nested_arrays_of_records() -> None:
        """Test flatten descends into sequences of mappings on several levels"""

        # Arrange
        record = {
            "statements": [
                {"bookings": [{"amount": ["1,00-"]}, {"amount": ["2,00-"]}]},
                {"bookings": [{"amount": ["3,00-"]}]},
            ]
        }

        # Act
        result = flatten(record)

        # Assert
        assert result == {"statements.bookings.amount": ["1,00-", "2,00-", "3,00-"]}

    @staticmethod
    def test_flatten_treats_a_mixed_sequence_as_an_array_of_records() -> None:
        """Test flatten unfolds a sequence holding both records and tokens under the same path"""

        # Act
        result = flatten({"postings": ["loose token", {"postingType": ["tax"]}]})

        # Assert
        assert result == {"postings": ["loose token"], "postings.postingType": ["tax"]}

    @staticmethod
    def test_flatten_accepts_a_scalar_root() -> None:
        """Test flatten keys a leaf root with the empty path"""
        assert flatten("value") == {"": ["value"]}

    @staticmethod
    def test_flatten_accepts_any_mapping() -> None:
        """Test flatten unfolds mappings that are not plain dicts"""

        # Act
        result = flatten(OrderedDict(postings=[OrderedDict(postingType=["tax"])]))

        # Assert
        assert result == {"postings.postingType": ["tax"]}

    @staticmethod
    def test_flatten_stringifies_non_string_keys() -> None:
        """Test flatten builds a field path from keys that are not strings"""
        assert flatten({1: {2: "value"}}) == {"1.2": ["value"]}

    @staticmethod
    def test_flatten_returns_a_plain_dict() -> None:
        """Test flatten does not leak the defaultdict used internally"""

        # Act
        result = flatten(BOOKING_GT)

        # Assert
        assert type(result) is dict  # pylint: disable=C0123
        with pytest.raises(KeyError):
            _ = result["unknown.path"]

    @staticmethod
    def test_flatten_is_order_independent_for_sub_records() -> None:
        """Test flatten of reordered sub-records differs only in the ordering of the collected values"""

        # Arrange
        swapped = dict(BOOKING_GT, postings=list(reversed(BOOKING_GT["postings"])))  # type: ignore

        # Act
        result = flatten(swapped)
        expected = flatten(BOOKING_GT)

        # Assert
        assert result != expected
        assert {path: sorted(values) for path, values in result.items()} == {
            path: sorted(values) for path, values in expected.items()
        }
