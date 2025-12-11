# -*- coding: utf-8 -*-
# File: xxx.py

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


import pytest

from dd_core.utils.object_types import (
    BioTag,
    CellType,
    LayoutType,
    TokenClasses,
    TokenClassWithTag,
    WordType,
)
from dd_datasets.info import DatasetCategories, get_merged_categories
from dd_datasets.instances import pubtabnet as pub_mod
from dd_datasets.instances import xfund as xfund_mod


@pytest.fixture
def pubcat() -> DatasetCategories:
    return DatasetCategories(
        init_categories=pub_mod._INIT_CATEGORIES,  # noqa: SLF001
        init_sub_categories=pub_mod._SUB_CATEGORIES,  # noqa: SLF001
    )


@pytest.fixture
def xfundcat() -> DatasetCategories:
    return DatasetCategories(
        init_categories=xfund_mod._INIT_CATEGORIES,  # noqa: SLF001
        init_sub_categories=xfund_mod._SUB_CATEGORIES,  # noqa: SLF001
    )


def test_init_get_categories_list_pub(pubcat: DatasetCategories) -> None:
    assert pubcat.get_categories(as_dict=False, init=True) == list(pub_mod._INIT_CATEGORIES)  # noqa: SLF001


def test_init_get_categories_dict_name_as_key_pub(pubcat: DatasetCategories) -> None:
    m = pubcat.get_categories(name_as_key=True, init=True)
    assert set(m.keys()) == set(pub_mod._INIT_CATEGORIES)  # noqa: SLF001
    assert set(m.values()) == set(range(1, len(pub_mod._INIT_CATEGORIES) + 1))  # noqa: SLF001


def test_init_get_categories_dict_idx_as_key_pub(pubcat: DatasetCategories) -> None:
    m = pubcat.get_categories(name_as_key=False, init=True)
    assert set(m.keys()) == set(range(1, len(pub_mod._INIT_CATEGORIES) + 1))  # noqa: SLF001
    assert set(m.values()) == set(pub_mod._INIT_CATEGORIES)  # noqa: SLF001


def test_get_categories_default_equals_init_before_changes(pubcat: DatasetCategories) -> None:
    assert pubcat.get_categories(as_dict=False) == list(pub_mod._INIT_CATEGORIES)  # noqa: SLF001


def test_filter_categories_keeps_subset_and_order_pub(pubcat: DatasetCategories) -> None:
    pubcat.filter_categories([LayoutType.WORD, LayoutType.TABLE])
    filtered = pubcat.get_categories(as_dict=False, filtered=True)
    assert filtered == [LayoutType.TABLE, LayoutType.WORD]


def test_is_filtered_flag_pub(pubcat: DatasetCategories) -> None:
    assert pubcat.is_filtered() is False
    pubcat.filter_categories(LayoutType.WORD)
    assert pubcat.is_filtered() is True


def test_filter_only_once_raises(pubcat: DatasetCategories) -> None:
    pubcat.filter_categories([LayoutType.WORD])
    with pytest.raises(Exception):
        pubcat.filter_categories([LayoutType.TABLE])


def test_set_cat_to_sub_cat_replace_cell_pub(pubcat: DatasetCategories) -> None:
    pubcat.set_cat_to_sub_cat({LayoutType.CELL: CellType.COLUMN_HEADER})
    cats = pubcat.get_categories(as_dict=False)
    assert LayoutType.CELL not in cats
    assert CellType.COLUMN_HEADER in cats and CellType.BODY in cats


def test_is_cat_to_sub_cat_flag_pub(pubcat: DatasetCategories) -> None:
    assert pubcat.is_cat_to_sub_cat() is False
    pubcat.set_cat_to_sub_cat({LayoutType.CELL: CellType.COLUMN_HEADER})
    assert pubcat.is_cat_to_sub_cat() is True


def test_set_cat_to_sub_cat_only_once_raises(pubcat: DatasetCategories) -> None:
    pubcat.set_cat_to_sub_cat({LayoutType.CELL: CellType.COLUMN_HEADER})
    with pytest.raises(Exception):
        pubcat.set_cat_to_sub_cat({LayoutType.WORD: WordType.CHARACTERS})


def test_pub_word_sub_keys(pubcat: DatasetCategories) -> None:
    sub = pubcat.get_sub_categories(categories=LayoutType.WORD, keys=True)
    assert sub[LayoutType.WORD] == [WordType.CHARACTERS]


def test_xfund_sub_keys_word_has_expected(xfundcat: DatasetCategories) -> None:
    sub = xfundcat.get_sub_categories(categories=LayoutType.WORD, keys=True)
    assert set(sub[LayoutType.WORD]) == {WordType.TOKEN_CLASS, WordType.TAG, WordType.TOKEN_TAG}


def test_xfund_sub_values_token_class_indices_start_at_1(xfundcat: DatasetCategories) -> None:
    res = xfundcat.get_sub_categories(
        categories=LayoutType.WORD,
        sub_categories={LayoutType.WORD: [WordType.TOKEN_CLASS]},
        keys=False,
        values_as_dict=True,
        name_as_key=False,
    )
    idx_map = res[LayoutType.WORD][WordType.TOKEN_CLASS]
    assert set(idx_map.keys()) == {1, 2, 3, 4}
    assert set(idx_map.values()) == {
        TokenClasses.OTHER,
        TokenClasses.QUESTION,
        TokenClasses.ANSWER,
        TokenClasses.HEADER,
    }


def test_xfund_sub_values_name_as_key_true(xfundcat: DatasetCategories) -> None:
    res = xfundcat.get_sub_categories(
        categories=LayoutType.WORD,
        sub_categories={LayoutType.WORD: [WordType.TAG]},
        keys=False,
        values_as_dict=True,
        name_as_key=True,
    )
    tag_map = res[LayoutType.WORD][WordType.TAG]
    assert isinstance(tag_map[BioTag.OUTSIDE], int)


def test_xfund_replace_word_with_token_tag_then_no_further_subcats(xfundcat: DatasetCategories) -> None:
    xfundcat.set_cat_to_sub_cat({LayoutType.WORD: WordType.TOKEN_TAG})
    cats = xfundcat.get_categories(as_dict=False)
    assert TokenClassWithTag.B_ANSWER in cats
    empty = xfundcat.get_sub_categories(categories=TokenClassWithTag.B_ANSWER, keys=True)
    assert empty == {TokenClassWithTag.B_ANSWER: []}


def test_get_categories_as_dict_after_replacement_indices_sequential(pubcat: DatasetCategories) -> None:
    pubcat.set_cat_to_sub_cat({LayoutType.CELL: CellType.COLUMN_HEADER})
    m = pubcat.get_categories(name_as_key=True)
    assert set(m.values()) == set(range(1, len(m) + 1))


def test_filtered_dict_idx_as_key_is_contiguous(pubcat: DatasetCategories) -> None:
    pubcat.filter_categories([LayoutType.TABLE, LayoutType.WORD])
    m = pubcat.get_categories(name_as_key=False, filtered=True)
    assert set(m.keys()) == set(range(1, 2 + 1))


def test_filtered_and_unfiltered_lengths(pubcat: DatasetCategories) -> None:
    pubcat.filter_categories([LayoutType.TABLE])
    all_len = len(pubcat.get_categories(as_dict=False, filtered=False))
    filt_len = len(pubcat.get_categories(as_dict=False, filtered=True))
    assert filt_len == 1 and filt_len < all_len


def test_get_merged_categories_union_and_sub_intersection(
    pubcat: DatasetCategories, xfundcat: DatasetCategories
) -> None:
    merged = get_merged_categories(pubcat, xfundcat)
    cats = merged.get_categories(as_dict=False)
    assert LayoutType.WORD in cats and LayoutType.TABLE in cats
    sub = merged.get_sub_categories(categories=LayoutType.WORD, keys=True)
    assert sub.get(LayoutType.WORD, []) in ([], None)


def test_merged_categories_locked_against_updates(pubcat: DatasetCategories, xfundcat: DatasetCategories) -> None:
    merged = get_merged_categories(pubcat, xfundcat)
    with pytest.raises(RuntimeWarning):
        merged.filter_categories([LayoutType.WORD])
    with pytest.raises(RuntimeWarning):
        merged.set_cat_to_sub_cat({LayoutType.CELL: CellType.COLUMN_HEADER})


def test_xfund_multiple_categories_keys_list(xfundcat: DatasetCategories) -> None:
    sub = xfundcat.get_sub_categories(categories=[LayoutType.WORD, LayoutType.TEXT], keys=True)
    assert LayoutType.WORD in sub and LayoutType.TEXT in sub


def test_values_as_list_not_dict_pub(pubcat: DatasetCategories) -> None:
    res = pubcat.get_sub_categories(
        categories=LayoutType.WORD,
        sub_categories={LayoutType.WORD: [WordType.CHARACTERS]},
        keys=False,
        values_as_dict=False,
    )
    assert res[LayoutType.WORD][WordType.CHARACTERS] == [WordType.CHARACTERS]


def test_invalid_init_sub_categories_type_raises() -> None:
    with pytest.raises(AssertionError):
        DatasetCategories(
            init_categories=[LayoutType.WORD], init_sub_categories={LayoutType.WORD: [WordType.CHARACTERS]}
        )
