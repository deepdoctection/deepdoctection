# -*- coding: utf-8 -*-
# File: info.py

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
General meta-data or categories
"""

from copy import copy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Literal, Mapping, Optional, Sequence, Union, no_type_check, overload

from ..utils.settings import DatasetType, ObjectTypes, TypeOrStr, get_type
from ..utils.utils import call_only_once

__all__ = ["DatasetInfo", "DatasetCategories", "get_merged_categories"]


@overload
def _get_dict(l: Sequence[ObjectTypes], name_as_key: Literal[True], starts_with: int = ...) -> dict[ObjectTypes, int]:
    ...


@overload
def _get_dict(l: Sequence[ObjectTypes], name_as_key: Literal[False], starts_with: int = ...) -> dict[int, ObjectTypes]:
    ...


@overload
def _get_dict(
    l: Sequence[ObjectTypes], name_as_key: bool, starts_with: int = ...
) -> Union[dict[ObjectTypes, int], dict[int, ObjectTypes]]:
    ...


def _get_dict(
    l: Sequence[ObjectTypes], name_as_key: bool, starts_with: int = 1
) -> Union[dict[ObjectTypes, int], dict[int, ObjectTypes]]:
    """
    Converts a list into a dict, where keys/values are the list indices.

    Args:
        l: A list of categories
        name_as_key: Whether to return the dict with category names as key (`True`)
        starts_with: index count start

    Returns:
        A dictionary of list indices/list elements.
    """
    if name_as_key:
        return {v: k for k, v in enumerate(l, starts_with)}
    return dict(enumerate(l, starts_with))


@dataclass
class DatasetInfo:
    """
    `DatasetInfo` is a simple dataclass that stores some meta-data information about a dataset.

     Attributes:
        name: Name of the dataset. Using the name you can retrieve the dataset from the `registry.DatasetRegistry`.
        description: Short description of the dataset.
        license: License to the dataset.
        url: url, where the dataset can be downloaded from.
        splits: A `dict` of splits. The value must store the relative path, where the split can be found.
        type: The type describes whether this is a dataset for object detection (pass 'OBJECT_DETECTION'),
              sequence classification (pass 'SEQUENCE_CLASSIFICATION') or token classification ('TOKEN_CLASSIFICATION').
              Optionally, pass `None`.
    """

    name: str
    short_description: str = field(default="")
    description: str = field(default="")
    license: str = field(default="")
    url: Union[str, Sequence[str]] = field(default="")
    splits: Mapping[str, str] = field(default_factory=dict)
    type: DatasetType = field(default=DatasetType.DEFAULT)

    def get_split(self, key: str) -> str:
        """
        Get the split directory by its key (if it exists).

        Args:
            key: The key to a split (i.e. `train`, `val`,`test`)

        Returns:
            The local directory path to the split. An empty string if the key doesn't exist.
        """

        return self.splits[key]


@dataclass
class DatasetCategories:
    """
    Categories and their sub-categories are managed in this separate class. Since categories can be filtered or
    sub-categories can be swapped with categories, the list of all categories must be adapted at the same time
    after replacement/filtering. DatasetCategories manages these transformations. The class is also responsible
    for the index/category name relationship and guarantees that a sequence of natural numbers for the categories
    is always returned as the category-id even after replacing and/or filtering.

    Attributes:
        init_categories: A list of `category_name`s. The list must include all categories that can occur within the
                         annotations.
        init_sub_categories: A dict of categories/sub-categories. Each sub-category that can appear in the
                             annotations in combination with a category must be listed.

    Example:

        An annotation file hast the category/sub-category combinations for three datapoints:

        ```python
        (cat1,s1),(cat1,s2), (cat2,s2).
        ```

        You must list `init_categories`, `init_sub_categories` as follows:

        ```python
        init_categories = [cat1,cat2]
        init_sub_categories = {cat1: [s1,s2],cat2: [s2]}
        ```

    Use `filter_categories` or `set_cat_to_sub_cat` to filter or swap categories with sub-categories.
    """

    init_categories: Sequence[ObjectTypes] = field(default_factory=list)
    init_sub_categories: Mapping[ObjectTypes, Mapping[ObjectTypes, Sequence[ObjectTypes]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._categories_update = self.init_categories
        self._cat_to_sub_cat: Optional[Mapping[ObjectTypes, ObjectTypes]] = None
        self._allow_update = True
        self._init_sanity_check()

    @overload
    def get_categories(
        self, *, name_as_key: Literal[True], init: bool = ..., filtered: bool = ...
    ) -> Mapping[ObjectTypes, int]:
        ...

    @overload
    def get_categories(
        self, *, name_as_key: Literal[False] = ..., init: bool = ..., filtered: bool = ...
    ) -> Mapping[int, ObjectTypes]:
        ...

    @overload
    def get_categories(
        self, as_dict: Literal[False], name_as_key: bool = False, init: bool = False, filtered: bool = False
    ) -> Sequence[ObjectTypes]:
        ...

    @overload
    def get_categories(
        self, as_dict: Literal[True] = ..., name_as_key: bool = False, init: bool = False, filtered: bool = False
    ) -> Union[Mapping[ObjectTypes, int], Mapping[int, ObjectTypes]]:
        ...

    def get_categories(
        self, as_dict: bool = True, name_as_key: bool = False, init: bool = False, filtered: bool = False
    ) -> Union[Sequence[ObjectTypes], Mapping[ObjectTypes, int], Mapping[int, ObjectTypes]]:
        """
        Get categories of a dataset. The returned value also respects modifications of the inventory like filtered
        categories of replaced categories with sub categories. However, you must correctly pass arguments to return the
        state you want.

        Args:
            as_dict: Will pass a dict if set to 'True' otherwise a list.
            name_as_key: Categories are stored as key/value pair in a dict with integers as keys. `name_as_key` set to
                         `False` will swap keys and values.
            init: If set to `True` it will return the list/dict of categories as initially provided. Manipulations
                  due to replacing/filtering will not be regarded.
            filtered: If set to `True` will return an unfiltered list of all categories. If a replacing has been
                      invoked selected sub categories will be returned.

        Returns:
            A dict of index/category names (or the other way around) or a list of category names.
        """
        if init:
            if as_dict:
                return _get_dict(self.init_categories, name_as_key)
            return list(copy(self.init_categories))
        if filtered:
            if as_dict:
                if hasattr(self, "_categories_filter_update"):
                    return _get_dict(self._categories_filter_update, name_as_key)
                return _get_dict(self._categories_update, name_as_key)
            if hasattr(self, "_categories_filter_update"):
                return self._categories_filter_update
            return list(copy(self._categories_update))
        if as_dict:
            return _get_dict(self._categories_update, name_as_key)
        return list(copy(self._categories_update))

    def get_sub_categories(
        self,
        categories: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
        sub_categories: Optional[Mapping[TypeOrStr, Union[TypeOrStr, Sequence[TypeOrStr]]]] = None,
        keys: bool = True,
        values_as_dict: bool = True,
        name_as_key: bool = False,
    ) -> Mapping[ObjectTypes, Any]:
        """
        Returns a dict of list with a category name and their sub categories.

        Args:
            categories: A single category or list of category names
            sub_categories: A mapping of categories to sub category keys on which the result should be filtered. Only
                               relevant, if `keys=False`
            keys: Will only pass keys if set to `True`.
            values_as_dict: Will generate a dict with indices and sub category value names if set to `True`.
            name_as_key: sub category values are stored as key/value pair in a dict with integers as keys.
                            name_as_key set to `False` will swap keys and values.

        Returns:
            Dict with all selected categories.
        """
        _categories: Sequence[ObjectTypes]
        if isinstance(categories, (ObjectTypes, str)):
            _categories = [get_type(categories)]
        elif isinstance(categories, list):
            _categories = [get_type(category) for category in categories]
        else:
            _categories = self.get_categories(as_dict=False, filtered=True)
            if _categories is None:
                _categories = []
        if sub_categories is None:
            sub_categories = {}

        sub_cat: dict[ObjectTypes, Union[ObjectTypes, list[ObjectTypes]]] = {}
        for cat in _categories:
            assert cat in self.get_categories(  # pylint: disable=E1135
                as_dict=False, filtered=True
            ), f"{cat} not in categories. Maybe it has been replaced with sub category"
            sub_cat_dict = self.init_sub_categories.get(cat)
            if sub_cat_dict is None:
                if self._cat_to_sub_cat:
                    for key, val in self._cat_to_sub_cat.items():
                        possible_sub_cats = self.init_sub_categories[key]
                        for pos in possible_sub_cats:
                            if pos == val:
                                sub_cat_dict_key = self.init_sub_categories[key]
                                sub_cat_list = sub_cat_dict_key.get(cat)
                                if sub_cat_list is None:
                                    sub_cat[cat] = []
                                else:
                                    sub_cat[cat] = list(copy(sub_cat_list))
            else:
                sub_cat[cat] = list(sub_cat_dict.keys())
        if not keys:
            sub_cat_values = {}
            for category, value in sub_cat.items():
                if category not in sub_categories:
                    continue
                sub_cat_tmp: dict[str, Union[dict[int, ObjectTypes], dict[ObjectTypes, int], Sequence[str]]] = {}
                sub_categories_list: Union[
                    ObjectTypes, str, list[Sequence[Union[ObjectTypes, str]]], Sequence[Union[ObjectTypes, str]]
                ]
                if isinstance(sub_categories[category], ObjectTypes):
                    sub_categories_list = [sub_categories[category]]
                else:
                    sub_categories_list = sub_categories[category]
                for sub_cat_key in value:
                    if sub_cat_key not in sub_categories_list:
                        continue
                    if values_as_dict:
                        if not name_as_key:
                            sub_cat_tmp[sub_cat_key] = dict(
                                enumerate(self.init_sub_categories[category][get_type(sub_cat_key)], 1)
                            )
                        else:
                            sub_cat_tmp[sub_cat_key] = {
                                v: k for k, v in enumerate(self.init_sub_categories[category][get_type(sub_cat_key)], 1)
                            }
                    else:
                        sub_cat_tmp[sub_cat_key] = self.init_sub_categories[category][get_type(sub_cat_key)]
                sub_cat_values[category] = sub_cat_tmp
            return sub_cat_values

        return sub_cat

    @call_only_once
    def set_cat_to_sub_cat(self, cat_to_sub_cat: dict[TypeOrStr, TypeOrStr]) -> None:
        """
        Change category representation if sub-categories are available. Pass a dictionary of the main category
        and the requested sub-category. This will change the dictionary of categories and the category names
        and category ids of the image annotation in the dataflow. Using more than one sub category per category will
        not make sense.

        This method can only be called once per object. Re-setting or further replacing of categories would make the
        code messy and is therefore not allowed.

        Example:
            ```python
            cat_to_sub_cat={cat1: sub_cat1}
            ```

            will replace cat1 with sub_cat1 as category. This will also be respected when returning datapoints.

        Args:
            cat_to_sub_cat: A dict of pairs of category/sub-category. Note that the combination must be available
                            according to the initial settings.
        """

        _cat_to_sub_cat = {get_type(key): get_type(value) for key, value in cat_to_sub_cat.items()}
        if not self._allow_update:
            raise RuntimeWarning("Replacing categories with sub categories is not allowed")
        self._categories_update = self.init_categories
        categories = self.get_categories(name_as_key=True)
        cats_or_sub_cats = [
            self.init_sub_categories.get(cat, {cat: [cat]}).get(_cat_to_sub_cat.get(cat, cat), [cat])
            for cat in categories  # pylint: disable=E1133
        ]
        self._cat_to_sub_cat = _cat_to_sub_cat
        _categories_update_list = list(chain(*cats_or_sub_cats))

        # we must keep the order of _categories_update_list if the elements are unique
        if len(set(_categories_update_list)) != len(_categories_update_list):
            self._categories_update = list(set(_categories_update_list))
        else:
            self._categories_update = _categories_update_list

    @call_only_once
    def filter_categories(self, categories: Union[TypeOrStr, list[TypeOrStr]]) -> None:
        """
        Filter categories of a dataset. This will keep all the categories chosen and remove all others.
        This method can only be called once per object.

        Args:
            categories: A single `category_name` or a list of `category_name`s.
        """

        if not self._allow_update:
            raise RuntimeWarning("Filtering categories is not allowed")
        if isinstance(categories, (ObjectTypes, str)):
            categories = [get_type(categories)]
        else:
            categories = [get_type(category) for category in categories]

        self._categories_filter_update: Sequence[ObjectTypes] = [
            cat for cat in self._categories_update if cat in categories
        ]

    @property
    def cat_to_sub_cat(self) -> Optional[Mapping[ObjectTypes, ObjectTypes]]:
        """
        `cat_to_sub_cat`
        """
        return self._cat_to_sub_cat

    def is_cat_to_sub_cat(self) -> bool:
        """
        Returns:
            `True` if a category is replaced with sub categories
        """
        if self._cat_to_sub_cat is not None:
            return True
        return False

    def is_filtered(self) -> bool:
        """
        Returns:
            `True` if categories are filtered
        """
        if hasattr(self, "_categories_filter_update"):
            return True
        return False

    def _init_sanity_check(self) -> None:
        # all values of possible sub categories must be listed
        assert all(isinstance(val, dict) for val in self.init_sub_categories.values())


@no_type_check
def get_merged_categories(*categories: DatasetCategories) -> DatasetCategories:
    """
    Given a set of `DatasetCategories`, a `DatasetCategories` instance will be returned that summarize the category
    properties of merged dataset. This means it will save the union of all possible categories as its init categories.
    Regarding sub categories, only those will be accessible if and only if they are a sub category of a category for all
    merged datasets. E.g. if dataset A has category `foo` with sub category `foo`:`bak` and dataset B has category `foo`
    as well but no sub category than the merged dataset will have no sub categories at all. Whereas in a similar setting
    dataset B has sub category `foo`:`bak`, then `bak` will be an optional sub category for the merged dataset as well.

    Args:
        categories: A tuple/list of dataset categories

    Returns:
        An instance of `DatasetCategories` to be used as `DatasetCategories` for merged datasets
    """

    # working with lists is not possible as the order of categories is important here
    init_categories = []
    categories_update = []
    categories_filtered = []
    for cat in categories:
        for label in cat.init_categories:
            if label not in init_categories:
                init_categories.append(label)
        for label in cat.get_categories(as_dict=False):
            if label not in categories_update:
                categories_update.append(label)
        for label in cat.get_categories(as_dict=False, filtered=True):
            if label not in categories_filtered:
                categories_filtered.append(label)

    # select categories with sub categories. Only categories that appear in this list can be candidates for having
    # sub categories in the merged dataset
    intersect_sub_cat_keys = set(categories[0].init_sub_categories.keys()).intersection(
        *[cat.init_sub_categories.keys() for cat in categories[1:]]
    )
    intersect_init_sub_cat = {}
    for key in intersect_sub_cat_keys:
        # select all sub categories from all datasets for a given key
        sub_cat_per_key = [cat.init_sub_categories[key] for cat in categories]
        # select only sub categories that appear in all datasets
        intersect_sub_cat_per_key = set(sub_cat_per_key[0].keys()).intersection(
            *[sub_cats.keys() for sub_cats in sub_cat_per_key[1:]]
        )
        # form a set of possible sub category values. To get a list of all values from all dataset, take the union
        intersect_init_sub_cat_values = {}
        for sub_cat_key in intersect_sub_cat_per_key:
            val: set[ObjectTypes] = set()
            for cat in categories:
                val.update(cat.init_sub_categories[key][sub_cat_key])
            intersect_init_sub_cat_values[sub_cat_key] = list(val)
        intersect_init_sub_cat[key] = intersect_init_sub_cat_values

    # Building sub cats such that the result is deterministic. Because we use sets in several occasions above the
    # construction is not deterministic but guarantees for unique values in all sub categories. Now we build the
    # ensemble dict of sub categories where we guarantee unique values on one hand side and always maintain the
    # same arrangements for all category/ sub category lists
    init_sub_cat: dict[ObjectTypes, Any] = {}
    for category in categories:
        for cat in intersect_sub_cat_keys:
            for sub_cat_key in category.init_sub_categories[cat]:
                if sub_cat_key in intersect_init_sub_cat[cat]:
                    for sub_cat_val in category.init_sub_categories[cat][sub_cat_key]:
                        if sub_cat_val in intersect_init_sub_cat[cat][sub_cat_key]:
                            if cat not in init_sub_cat:
                                init_sub_cat[cat] = {}
                            if sub_cat_key not in init_sub_cat[cat]:
                                init_sub_cat[cat][sub_cat_key] = [sub_cat_val]
                            else:
                                init_sub_cat[cat][sub_cat_key].append(sub_cat_val)

    # Next, build the DatasetCategories instance.
    merged_categories = DatasetCategories(init_categories=init_categories, init_sub_categories=init_sub_cat)
    merged_categories._categories_update = categories_update  # pylint: disable = W0212
    merged_categories._allow_update = False  # pylint: disable = W0212
    setattr(merged_categories, "_categories_filter_update", categories_filtered)
    return merged_categories
