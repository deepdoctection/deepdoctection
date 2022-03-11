# -*- coding: utf-8 -*-
# File: cats.py

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
Categories related mapping functions. They can be set within a pipeline directly after a dataflow
builder method of a dataset.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Union

from ..datapoint.image import Image
from .maputils import cur


@cur  # type: ignore
def cat_to_sub_cat(
    dp: Image, categories_dict_names_as_key: Dict[str, str], cat_to_sub_cat_dict: Optional[Dict[str, str]] = None
) -> Image:
    """
    Replace some category with its affiliated sub category of CategoryAnnotations. Suppose your category name is 'foo'
    and comes along with sub_category_annotations 'foo_1' and 'foo_2' then this adapter will replace 'foo' with
    'foo_1' or 'foo_2', respectively.

    :param dp: Image datapoint
    :param categories_dict_names_as_key: A dict of all possible categories and their ids
    :param cat_to_sub_cat_dict: e.g. {"foo": "sub_cat_1", "bak":"sub_cat_2"}
    :return: Image with updated Annotations
    """

    if cat_to_sub_cat_dict is None:
        return dp
    categories_dict = categories_dict_names_as_key
    for ann in dp.get_annotation_iter(category_names=list(cat_to_sub_cat_dict.keys())):
        sub_cat_type = cat_to_sub_cat_dict.get(ann.category_name, "")
        sub_cat = ann.get_sub_category(sub_cat_type)
        if sub_cat:
            ann.category_name = sub_cat.category_name
            ann.category_id = categories_dict[ann.category_name]

    return dp


@cur  # type: ignore
def re_assign_cat_ids(dp: Image, categories_dict_name_as_key: Dict[str, str]) -> Image:
    """
    Re-assigning category ids is sometimes necessary to align with categories of the :class:`DatasetCategories` . E.g.
    consider the situation where some categories are filtered. In order to guarantee alignment of category ids of the
    :class:`DatasetCategories` the ids in the annotation have to be re-assigned.

    :param dp: Image
    :param categories_dict_name_as_key:
    :return: Image
    """

    for ann in dp.get_annotation_iter():
        ann.category_id = categories_dict_name_as_key[ann.category_name]

    return dp


@cur  # type: ignore
def filter_cat(dp: Image, categories_as_list_filtered: List[str], categories_as_list_unfiltered: List[str]) -> Image:
    """
    Filters category annotations based on the on a list of categories to be kept and a list of all possible
    category names that might be available in dp.

    :param dp: Image datapoint
    :param categories_as_list_filtered: A list of category names with categories to keep. Using a dataset e.g.
                                        my_data.categories.get_categories(as_dict=False,filtered=True)
    :param categories_as_list_unfiltered: A list of all available category names. Using a dataset e.g.
                                          my_data.categories.get_categories(as_dict=False)
    :return: Image with filtered Annotations
    """

    cats_to_remove_list = [cat for cat in categories_as_list_unfiltered if cat not in categories_as_list_filtered]

    remove_cats_mapper = remove_cats(category_names=cats_to_remove_list)  # type: ignore # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp)

    categories_dict_name_as_key = {v: str(k) for k, v in enumerate(categories_as_list_filtered, 1)}
    re_assign_cat_ids_mapper = re_assign_cat_ids(  # pylint: disable=E1120
        categories_dict_name_as_key=categories_dict_name_as_key  # type: ignore
    )
    dp = re_assign_cat_ids_mapper(dp)

    return dp


@cur  # type: ignore
def image_to_cat_id(
    dp: Image,
    category_names: Optional[Union[str, List[str]]] = None,
    sub_category_names: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
) -> Dict[str, List[int]]:
    """
    Extracts all category_ids with given names into a defaultdict with names as keys.

    **Example:**

              dp contains annotations

              {'1', '2', '1', '1', '2', '3'}

              where

              {'foo': '1', 'bak': '2', 'baz': '3'}.

              The returned value will be

              {'foo':['1', '1', '1'], 'bak':[ '2', '2'], 'baz':[ '3']}.

    :param dp: Image datapoint
    :param category_names: A list of category names
    :param sub_category_names: A dict {'cat':'sub_cat'} or a list. Will dump the results with sub_cat as key
    :return: A defaultdict of lists
    """

    cat_container = defaultdict(list)
    if isinstance(category_names, str):
        category_names = [category_names]
    if not category_names:
        category_names = []

    if sub_category_names is None:
        sub_category_names = {}  # type: ignore

    assert sub_category_names is not None

    for key, val in sub_category_names.items():
        if isinstance(val, str):
            val = [val]
            sub_category_names[key] = val  # type: ignore

    for ann in dp.get_annotation_iter():
        if ann.category_name in category_names:
            cat_container[ann.category_name].append(int(ann.category_id))
        if ann.category_name in sub_category_names.keys():
            for sub_cat_name in sub_category_names[ann.category_name]:
                sub_cat = ann.get_sub_category(sub_cat_name)
                if sub_cat is not None:
                    cat_container[sub_cat_name].append(int(sub_cat.category_id))

    return cat_container


@cur  # type: ignore
def remove_cats(
    dp: Image,
    category_names: Optional[Union[str, List[str]]] = None,
    sub_categories: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
) -> Image:
    """
    Remove categories according to given category names or sub category names. Note that these will change the container
    in which the objects are stored.

    :param dp: A datapoint image
    :param category_names: A single category name or a list of categories to remove. On default will remove
                           nothing.
    :param sub_categories: A dict with category names and a list of their sub categories to be removed
    :return: A datapoint image with removed categories
    """

    if isinstance(category_names, str):
        category_names = [category_names]
    if category_names is None:
        category_names = []

    if sub_categories is None:
        sub_categories = {}  # type: ignore

    anns_to_remove = []

    for ann in dp.get_annotation_iter():
        if ann.category_name in category_names:
            anns_to_remove.append(ann)
        if ann.category_name in sub_categories.keys():  # type: ignore
            sub_cats_to_remove = sub_categories[ann.category_name]  # type: ignore
            if isinstance(sub_cats_to_remove, str):
                sub_cats_to_remove = [sub_cats_to_remove]
            for sub_cat in sub_cats_to_remove:
                ann.remove_sub_category(sub_cat)

    for ann in anns_to_remove:
        dp.remove(ann)

    return dp
