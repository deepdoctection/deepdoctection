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
from typing import Dict, List, Mapping, Optional, Sequence, Union, overload, Literal

from ..datapoint.annotation import ImageAnnotation, ContainerAnnotation
from ..datapoint.image import Image
from .maputils import curry


@curry
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


@curry
def re_assign_cat_ids(dp: Image, categories_dict_name_as_key: Dict[str, str]) -> Image:
    """
    Re-assigning category ids is sometimes necessary to align with categories of the :class:`DatasetCategories` . E.g.
    consider the situation where some categories are filtered. In order to guarantee alignment of category ids of the
    :class:`DatasetCategories` the ids in the annotation have to be re-assigned.

    Annotations that as not in the dictionary provided will removed from the image.

    :param dp: Image
    :param categories_dict_name_as_key:
    :return: Image
    """

    anns_to_remove: List[ImageAnnotation] = []
    for ann in dp.get_annotation_iter():
        if ann.category_name in categories_dict_name_as_key:
            ann.category_id = categories_dict_name_as_key[ann.category_name]
        else:
            anns_to_remove.append(ann)

    for ann in anns_to_remove:
        dp.remove(ann)

    return dp


@curry
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

    remove_cats_mapper = remove_cats(category_names=cats_to_remove_list)  # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp)

    categories_dict_name_as_key = {v: str(k) for k, v in enumerate(categories_as_list_filtered, 1)}
    re_assign_cat_ids_mapper = re_assign_cat_ids(  # pylint: disable=E1120
        categories_dict_name_as_key=categories_dict_name_as_key
    )
    dp = re_assign_cat_ids_mapper(dp)

    return dp


@overload
@curry
def image_to_cat_id(
    dp: Image,
    category_names: Optional[Union[str, Sequence[str]]] = None,
    sub_category_names: Optional[Union[Mapping[str, str], Mapping[str, Sequence[str]]]] = None,
    id_name_or_value: Literal["id"] = ...) -> Dict[str, List[int]]: ...


@overload
@curry
def image_to_cat_id(
    dp: Image,
    category_names: Optional[Union[str, Sequence[str]]] = None,
    sub_category_names: Optional[Union[Mapping[str, str], Mapping[str, Sequence[str]]]] = None,
    id_name_or_value: Literal["name", "value"] = ...) -> Dict[str, List[str]]: ...


@curry
def image_to_cat_id(
    dp: Image,
    category_names: Optional[Union[str, Sequence[str]]] = None,
    sub_category_names: Optional[Union[Mapping[str, str], Mapping[str, Sequence[str]]]] = None,
    id_name_or_value: str = "id"
) -> Dict[str, List[int]]:
    """
    Extracts all category_ids or sub category information with given names into a defaultdict with names as keys.

    **Example:**

        dp contains annotations

        `{'1', '2', '1', '1', '2', '3'}`

        where

        `{'foo': '1', 'bak': '2', 'baz': '3'}`.

        The returned value will be

        `{'foo':['1', '1', '1'], 'bak':[ '2', '2'], 'baz':[ '3']}`.

    :param dp: Image datapoint
    :param category_names: A list of category names
    :param sub_category_names: A dict {'cat':'sub_cat'} or a list. Will dump the results with sub_cat as key
    :param id_name_or_value: Only relevant for sub categories. It will extract the sub category id, the name or, if the
                             sub category is a container, it will extract a value.
    :return: A defaultdict of lists
    """

    cat_container = defaultdict(list)
    if isinstance(category_names, str):
        category_names = [category_names]
    if not category_names:
        category_names = []

    tmp_sub_category_names: Dict[str, Sequence[str]] = {}

    if sub_category_names is not None:
        for key, val in sub_category_names.items():
            if isinstance(val, str):
                val = [val]
            tmp_sub_category_names[key] = val

    assert id_name_or_value in ("id", "name", "value")

    for ann in dp.get_annotation_iter():
        if ann.category_name in category_names:
            cat_container[ann.category_name].append(int(ann.category_id))
        if ann.category_name in tmp_sub_category_names:
            for sub_cat_name in tmp_sub_category_names[ann.category_name]:
                sub_cat = ann.get_sub_category(sub_cat_name)
                if sub_cat is not None:
                    if id_name_or_value=="id":
                        cat_container[sub_cat_name].append(int(sub_cat.category_id))
                    if id_name_or_value=="name":
                        cat_container[sub_cat_name].append(sub_cat.category_name)
                    if id_name_or_value=="value":
                        assert isinstance(sub_cat, ContainerAnnotation)
                        cat_container[sub_cat_name].append(sub_cat.value)

    return cat_container


@curry
def remove_cats(
    dp: Image,
    category_names: Optional[Union[str, Sequence[str]]] = None,
    sub_categories: Optional[Union[Mapping[str, str], Mapping[str, Sequence[str]]]] = None,
    relationships: Optional[Union[Mapping[str, str], Mapping[str, Sequence[str]]]] = None,
) -> Image:
    """
    Remove categories according to given category names or sub category names. Note that these will change the container
    in which the objects are stored.

    :param dp: A datapoint image
    :param category_names: A single category name or a list of categories to remove. On default will remove
                           nothing.
    :param sub_categories: A dict with category names and a list of their sub categories to be removed
    :param relationships: A dict with category names and a list of relationship names to be removed
    :return: A datapoint image with removed categories
    """

    if isinstance(category_names, str):
        category_names = [category_names]
    if category_names is None:
        category_names = []

    if sub_categories is None:
        sub_categories = {}

    if relationships is None:
        relationships = {}

    anns_to_remove = []

    for ann in dp.get_annotation_iter():
        if ann.category_name in category_names:
            anns_to_remove.append(ann)
        if ann.category_name in sub_categories.keys():
            sub_cats_to_remove = sub_categories[ann.category_name]
            if isinstance(sub_cats_to_remove, str):
                sub_cats_to_remove = [sub_cats_to_remove]
            for sub_cat in sub_cats_to_remove:
                ann.remove_sub_category(sub_cat)
        if ann.category_name in relationships.keys():
            relationships_to_remove = relationships[ann.category_name]
            if isinstance( relationships_to_remove, str):
                relationships_to_remove = [relationships_to_remove]
            for relation in relationships_to_remove:
                ann.remove_relationship(key=relation)

    for ann in anns_to_remove:
        dp.remove(ann)

    return dp
