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
Mapping functions for handling categories
"""

from collections import defaultdict
from typing import Any, Literal, Mapping, Optional, Sequence, Union

from ..datapoint.annotation import DEFAULT_CATEGORY_ID, CategoryAnnotation, ContainerAnnotation
from ..datapoint.image import Image
from ..utils.settings import ObjectTypes, SummaryType, TypeOrStr, get_type
from .maputils import LabelSummarizer, curry


@curry
def cat_to_sub_cat(
    dp: Image,
    categories_dict_names_as_key: Optional[dict[TypeOrStr, int]] = None,
    cat_to_sub_cat_dict: Optional[dict[TypeOrStr, TypeOrStr]] = None,
) -> Image:
    """
    Replace some categories with sub categories.

    Example:
        ```python
        categories_dict_names_as_key = {'foo': 1}
        cat_to_sub_cat_dict = {'foo': 'foo_1', 'bak': 'bak_1'}

        dp = cat_to_sub_cat(categories_dict_names_as_key, cat_to_sub_cat_dict)(dp)
        ```

        will transform

        ```python
        ImageAnnotation(category_name='foo', category_id=1, ...)
        ```

        to

        ```python
        ImageAnnotation(category_name='foo_1', category_id=1, ...)
        ```


    Args:
        dp: Image datapoint
        categories_dict_names_as_key: A dict of all possible categories and their ids
        cat_to_sub_cat_dict: e.g. `{'foo': 'sub_cat_1', 'bak': 'sub_cat_2'}`

    Returns:
        Image with updated `ImageAnnotation`s
    """
    if categories_dict_names_as_key is None:
        categories_dict_names_as_key = {}
    if cat_to_sub_cat_dict is None:
        return dp
    cat_to_sub_cat_dict_obj_type = {get_type(key): get_type(value) for key, value in cat_to_sub_cat_dict.items()}
    for ann in dp.get_annotation(category_names=list(cat_to_sub_cat_dict_obj_type.keys())):
        sub_cat_type = cat_to_sub_cat_dict_obj_type[get_type(ann.category_name)]
        sub_cat = ann.get_sub_category(sub_cat_type)
        if sub_cat:
            ann.category_name = sub_cat.category_name
            ann.category_id = categories_dict_names_as_key.get(ann.category_name, DEFAULT_CATEGORY_ID)

    return dp


@curry
def re_assign_cat_ids(
    dp: Image,
    categories_dict_name_as_key: Optional[dict[TypeOrStr, int]] = None,
    cat_to_sub_cat_mapping: Optional[Mapping[ObjectTypes, Any]] = None,
) -> Image:
    """
    Re-assigning `category_id`s is sometimes necessary to align with categories of the `DatasetCategories` .

    Example:
        Consider the situation where some categories are filtered. In order to guarantee alignment of category ids of
        the `DatasetCategories` the ids in the annotation have to be re-assigned.

        Annotations that are not in the dictionary provided will be removed.

    Args:
        dp: Image
        categories_dict_name_as_key: e.g. `{LayoutType.word: 1}`
        cat_to_sub_cat_mapping: e.g. `{LayoutType.word: {WordType.token_class:
                                                          {FundsFirstPage.REPORT_DATE: 1,
                                                           FundsFirstPage.REPORT_TYPE: 2,
                                                           FundsFirstPage.UMBRELLA: 3,
                                                           FundsFirstPage.FUND_NAME: 4,
                                                           TokenClasses.OTHER: 5},
                                                           WordType.TAG:{ BioTag.INSIDE: 1,
                                                                          BioTag.OUTSIDE: 2,
                                                                          BioTag.BEGIN: 3}}}`
            To re-assign the `category_id`s of an image summary, use the key 'default_type' for the default category,
             e.g. `{DefaultType.DEFAULT_TYPE: {PageType.DOCUMENT_TYPE: {DocumentType.INVOICE:1,
            DocumentType.BANK_STATEMENT:2}}}`
    Returns:
        Image
    """

    ann_ids_to_remove: list[str] = []
    for ann in dp.get_annotation():
        if categories_dict_name_as_key is not None:
            if ann.category_name in categories_dict_name_as_key:
                ann.category_id = categories_dict_name_as_key[ann.category_name]
            else:
                ann_ids_to_remove.append(ann.annotation_id)

        if cat_to_sub_cat_mapping:
            if ann.category_name in cat_to_sub_cat_mapping:
                sub_cat_keys_to_sub_cat_values = cat_to_sub_cat_mapping[get_type(ann.category_name)]
                for key in sub_cat_keys_to_sub_cat_values:
                    sub_cat_values_dict = sub_cat_keys_to_sub_cat_values[key]
                    sub_category = ann.get_sub_category(key)
                    sub_category.category_id = sub_cat_values_dict.get(sub_category.category_name, DEFAULT_CATEGORY_ID)

    if cat_to_sub_cat_mapping:
        if "default_type" in cat_to_sub_cat_mapping:
            sub_cat_keys_to_sub_cat_values = cat_to_sub_cat_mapping[get_type("default_type")]
            for key in sub_cat_keys_to_sub_cat_values:
                sub_cat_values_dict = sub_cat_keys_to_sub_cat_values[key]
                sub_category = dp.summary.get_sub_category(key)
                sub_category.category_id = sub_cat_values_dict.get(sub_category.category_name, DEFAULT_CATEGORY_ID)

    dp.remove(annotation_ids=ann_ids_to_remove)

    return dp


@curry
def filter_cat(
    dp: Image, categories_as_list_filtered: list[TypeOrStr], categories_as_list_unfiltered: list[TypeOrStr]
) -> Image:
    """
    Filters category annotations based on the on a list of categories to be kept and a list of all possible
    category names that might be available in dp.

    Args:
        dp: Image datapoint
        categories_as_list_filtered: A list of `category_name`s with categories to keep. Using a dataset e.g.
                                        `my_data.categories.get_categories(as_dict=False,filtered=True)`
        categories_as_list_unfiltered: A list of all available `category_name`s. Using a dataset e.g.
                                          `my_data.categories.get_categories(as_dict=False)`
    Returns:
        Image with filtered Annotations
    """

    cats_to_remove_list = [cat for cat in categories_as_list_unfiltered if cat not in categories_as_list_filtered]

    remove_cats_mapper = remove_cats(category_names=cats_to_remove_list)  # pylint: disable=E1120  # 259
    dp = remove_cats_mapper(dp)

    categories_dict_name_as_key = {v: k for k, v in enumerate(categories_as_list_filtered, 1)}
    re_assign_cat_ids_mapper = re_assign_cat_ids(  # pylint: disable=E1120
        categories_dict_name_as_key=categories_dict_name_as_key
    )
    dp = re_assign_cat_ids_mapper(dp)

    return dp


@curry
def filter_summary(
    dp: Image,
    sub_cat_to_sub_cat_names_or_ids: Mapping[TypeOrStr, Sequence[TypeOrStr]],
    mode: Literal["name", "id", "value"] = "name",
) -> Optional[Image]:
    """
    Filters datapoints with given summary conditions. If several conditions are given, it will filter out datapoints
    that do not satisfy all conditions.

    Args:
        dp: Image datapoint
        sub_cat_to_sub_cat_names_or_ids: A dict of list. The key correspond to the sub category key to look for in
                                            the summary. The value correspond to a sequence of either category names
                                            or category ids
        mode: With respect to the previous argument, it will look if the `category_name`, the `value` or the
              `category_id` corresponds to any of the given values.
    Returns:
        Image or `None`
    """
    for key, values in sub_cat_to_sub_cat_names_or_ids.items():
        if mode == "name":
            if dp.summary.get_sub_category(get_type(key)).category_name in values:
                return dp
        elif mode == "value":
            if dp.summary.get_sub_category(get_type(key)).value in values:  # type: ignore
                return dp
        else:
            if dp.summary.get_sub_category(get_type(key)).category_id in values:
                return dp
    return None


@curry
def image_to_cat_id(
    dp: Image,
    category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    sub_categories: Optional[Union[Mapping[TypeOrStr, TypeOrStr], Mapping[TypeOrStr, Sequence[TypeOrStr]]]] = None,
    summary_sub_category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    id_name_or_value: Literal["id", "name", "value"] = "id",
) -> tuple[dict[TypeOrStr, list[int]], str]:
    """
    Extracts all category_ids, sub category information or summary sub category information with given names into a
    defaultdict. This mapping is useful when running evaluation with e.g. an accuracy metric.

    Example 1:

        dp contains image annotations
        ```python
        ImageAnnotation(category_name='foo',category_id='1',...),
        ImageAnnotation(category_name='bak',category_id='2',...),
        ImageAnnotation(category_name='baz',category_id='3',...),
        ImageAnnotation(category_name='foo',category_id='1',...),
        ```

        Then

        ```python
        image_to_cat_id(category_names=['foo', 'bak', 'baz'])(dp)
        ```
        will return

        ```python
        ({'foo':[1,1], 'bak':[2], 'baz':[3]}, image_id)
        ```

    Example 2:

        dp contains image annotations as given in Example 1. Moreover, the 'foo' image annotation have sub categories:

        ```python
        foo_sub_1: CategoryAnnotation(category_name='sub_1', category_id='4')
        foo_sub_1: CategoryAnnotation(category_name='sub_1', category_id='5')

        image_to_cat_id(sub_categories={'foo':'foo_sub_1'})
        ```

        will return

        ```python
        ({'foo_sub_1':[5,6]}, image_id)
        ```

    Args:
        dp: Image
        category_names: A list of category names
        sub_categories: A dict `{'cat':'sub_cat'}` or a list. Will dump the results with sub_cat as key
        id_name_or_value: Only relevant for sub categories. It will extract the sub category id, the name or, if the
                             sub category is a container, it will extract a value.
        summary_sub_category_names: A list of summary sub categories

    Returns:
        A defaultdict of lists
    """

    cat_container = defaultdict(list)

    if isinstance(category_names, str):
        category_names = [category_names]
    if not category_names:
        category_names = []

    if isinstance(summary_sub_category_names, str):
        summary_sub_category_names = [summary_sub_category_names]
    if not summary_sub_category_names:
        summary_sub_category_names = []

    tmp_sub_category_names: dict[str, Sequence[str]] = {}

    if sub_categories is not None:
        for key, val in sub_categories.items():
            if isinstance(val, str):
                val = [val]
            tmp_sub_category_names[key] = val

    if id_name_or_value not in ("id", "name", "value"):
        raise ValueError(f"id_name_or_value must be in ('id', 'name', 'value') but is {id_name_or_value}")

    if category_names or sub_categories:
        for ann in dp.get_annotation():
            if ann.category_name in category_names:
                cat_container[ann.category_name].append(ann.category_id)
            if ann.category_name in tmp_sub_category_names:
                for sub_cat_name in tmp_sub_category_names[ann.category_name]:
                    sub_cat = ann.get_sub_category(get_type(sub_cat_name))
                    if sub_cat is not None:
                        if id_name_or_value == "id":
                            cat_container[sub_cat_name].append(sub_cat.category_id)
                        if id_name_or_value == "name":
                            cat_container[sub_cat_name].append(sub_cat.category_name)  # type: ignore
                        if id_name_or_value == "value":
                            if not isinstance(sub_cat, ContainerAnnotation):
                                raise ValueError(
                                    f"sub category {sub_cat_name} does not have a ContainerAnnotation. Choose another"
                                    f"value for argument id_name_or_value"
                                )
                            cat_container[sub_cat_name].append(sub_cat.value)  # type: ignore

    if summary_sub_category_names:
        for sub_cat_name in summary_sub_category_names:
            sub_cat = dp.summary.get_sub_category(get_type(sub_cat_name))
            if id_name_or_value == "id":
                cat_container[sub_cat_name].append(sub_cat.category_id)
            if id_name_or_value == "name":
                cat_container[sub_cat_name].append(sub_cat.category_name)  # type: ignore
            if id_name_or_value == "value":
                if not isinstance(sub_cat, ContainerAnnotation):
                    raise ValueError(
                        f"sub category {sub_cat_name} does not have a ContainerAnnotation. Choose another"
                        f"value for argument id_name_or_value"
                    )
                cat_container[sub_cat_name].append(sub_cat.value)  # type: ignore

    return cat_container, dp.image_id


@curry
def remove_cats(
    dp: Image,
    category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
    sub_categories: Optional[Union[Mapping[TypeOrStr, TypeOrStr], Mapping[TypeOrStr, Sequence[TypeOrStr]]]] = None,
    relationships: Optional[Union[Mapping[TypeOrStr, TypeOrStr], Mapping[TypeOrStr, Sequence[TypeOrStr]]]] = None,
    summary_sub_categories: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
) -> Image:
    """
    Remove categories according to given category names or sub category names. Note that these will change the container
    in which the objects are stored.

    Args:
        dp: A datapoint image
        category_names: A single category name or a list of categories to remove. On default will remove
                           nothing.
        sub_categories: A dict with category names and a list of their sub categories to be removed
        relationships: A dict with category names and a list of relationship names to be removed
        summary_sub_categories: A single sub category or a list of sub categories from a summary to be removed

    Returns:
        A datapoint image with removed categories
    """

    if isinstance(category_names, str):
        category_names = [category_names]
    if category_names is None:
        category_names = []

    if sub_categories is None:
        sub_categories = {}

    if relationships is None:
        relationships = {}

    if isinstance(summary_sub_categories, str):
        summary_sub_categories = [summary_sub_categories]

    ann_ids_to_remove = []

    for ann in dp.get_annotation():
        if ann.category_name in category_names:
            ann_ids_to_remove.append(ann.annotation_id)
        if ann.category_name in sub_categories.keys():
            sub_cats_to_remove = sub_categories[ann.category_name]
            if isinstance(sub_cats_to_remove, str):
                sub_cats_to_remove = [sub_cats_to_remove]
            for sub_cat in sub_cats_to_remove:
                ann.remove_sub_category(get_type(sub_cat))
        if ann.category_name in relationships.keys():
            relationships_to_remove = relationships[ann.category_name]
            if isinstance(relationships_to_remove, str):
                relationships_to_remove = [relationships_to_remove]
            for relation in relationships_to_remove:
                ann.remove_relationship(key=get_type(relation))

    dp.remove(annotation_ids=ann_ids_to_remove)

    if summary_sub_categories is not None:
        for sub_cat in summary_sub_categories:
            dp.summary.remove_sub_category(get_type(sub_cat))

    return dp


@curry
def add_summary(dp: Image, categories: Mapping[int, ObjectTypes]) -> Image:
    """
    Adding a summary with the number of categories in an image.

    Args:
        dp: Image
        categories: A dict of all categories, e.g. `{"1": "text", "2":"title", ...}`

    Returns:
        Image
    """
    category_list = list(categories.values())
    anns = dp.get_annotation(category_names=category_list)
    summarizer = LabelSummarizer(categories)
    for ann in anns:
        summarizer.dump(ann.category_id)
    summary_dict = summarizer.get_summary()
    summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)
    for cat_id, val in summary_dict.items():
        summary.dump_sub_category(
            categories[cat_id], CategoryAnnotation(category_name=categories[cat_id], category_id=val)
        )
    dp.summary = summary
    return dp
