# -*- coding: utf-8 -*-
# File: wandbstruct.py

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

"""
Module for W&B (Weights & Biases) mapping and visualization.
"""
from __future__ import annotations

from typing import Mapping, Optional

from lazy_imports import try_import

from dd_datapoint.datapoint.annotation import DEFAULT_CATEGORY_ID, ImageAnnotation
from dd_datapoint.datapoint.image import Image
from dd_datapoint.utils.object_types import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..mapper.maputils import curry

with try_import() as wb_import_guard:
    from wandb import Classes  # type: ignore
    from wandb import Image as Wbimage


__all__ = ["to_wandb_image"]


def _get_category_attributes(
    ann: ImageAnnotation, cat_to_sub_cat: Optional[Mapping[ObjectTypes, ObjectTypes]] = None
) -> tuple[ObjectTypes, int, Optional[float]]:
    """
    Gets the category attributes for an annotation, optionally using a mapping from category to sub-category.

    Args:
        ann: `ImageAnnotation`
        cat_to_sub_cat: Optional mapping from `ObjectTypes` to `ObjectTypes`.

    Returns:
        Tuple of `ObjectTypes`, `category_id`, and `score`.
    """
    if cat_to_sub_cat:
        sub_cat_key = cat_to_sub_cat.get(get_type(ann.category_name))
        if sub_cat_key in ann.sub_categories:
            sub_cat = ann.get_sub_category(sub_cat_key)
            return get_type(sub_cat.category_name), sub_cat.category_id, sub_cat.score
        return DefaultType.DEFAULT_TYPE, DEFAULT_CATEGORY_ID, 0.0
    return get_type(ann.category_name), ann.category_id, ann.score


@curry
def to_wandb_image(
    dp: Image,
    categories: Mapping[int, TypeOrStr],
    sub_categories: Optional[Mapping[int, TypeOrStr]] = None,
    cat_to_sub_cat: Optional[Mapping[ObjectTypes, ObjectTypes]] = None,
) -> tuple[str, Wbimage]:
    """
    Converts a deepdoctection `Image` into a `W&B` image.

    Args:
        dp: deepdoctection `Image`
        categories: Dict of categories. The categories refer to categories of `ImageAnnotation`.
        sub_categories: Dict of `sub_categories`. If provided, these categories will define the classes for the table.
        cat_to_sub_cat: Dict of category to sub_category keys. Suppose your category `foo` has a sub-category defined
                        by the key `sub_foo`. The range of sub-category values must then be given by `sub_categories`,
                        and to extract the sub-category values, one must pass `{"foo": "sub_foo"}`.

    Returns:
        Tuple of `image_id` and a W&B image.

    Example:
        ```python
        to_wandb_image(dp, categories)
        ```
    """
    if dp.image is None:
        raise ValueError("Cannot convert to W&B image type when Image.image is None")

    boxes = []
    anns = dp.get_annotation(category_names=list(categories.values()))

    if sub_categories:
        class_labels = dict(sub_categories.items())
        class_set = Classes([{"name": val, "id": key} for key, val in sub_categories.items()])
    else:
        class_set = Classes([{"name": val, "id": key} for key, val in categories.items()])
        class_labels = dict(categories.items())

    for ann in anns:
        bounding_box = ann.get_bounding_box(dp.image_id)
        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(dp.width, dp.height, True)
        category_name, category_id, score = _get_category_attributes(ann, cat_to_sub_cat)
        if category_name:
            box = {
                "position": {"middle": bounding_box.center, "width": bounding_box.width, "height": bounding_box.height},
                "domain": "pixel",
                "class_id": category_id,
                "box_caption": category_name,
            }
            if score:
                box["scores"] = {"acc": score}
            boxes.append(box)

    predictions = {"predictions": {"box_data": boxes, "class_labels": class_labels}}

    return dp.image_id, Wbimage(dp.image[:, :, ::-1], mode="RGB", boxes=predictions, classes=class_set)

