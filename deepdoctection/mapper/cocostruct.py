# -*- coding: utf-8 -*-
# File: cocostruct.py

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
Module for mapping annotations in coco style structure
"""

import os
from typing import Mapping, Optional

from ..datapoint.annotation import CategoryAnnotation, ImageAnnotation
from ..datapoint.box import BoundingBox
from ..datapoint.image import Image
from ..utils.fs import load_image_from_file
from ..utils.settings import ObjectTypes
from ..utils.types import CocoDatapointDict, JsonDict
from .maputils import MappingContextManager, curry, maybe_get_fake_score


@curry
def coco_to_image(
    dp: CocoDatapointDict,
    categories: dict[int, ObjectTypes],
    load_image: bool,
    filter_empty_image: bool,
    fake_score: bool,
    coarse_mapping: Optional[Mapping[int, int]] = None,
    coarse_sub_cat_name: Optional[ObjectTypes] = None,
) -> Optional[Image]:
    """
    Maps a dataset in `COCO` format that has been serialized to image format.

    This serialized input requirements hold when a `COCO` style sheet is loaded via `SerializerCoco.load`.

    Args:
        dp: A datapoint in serialized COCO format.
        categories: A dict of categories, e.g. `DatasetCategories.get_categories`.
        load_image: If `True`, it will load image to `Image.image`.
        filter_empty_image: Will return `None` if datapoint has no annotations.
        fake_score: If `dp` does not contain a score, a fake score with uniform random variables in `(0,1)` will be
                    added.
        coarse_mapping: A mapping to map categories into broader categories. Note that the coarser categories must
                        already be included in the original mapping.
        coarse_sub_cat_name: A name to be provided as sub category key for a coarse mapping.

    Returns:
        `Image` or `None`.

    Raises:
        ValueError: If `coarse_sub_cat_name` is provided but `coarse_mapping` is not.

    Note:
        A coarse mapping must be provided when `coarse_sub_cat_name` has been passed.
    """

    if coarse_sub_cat_name and coarse_mapping is None:
        raise ValueError("A coarse mapping must be provided when coarse_sub_cat_name have been passed")

    anns = dp.get("annotations", [])
    if not anns and filter_empty_image:
        return None

    with MappingContextManager(dp.get("file_name")) as mapping_context:
        image = Image(file_name=os.path.split(dp["file_name"])[1], location=dp["file_name"], external_id=dp.get("id"))

        if load_image:
            image.image = load_image_from_file(dp["file_name"])
        image.set_width_height(float(dp.get("width", 0)), float(dp.get("height", 0)))

        for ann in anns:
            if ann.get("ignore", 0) == 1:
                continue

            # will do the same sanity checks as for Tensorpack Faster RCNN
            box = ann.get("bbox", [])
            x_1, y_1, w, h = list(map(float, box))
            x_2, y_2 = x_1 + w, y_1 + h
            x_1 = min(max(x_1, 0), image.width if image.width else float(dp.get("width", 0)))
            x_2 = min(max(x_2, 0), image.width if image.width else float(dp.get("width", 0)))
            y_1 = min(max(y_1, 0), image.height if image.height else float(dp.get("height", 0)))
            y_2 = min(max(y_2, 0), image.height if image.height else float(dp.get("height", 0)))
            w, h = x_2 - x_1, y_2 - y_1

            bbox = BoundingBox(absolute_coords=True, ulx=x_1, uly=y_1, height=h, width=w)

            annotation = ImageAnnotation(
                category_name=categories[ann["category_id"]],
                bounding_box=bbox,
                category_id=ann["category_id"],
                score=maybe_get_fake_score(fake_score),
                external_id=ann["id"],
            )
            image.dump(annotation)

            if coarse_sub_cat_name and coarse_mapping:
                sub_cat = CategoryAnnotation(
                    category_name=categories[coarse_mapping[ann["category_id"]]],
                    category_id=coarse_mapping[ann["category_id"]],
                )
                annotation.dump_sub_category(coarse_sub_cat_name, sub_cat)

    if mapping_context.context_error:
        return None

    return image


def image_to_coco(dp: Image) -> tuple[JsonDict, list[JsonDict]]:
    """
    Converts an image back into the `COCO` format.

    As images and annotations are separated, it will return a dict with the image information and one for its
    annotations.

    Args:
        dp: An `Image`.

    Returns:
        A tuple of dicts, the first corresponding to the COCO-image object, the second to their COCO-annotations.

    Raises:
        TypeError: If `dp` is not of type `Image`.
    """

    if not isinstance(dp, Image):
        raise TypeError(f"datapoints must be of type Image, is of type {type(dp)}")

    img: JsonDict = {}
    anns: list[JsonDict] = []

    img["id"] = int("".join([s for s in dp.image_id if s.isdigit()]))
    img["width"] = dp.width
    img["height"] = dp.height
    img["file_name"] = dp.file_name

    for img_ann in dp.get_annotation():
        ann: JsonDict = {
            "id": int("".join([s for s in img_ann.annotation_id if s.isdigit()])),
            "image_id": img["id"],
            "category_id": img_ann.category_id,
        }
        if img_ann.score:
            ann["score"] = img_ann.score
        ann["iscrowd"] = 0
        bounding_box = img_ann.get_bounding_box(dp.image_id)
        ann["area"] = (
            bounding_box.area
            if bounding_box.absolute_coords
            else bounding_box.transform(dp.width, dp.height, absolute_coords=True).area
        )
        ann["bbox"] = bounding_box.to_list(mode="xywh")
        anns.append(ann)

    return img, anns
