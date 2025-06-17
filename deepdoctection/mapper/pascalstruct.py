# -*- coding: utf-8 -*-
# File: iiitarstruct.py

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
Mapping for PASCAL VOC dataset structure to `Image` format.
"""

import os
from typing import Optional

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.box import BoundingBox
from ..datapoint.image import Image
from ..utils.fs import load_image_from_file
from ..utils.settings import get_type
from ..utils.types import JsonDict
from .maputils import MappingContextManager, curry, maybe_get_fake_score


@curry
def pascal_voc_dict_to_image(
    dp: JsonDict,
    categories_name_as_key: dict[str, int],
    load_image: bool,
    filter_empty_image: bool,
    fake_score: bool,
    category_name_mapping: Optional[dict[str, str]] = None,
) -> Optional[Image]:
    """
    Maps a dataset in a structure equivalent to the PASCAL VOC annotation style to the `Image` format.

    Args:
        dp: A datapoint in PASCAL VOC format. Note that another conversion from XML to a dict structure is required.
        categories_name_as_key: A dict of categories, e.g. `DatasetCategories.get_categories(name_as_key=True)`.
        load_image: If `True`, it will load the image to the attribute `Image.image`.
        filter_empty_image: Will return `None` if the datapoint has no annotations.
        fake_score: If `dp` does not contain a score, a fake score with uniform random variables in (0,1) will be added.
        category_name_mapping: Map incoming category names, e.g. `{"source_name": "target_name"}`.

    Returns:
        `Image` or `None`.
    """

    anns = dp.get("objects", [])
    if not anns and filter_empty_image:
        return None

    with MappingContextManager(dp.get("filename")) as mapping_context:
        image = Image(
            file_name=os.path.split(dp["filename"])[1].replace(".xml", ".jpg"),
            location=dp["filename"].replace(".xml", ".jpg").replace("xml", "images"),
        )

        if load_image:
            image.image = load_image_from_file(image.location)
        image.set_width_height(float(dp.get("width", 0)), float(dp.get("height", 0)))

        for ann in anns:
            x_1 = min(max(ann["xmin"], 0), image.width if image.width else float(dp.get("width", 0)))
            x_2 = min(max(ann["xmax"], 0), image.width if image.width else float(dp.get("width", 0)))
            y_1 = min(max(ann["ymin"], 0), image.height if image.height else float(dp.get("height", 0)))
            y_2 = min(max(ann["ymax"], 0), image.height if image.height else float(dp.get("height", 0)))

            bbox = BoundingBox(absolute_coords=True, ulx=x_1, uly=y_1, lrx=x_2, lry=y_2)

            if category_name_mapping is not None:
                label = category_name_mapping.get(ann["name"])
                if not label:
                    label = ann["name"]
            else:
                label = ann["label"]
            assert isinstance(label, str)

            annotation = ImageAnnotation(
                category_name=get_type(label),
                bounding_box=bbox,
                category_id=categories_name_as_key[label],
                score=maybe_get_fake_score(fake_score),
            )
            image.dump(annotation)

    if mapping_context.context_error:
        return None
    return image
