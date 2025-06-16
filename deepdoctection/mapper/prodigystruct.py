# -*- coding: utf-8 -*-
# File: prodigystruct.py

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
Module for mapping annotations to and from prodigy data structure.
"""

import os
from typing import Mapping, Optional, Sequence

from ..datapoint import BoundingBox, Image, ImageAnnotation
from ..utils.settings import ObjectTypes, get_type
from ..utils.types import JsonDict, PathLikeOrStr
from .maputils import MappingContextManager, curry, maybe_get_fake_score

_PRODIGY_IMAGE_PREFIX = "data:image/png;base64,"


@curry
def prodigy_to_image(
    dp: JsonDict,
    categories_name_as_key: Mapping[ObjectTypes, int],
    load_image: bool,
    fake_score: bool,
    path_reference_ds: Optional[PathLikeOrStr] = None,
    accept_only_answer: bool = False,
    category_name_mapping: Optional[Mapping[str, str]] = None,
) -> Optional[Image]:
    """
    Maps a datapoint of annotation structure from Prodigy database to an `Image` structure.

    Args:
        dp: A datapoint in dict structure as returned from Prodigy database.
        categories_name_as_key: A dict of categories, e.g. `DatasetCategories.get_categories(name_as_key=True)`.
        load_image: If `True`, it will load image to `Image.image`.
        fake_score: If `dp` does not contain a score, a fake score with uniform random variables in (0,1) will be added.
        path_reference_ds: A path to a reference-dataset. It must point to the basedir where the file of the datapoint
                           can be found.
        accept_only_answer: Filter every datapoint that has the answer `reject` or `ignore`.
        category_name_mapping: Map incoming category names, e.g. `{"source_name":"target_name"}`.

    Returns:
        `Image`

    Note:
        If `accept_only_answer` is `True`, only datapoints with the answer `accept` will be processed.

    """

    if accept_only_answer and dp.get("answer") != "accept":
        return None

    file_name: Optional[str] = None
    meta = dp.get("meta")
    if meta:
        file_name = meta.get("file")

    if not file_name:
        file_name = dp.get("id")
    if not file_name:
        file_name = dp.get("text")
    if not file_name:
        path = dp.get("path")
        if path:
            path, file_name = os.path.split(path)
    if not file_name:
        file_name = ""

    external_id = dp.get("image_id")

    if not external_id:
        external_id = file_name

    with MappingContextManager(file_name) as mapping_context:
        path_reference_location = ""
        if path_reference_ds:
            path_reference_location = os.path.join(path_reference_ds, file_name)
        location = dp.get("path", path_reference_location)
        if not os.path.isfile(location):
            location = None

        image = Image(file_name=file_name, location=location, external_id=external_id)

        if dp.get("image"):
            image.image = dp["image"].split(",")[1]
            if not load_image:
                image.clear_image()
        elif "width" in dp and "height" in dp:
            image.set_width_height(dp["width"], dp["height"])

        spans = dp.get("spans", [])

        for span in spans:
            ulx, uly = list(map(float, span["points"][0]))
            lrx, lry = list(map(float, span["points"][2]))
            ulx = min(max(ulx, 0), image.width if image.width else ulx)
            uly = min(max(uly, 0), image.height if image.height else uly)
            lrx = min(max(lrx, 0), image.width if image.width else lrx)
            lry = min(max(lry, 0), image.height if image.height else lry)
            upper_left = [ulx, uly]
            lower_right = [lrx, lry]

            bbox = BoundingBox(
                absolute_coords=True, ulx=upper_left[0], uly=upper_left[1], lrx=lower_right[0], lry=lower_right[1]
            )
            external_id = span.get("annotation_id")
            if not external_id:
                external_id = span.get("id")

            score = span.get("score")

            if not score:
                score = maybe_get_fake_score(fake_score)

            if category_name_mapping is not None:
                label = category_name_mapping.get(span["label"])
                if not label:
                    label = span["label"]
            else:
                label = span["label"]
            if not isinstance(label, str):
                raise TypeError("label must be a string")

            annotation = ImageAnnotation(
                category_name=label,
                bounding_box=bbox,
                category_id=categories_name_as_key[get_type(label)],
                score=score,
                external_id=external_id,
            )
            image.dump(annotation)

    if mapping_context.context_error:
        return None
    return image


@curry
def image_to_prodigy(dp: Image, category_names: Optional[Sequence[ObjectTypes]] = None) -> JsonDict:
    """
    Transforms the normalized image representation of datasets into the format for visualizing the annotation
    components in Prodigy.

    Args:
        dp: An `Image`.
        category_names: A list of category names to filter the annotations.

    Returns:
        A dictionary with compulsory keys: `text` and `spans`.

    Example:
        ```python
        image_to_prodigy(image_instance)
        ```

    """

    output: JsonDict = {}
    img_str = dp.get_image().to_b64()
    if img_str is not None:
        output["image"] = _PRODIGY_IMAGE_PREFIX + img_str
    output["text"] = dp.file_name
    output["image_id"] = dp.image_id

    spans = []
    for ann in dp.get_annotation(category_names=category_names):
        bounding_box = ann.get_bounding_box(dp.image_id)
        if not bounding_box.absolute_coords:
            bounding_box = bounding_box.transform(dp.width, dp.height, absolute_coords=True)
        boxes = [
            [bounding_box.ulx, bounding_box.uly],
            [bounding_box.ulx, bounding_box.lry],
            [bounding_box.lrx, bounding_box.lry],
            [bounding_box.lrx, bounding_box.uly],
        ]

        span: JsonDict = {
            "label": ann.category_name.value,  # type: ignore
            "annotation_id": ann.annotation_id,
            "type": "rect",
            "points": boxes,
            "score": float(round(ann.score, 3)) if ann.score is not None else None,
        }
        spans.append(span)

    output["spans"] = spans
    output["meta"] = {"file_name": dp.file_name}
    return output
