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
Module for mapping annotations to and from prodigy data structure
"""

import os
from typing import Dict, Optional

from ..datapoint import Image, BoundingBox, ImageAnnotation
from ..utils.detection_types import JsonDict
from .utils import MappingContextManager, maybe_get_fake_score, cur

__all__ = ["prodigy_to_image", "image_to_prodigy"]

_PRODIGY_IMAGE_PREFIX = "data:image/png;base64,"


@cur  # type: ignore
def prodigy_to_image(
    dp: JsonDict,
    categories_name_as_key: Dict[str, str],
    load_image: bool,
    fake_score: bool,
    path_reference_ds: str = "",
) -> Optional[Image]:
    """
    Map a datapoint of annotation structure as given as from Prodigy database to an Image
    structure.

    :param dp: A datapoint in dict structure as returned from Prodigy database
    :param categories_name_as_key: A dict of categories, e.g. DatasetCategories.get_categories(name_as_key=True)
    :param load_image: If 'True' it will load image to attr:`Image.image`
    :param fake_score: If dp does not contain a score, a fake score with uniform random variables in (0,1)
                       will be added.
    :param path_reference_ds: A path to a reference-dataset. It must point to the basedir where the file
                              of the datapoint can be found.
    :return: Image
    """

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
        location = dp.get("path") if dp.get("path") else os.path.join(path_reference_ds, file_name)
        if not os.path.isfile(location):  # type: ignore
            location = None

        image = Image(file_name=file_name, location=location, external_id=external_id)  # type: ignore

        if dp.get("image"):
            image.image = dp["image"].split(",")[1]
            if not load_image:
                image.clear_image()
        elif "width" in dp and "height" in dp:
            image.set_width_height(dp["width"], dp["height"])

        spans = dp.get("spans", [])

        for span in spans:
            upper_left = list(map(float, span["points"][0]))
            lower_right = list(map(float, span["points"][2]))
            bbox = BoundingBox(
                absolute_coords=True, ulx=upper_left[0], uly=upper_left[1], lrx=lower_right[0], lry=lower_right[1]
            )
            external_id = span.get("annotation_id")
            if not external_id:
                external_id = span.get("id")

            score = span.get("score")

            if not score:
                score = maybe_get_fake_score(fake_score)

            annotation = ImageAnnotation(
                category_name=span["label"],
                bounding_box=bbox,
                category_id=categories_name_as_key[span["label"]],
                score=score,
                external_id=external_id,
            )
            image.dump(annotation)

    if mapping_context.context_error:
        return None
    return image


def image_to_prodigy(dp: Image) -> JsonDict:
    """
    The mapper to transform the normalized image representation of datasets into the format
    for visualising the annotation components in Prodigy.

    :param dp: An image
    :return: A dictionary with compulsory keys: "text" and "spans"
    """

    assert isinstance(dp, Image), f"datapoints must be of type Image, is of type {type(dp)}"
    output: JsonDict = {}
    img_str = dp.get_image(type_id="b64")
    if img_str is None:
        img_str = ""
    output["image"] = _PRODIGY_IMAGE_PREFIX + img_str
    output["text"] = dp.file_name
    output["image_id"] = dp.image_id

    spans = []
    for ann in dp.get_annotation_iter():
        assert isinstance(ann.bounding_box, BoundingBox)
        box: JsonDict = {"label": ann.category_name, "annotation_id": ann.annotation_id}
        if ann.score is not None:
            box["score"] = float(round(ann.score, 3))
        box["type"] = "rect"
        if ann.image is not None:
            bounding_box = ann.image.get_embedding(dp.image_id)
        else:
            bounding_box = ann.bounding_box

        boxes = [
            [bounding_box.ulx, bounding_box.uly],
            [bounding_box.ulx, bounding_box.lry],
            [bounding_box.lrx, bounding_box.lry],
            [bounding_box.lrx, bounding_box.uly],
        ]

        box["points"] = [list(map(float, box)) for box in boxes]

        spans.append(box)
    output["spans"] = spans

    return output
