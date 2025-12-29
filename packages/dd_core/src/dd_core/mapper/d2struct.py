# -*- coding: utf-8 -*-
# File: d2struct.py

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
Module for mapping annotations into standard Detectron2 dataset dict.
"""
from __future__ import annotations

import os.path
from typing import Optional, Sequence, Union

from lazy_imports import try_import

from ..datapoint.image import Image
from ..mapper.maputils import curry
from ..utils.object_types import TypeOrStr
from ..utils.types import Detectron2Dict

with try_import() as d2_import_guard:
    from detectron2.structures import BoxMode


@curry
def image_to_d2_frcnn_training(
    dp: Image,
    add_mask: bool = False,
    category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
) -> Optional[Detectron2Dict]:
    """
    Maps an image to a standard dataset dict as described in
    <https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html>. It further checks if the image is physically
    available, for otherwise the annotation will be filtered.
    Note, that the returned dict will not suffice for training as gt for RPN and anchors still need to be created.

    Args:
        dp: Image
        add_mask: `True` is not implemented (yet).
        category_names: A list of category names for training a model. Pass nothing to train with all annotations

    Returns:
        Dict with 'image', 'width', 'height', 'image_id', 'annotations' where 'annotations' is a list of dict
        with 'bbox_mode' (D2 internal bounding box description), 'bbox' and 'category_id'.
    """
    if not os.path.isfile(dp.location) and dp.image is None:
        return None

    output: Detectron2Dict = {"file_name": str(dp.location)}

    if dp.image is not None:
        output["image"] = dp.image.astype("float32")
    output["width"] = dp.width
    output["height"] = dp.height
    output["image_id"] = dp.image_id

    anns = dp.get_annotation(category_names=category_names)

    if not anns:
        return None

    annotations = []

    for ann in anns:
        box = ann.get_bounding_box(dp.image_id)
        if not box.absolute_coords:
            box = box.transform(dp.width, dp.height, absolute_coords=True)

        # Detectron2 does not fully support BoxMode.XYXY_REL
        mapped_ann: dict[str, Union[str, int, list[float]]] = {
            "bbox_mode": BoxMode.XYXY_ABS,
            "bbox": box.to_list(mode="xyxy"),
            "category_id": ann.category_id - 1,
        }
        annotations.append(mapped_ann)

        if add_mask:
            raise NotImplementedError("Segmentation in deepdoctection is not supported")

    output["annotations"] = annotations

    return output
