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
Module for mapping annotations into standard Detectron2 dataset dict
"""


import os.path
from typing import Dict, List, Optional, Sequence, Union

from detectron2.structures import BoxMode

from ..datapoint.image import Image
from ..mapper.maputils import curry
from ..utils.detection_types import JsonDict
from ..utils.settings import ObjectTypes


@curry
def image_to_d2_frcnn_training(
    dp: Image,
    add_mask: bool = False,
    category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
) -> Optional[JsonDict]:
    """
    Maps an image to a standard dataset dict as described in
    https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html. It further checks if the image is physically
    available, for otherwise the annotation will be filtered.
    Note, that the returned dict will not suffice for training as gt for RPN and anchors still need to be created.

    :param dp: Image
    :param add_mask: True is not implemented (yet).
    :param category_names: A list of category names for training a model. Pass nothing to train with all annotations
    :return: Dict with 'image', 'width', 'height', 'image_id', 'annotations' where 'annotations' is a list of dict
             with 'bbox_mode' (D2 internal bounding box description), 'bbox' and 'category_id'.
    """
    if not os.path.isfile(dp.location) and dp.image is None:
        return None

    output: JsonDict = {"file_name": dp.location}

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
        if ann.bounding_box is None:
            raise ValueError("BoundingBox cannot be None")
        mapped_ann: Dict[str, Union[str, int, List[float]]] = {
            "bbox_mode": BoxMode.XYXY_ABS,
            "bbox": ann.bounding_box.to_list(mode="xyxy"),
            "category_id": int(ann.category_id) - 1,
        }
        annotations.append(mapped_ann)

        if add_mask:
            raise NotImplementedError

    output["annotations"] = annotations

    return output
