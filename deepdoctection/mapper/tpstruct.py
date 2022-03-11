# -*- coding: utf-8 -*-
# File: tpstruct.py

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
Module for mapping annotation for training environments
"""
import os.path
from typing import Optional

import numpy as np

from ..datapoint.image import Image
from ..utils.detection_types import JsonDict
from .maputils import cur


@cur  # type: ignore
def image_to_tp_frcnn_training(dp: Image, add_mask: bool = False) -> Optional[JsonDict]:
    """
    Maps an image to a dict to be consumed by Tensorpack Faster-RCNN bounding box detection. Note, that the returned
    will not suffice for training as gt for RPN and anchors still need to be created.

    :param dp: Image
    :param add_mask: True is not implemented (yet).
    :return: Dict with 'image', 'gt_boxes', 'gt_labels' and 'file_name', provided there are some detected objects in the
             image
    """

    output: JsonDict = {}
    if dp.image is not None:
        output["image"] = dp.image.astype("float32")
    anns = dp.get_annotation()
    all_boxes = []
    all_categories = []
    if not anns:
        return None

    for ann in anns:
        assert ann.bounding_box is not None
        all_boxes.append(ann.bounding_box.to_list(mode="xyxy"))
        all_categories.append(ann.category_id)

        if add_mask:
            raise NotImplementedError

    output["gt_boxes"] = np.asarray(all_boxes, dtype="float32")
    output["gt_labels"] = np.asarray(all_categories, dtype="int32")
    if not os.path.isfile(dp.location) and dp.image is None:
        return None

    output["file_name"] = dp.location  # full path

    return output
