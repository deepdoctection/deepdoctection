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
from typing import Optional, Sequence, Union

import numpy as np
from lazy_imports import try_import

from ..datapoint.annotation import ImageAnnotation
from ..datapoint.image import Image
from ..utils.settings import TypeOrStr
from ..utils.types import JsonDict
from .maputils import curry

with try_import() as import_guard:
    from tensorflow import convert_to_tensor, uint8  # type: ignore # pylint: disable=E0401
    from tensorflow.image import non_max_suppression  # type: ignore # pylint: disable=E0401


@curry
def image_to_tp_frcnn_training(
    dp: Image,
    add_mask: bool = False,
    category_names: Optional[Union[TypeOrStr, Sequence[TypeOrStr]]] = None,
) -> Optional[JsonDict]:
    """
    Maps an image to a dict to be consumed by Tensorpack Faster-RCNN bounding box detection. Note, that the returned
    dict will not suffice for training as gt for RPN and anchors still need to be created.

    :param dp: Image
    :param add_mask: True is not implemented (yet).
    :param category_names: A list of category names for training a model. Pass nothing to train with all annotations
    :return: Dict with `image`, `gt_boxes`, `gt_labels` and `file_name`, provided there are some detected objects in the
             image
    """

    output: JsonDict = {}
    if dp.image is not None:
        output["image"] = dp.image.astype("float32")
    anns = dp.get_annotation(category_names=category_names)
    all_boxes = []
    all_categories = []
    if not anns:
        return None

    for ann in anns:
        box = ann.get_bounding_box(dp.image_id)
        all_boxes.append(box.to_list(mode="xyxy"))
        all_categories.append(ann.category_id)

        if add_mask:
            raise NotImplementedError()

    output["gt_boxes"] = np.asarray(all_boxes, dtype="float32")
    output["gt_labels"] = np.asarray(all_categories, dtype="int32")
    if not os.path.isfile(dp.location) and dp.image is None:
        return None

    output["file_name"] = dp.location  # full path

    return output


def tf_nms_image_annotations(
    anns: Sequence[ImageAnnotation], threshold: float, image_id: Optional[str] = None, prio: str = ""
) -> Sequence[str]:
    """
    Processing given image annotations through NMS. This is useful, if you want to supress some specific image
    annotation, e.g. given by name or returned through different predictors. This is the tf version, for pt check
    `mapper.d2struct`

    :param anns: A sequence of ImageAnnotations. All annotations will be treated as if they belong to one category
    :param threshold: NMS threshold
    :param image_id: id in order to get the embedding bounding box
    :param prio: If an annotation has prio, it will overwrite its given score to 1 so that it will never be suppressed
    :return: A list of annotation_ids that belong to the given input sequence and that survive the NMS process
    """
    if len(anns) == 1:
        return [anns[0].annotation_id]
    if not anns:
        return []
    ann_ids = np.array([ann.annotation_id for ann in anns], dtype="object")

    boxes = convert_to_tensor([ann.get_bounding_box(image_id).to_list(mode="xyxy") for ann in anns])

    def priority_to_confidence(ann: ImageAnnotation, priority: str) -> float:
        if ann.category_name == priority:
            return 1.0
        if ann.score:
            return ann.score
        raise ValueError("score cannot be None")

    scores = convert_to_tensor([priority_to_confidence(ann, prio) for ann in anns])
    class_mask = convert_to_tensor(len(boxes), dtype=uint8)
    keep = non_max_suppression(boxes, scores, class_mask, iou_threshold=threshold)
    ann_ids_keep = ann_ids[keep]
    if not isinstance(ann_ids_keep, str):
        return ann_ids_keep.tolist()
    return []
