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
    Maps an `Image` to a dict to be consumed by Tensorpack Faster-RCNN bounding box detection.

    Note:
        The returned dict will not suffice for training as ground truth for RPN and anchors still need to be created.

    Args:
        dp: `Image`.
        add_mask: `True` is not implemented.
        category_names: A list of category names for training a model. Pass nothing to train with all annotations.

    Returns:
        Dict with `image`, `gt_boxes`, `gt_labels` and `file_name`, provided there are some detected objects in the
        image.

    Example:
        ```python
        image_to_tp_frcnn_training(dp)
        ```
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
    Processes given `ImageAnnotation` through `NMS`.

    This is useful if you want to suppress some specific image annotation, e.g., given by name or returned through
     different predictors. This is the TensorFlow version; for PyTorch, check `mapper.d2struct`.

    Args:
        anns: A sequence of `ImageAnnotation`. All annotations will be treated as if they belong to one category.
        threshold: NMS threshold.
        image_id: ID in order to get the embedding bounding box.
        prio: If an annotation has `prio`, it will overwrite its given score to 1 so that it will never be suppressed.

    Returns:
        A list of `annotation_id` that belong to the given input sequence and that survive the NMS process.

    Example:
        ```python
        tf_nms_image_annotations(anns, threshold)
        ```
    """
    if len(anns) == 1:
        return [anns[0].annotation_id]

    if not anns:
        return []

    # First, identify priority annotations that should always be kept
    priority_ann_ids = []

    if prio:
        for ann in anns:
            if ann.category_name == prio:
                priority_ann_ids.append(ann.annotation_id)

    # If all annotations are priority or none are left for NMS, return all priority IDs
    if len(priority_ann_ids) == len(anns):
        return priority_ann_ids

    def priority_to_confidence(ann: ImageAnnotation, priority: str) -> float:
        if ann.category_name == priority:
            return 1.0
        if ann.score:
            return ann.score
        raise ValueError("score cannot be None")

    # Perform NMS only on non-priority annotations
    ann_ids = np.array([ann.annotation_id for ann in anns], dtype="object")

    # Get boxes for non-priority annotations
    boxes = convert_to_tensor(
        [ann.get_bounding_box(image_id).to_list(mode="xyxy") for ann in anns if ann.bounding_box is not None]
    )

    scores = convert_to_tensor([priority_to_confidence(ann, prio) for ann in anns])
    class_mask = convert_to_tensor(len(boxes), dtype=uint8)

    keep = non_max_suppression(boxes, scores, class_mask, iou_threshold=threshold)
    kept_ids = ann_ids[keep]

    # Convert to list if necessary
    if isinstance(kept_ids, str):
        kept_ids = [kept_ids]
    elif not isinstance(kept_ids, list):
        kept_ids = kept_ids.tolist()

    # Combine priority annotations with surviving non-priority annotations
    return list(set(priority_ann_ids + kept_ids))
