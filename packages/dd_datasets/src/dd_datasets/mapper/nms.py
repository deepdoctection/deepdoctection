# -*- coding: utf-8 -*-
# File: nms.py

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
Module for NMS (Non-Maximum Suppression) operations on image annotations.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from lazy_imports import try_import

from dd_datapoint.datapoint.annotation import ImageAnnotation

with try_import() as pt_import_guard:
    import torch
    from torchvision.ops import boxes as box_ops  # type: ignore


__all__ = ["batched_nms", "pt_nms_image_annotations", "pt_nms_image_annotations_depr"]


def batched_nms(boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Same as `torchvision.ops.boxes.batched_nms`, but with `float()`.

    Args:
        boxes: A `torch.Tensor` of shape (N, 4) containing bounding boxes.
        scores: A `torch.Tensor` of shape (N,) containing scores for each box.
        idxs: A `torch.Tensor` of shape (N,) containing the class indices for each box.
        iou_threshold: A float representing the IoU threshold for suppression.

    Returns:
        A `torch.Tensor` containing the indices of the boxes to keep.

    Note:
        `Fp16` does not have enough range for batched NMS, so `float()` is used.
        Torchvision already has a strategy to decide whether to use coordinate trick or for loop to implement
        `batched_nms`.
    """
    assert boxes.shape[-1] == 4
    # Note: Torchvision already has a strategy (https://github.com/pytorch/vision/issues/1311)
    # to decide whether to use coordinate trick or for loop to implement batched_nms. So we
    # just call it directly.
    # Fp16 does not have enough range for batched NMS, so adding float().
    return box_ops.batched_nms(boxes.float(), scores, idxs, iou_threshold)


def pt_nms_image_annotations_depr(
    anns: Sequence[ImageAnnotation], threshold: float, image_id: Optional[str] = None, prio: str = ""
) -> Sequence[str]:
    """
    Processing given image annotations through NMS. This is useful, if you want to supress some specific image
    annotation, e.g. given by name or returned through different predictors. This is the pt version, for tf check
    `mapper.tpstruct`

    Args:
        anns: A sequence of ImageAnnotations. All annotations will be treated as if they belong to one category
        threshold: NMS threshold
        image_id: id in order to get the embedding bounding box
        prio: If an annotation has prio, it will overwrite its given score to 1 so that it will never be suppressed

    Returns:
        A list of `annotation_id`s that belong to the given input sequence and that survive the NMS process
    """
    if len(anns) == 1:
        return [anns[0].annotation_id]

    # if all annotations are the same and prio is set, we need to return all annotation ids
    if prio and all(ann.category_name == anns[0].category_name for ann in anns):
        return [ann.annotation_id for ann in anns]

    if not anns:
        return []
    ann_ids = np.array([ann.annotation_id for ann in anns], dtype="object")
    # safety net to ensure we do not run into a ValueError
    boxes = torch.tensor(
        [ann.get_bounding_box(image_id).to_list(mode="xyxy") for ann in anns if ann.bounding_box is not None]
    )

    def priority_to_confidence(ann: ImageAnnotation, priority: str) -> float:
        if ann.category_name == priority:
            return 1.0
        if ann.score:
            return ann.score
        raise ValueError("score cannot be None")

    scores = torch.tensor([priority_to_confidence(ann, prio) for ann in anns])
    class_mask = torch.ones(len(boxes), dtype=torch.uint8)
    keep = batched_nms(boxes, scores, class_mask, threshold)
    ann_ids_keep = ann_ids[keep]
    if not isinstance(ann_ids_keep, str):
        return ann_ids_keep.tolist()
    return []


def pt_nms_image_annotations(
    anns: Sequence[ImageAnnotation], threshold: float, image_id: Optional[str] = None, prio: str = ""
) -> Sequence[str]:
    """
    Processes given image annotations through NMS (Non-Maximum Suppression). Useful for suppressing specific image
    annotations, e.g., given by name or returned through different predictors. This is the pt version, for tf check
    `mapper.tpstruct`

    Args:
        anns: A sequence of `ImageAnnotation`. All annotations will be treated as if they belong to one category.
        threshold: NMS threshold.
        image_id: ID to get the embedding bounding box.
        prio: If an annotation has priority, its score will be set to 1 so that it will never be suppressed.

    Returns:
        A list of `annotation_id` that belong to the given input sequence and that survive the NMS process.
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
    boxes = torch.tensor(
        [ann.get_bounding_box(image_id).to_list(mode="xyxy") for ann in anns if ann.bounding_box is not None]
    )

    scores = torch.tensor([priority_to_confidence(ann, prio) for ann in anns])
    class_mask = torch.ones(len(boxes), dtype=torch.uint8)

    keep = batched_nms(boxes, scores, class_mask, threshold)
    kept_ids = ann_ids[keep]

    # Convert to list if necessary
    if isinstance(kept_ids, str):
        kept_ids = [kept_ids]
    elif not isinstance(kept_ids, list):
        kept_ids = kept_ids.tolist()

    # Combine priority annotations with surviving non-priority annotations
    return list(set(priority_ann_ids + kept_ids))

