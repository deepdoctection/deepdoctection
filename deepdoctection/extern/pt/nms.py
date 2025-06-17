# -*- coding: utf-8 -*-
# File: nms.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Module for custom NMS functions.
"""
from __future__ import annotations

from lazy_imports import try_import

with try_import() as import_guard:
    import torch
    from torchvision.ops import boxes as box_ops  # type: ignore


# Copy & paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/nms.py
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
