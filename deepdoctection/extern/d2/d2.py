# -*- coding: utf-8 -*-
# File: d2.py

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
Module for inference on D2 model
"""

from typing import Dict, List

from ...utils.detection_types import ImageType
from ...utils.file_utils import detectron2_available, pytorch_available
from ..base import DetectionResult
from ..common import InferenceResize

if pytorch_available():
    import torch
    from torch import nn  # pylint: disable=W0611

if detectron2_available():
    from detectron2.layers import batched_nms
    from detectron2.structures import Instances  # pylint: disable=W0611


def _d2_post_processing(
    predictions: Dict[str, "Instances"], nms_thresh_class_agnostic: float
) -> Dict[str, "Instances"]:
    """
    D2 postprocessing steps, so that detection outputs are aligned with outputs of other packages (e.g. Tensorpack).
    Apply a class agnostic NMS.

    :param predictions: Prediction outputs from the model.
    :param nms_thresh_class_agnostic: Nms being performed over all class predictions
    :return: filtered predictions outputs
    """
    instances = predictions["instances"]
    class_masks = torch.ones(instances.pred_classes.shape, dtype=torch.uint8)
    keep = batched_nms(instances.pred_boxes.tensor, instances.scores, class_masks, nms_thresh_class_agnostic)
    fg_instances_keep = instances[keep]
    return {"instances": fg_instances_keep}


def d2_predict_image(
    np_img: ImageType,
    predictor: "nn.Module",
    preproc_short_edge_size: int,
    preproc_max_size: int,
    nms_thresh_class_agnostic: float,
) -> List[DetectionResult]:
    """
    Run detection on one image, using the D2 model callable. It will also handle the preprocessing internally which
    is using a custom resizing within some bounds.

    :param np_img: ndarray
    :param predictor: torch nn module implemented in Detectron2
    :param preproc_short_edge_size: the short edge to resize to
    :param preproc_max_size: upper bound of one edge when resizing
    :param nms_thresh_class_agnostic: class agnostic nms threshold
    :return: list of DetectionResult
    """
    height, width = np_img.shape[:2]
    resizer = InferenceResize(preproc_short_edge_size, preproc_max_size)
    resized_img = resizer.get_transform(np_img).apply_image(np_img)
    image = torch.as_tensor(resized_img.astype("float32").transpose(2, 0, 1))

    with torch.no_grad():
        inputs = {"image": image, "height": height, "width": width}
        predictions = predictor([inputs])[0]
        predictions = _d2_post_processing(predictions, nms_thresh_class_agnostic)
    instances = predictions["instances"]
    results = [
        DetectionResult(
            instances[k].pred_boxes.tensor.tolist()[0],
            instances[k].scores.tolist()[0],
            instances[k].pred_classes.tolist()[0],
        )
        for k in range(len(instances))
    ]
    return results
