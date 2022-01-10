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

from typing import List

import numpy as np

import torch
from torch import nn

from ..common import CustomResize
from ..base import DetectionResult


def d2_predict_image(np_img: np.ndarray,predictor: nn.Module,  preproc_short_edge_size: int,
                     preproc_max_size: int) -> List[DetectionResult]:
    """
    Run detection on one image, using the D2 model callable. It will also handle the preprocessing internally which
    is using a custom resizing within some bounds.

    :param np_img: ndarray
    :param predictor: torch nn module implemented in Detectron2
    :param preproc_short_edge_size: the short edge to resize to
    :param preproc_max_size: upper bound of one edge when resizing
    :return: list of DetectionResult
    """
    height, width = np_img.shape[:2]
    resizer = CustomResize(preproc_short_edge_size, preproc_max_size)
    resized_img = resizer.augment(np_img)
    image = torch.as_tensor(resized_img.astype("float32").transpose(2,0,1))
    inputs = {"image": image, "height": height, "width": width}
    predictions = predictor([inputs])[0]
    results = [DetectionResult(predictions.boxes[k],
                               predictions.scores[k],
                               predictions.pred_classes[k]) for k in range(len(predictions))]
    return results
