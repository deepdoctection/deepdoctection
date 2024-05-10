# -*- coding: utf-8 -*-
# File: model_rpn.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/predict.py>
"""
from __future__ import annotations

from typing import List

import numpy as np
from lazy_imports import try_import

from ....utils.transform import InferenceResize
from ...base import DetectionResult
from .common import clip_boxes

with try_import() as import_guard:
    from tensorpack.predict.base import OfflinePredictor  # pylint: disable=E0401

with try_import() as sp_import_guard:
    from scipy import interpolate

with try_import() as cv2_import_guard:
    import cv2


def _scale_box(box, scale):
    w_half = (box[2] - box[0]) * 0.5
    h_half = (box[3] - box[1]) * 0.5
    x_c = (box[2] + box[0]) * 0.5
    y_c = (box[3] + box[1]) * 0.5

    w_half *= scale
    h_half *= scale

    scaled_box = np.zeros_like(box)
    scaled_box[0] = x_c - w_half
    scaled_box[2] = x_c + w_half
    scaled_box[1] = y_c - h_half
    scaled_box[3] = y_c + h_half
    return scaled_box


def _paste_mask(box, mask, shape, mrcnn_accurate_paste):
    """
    paste mask

    :param box: 4 float
    :param mask: MxM floats
    :param shape: h,w
    :return A uint8 binary image of hxw.
    """

    assert mask.shape[0] == mask.shape[1], mask.shape

    if mrcnn_accurate_paste:
        # This method is accurate but much slower.
        mask = np.pad(mask, [(1, 1), (1, 1)])
        box = _scale_box(box, float(mask.shape[0]) / (mask.shape[0] - 2))

        mask_pixels = np.arange(0.0, mask.shape[0]) + 0.5
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        h, w = shape
        y_s = np.arange(0.0, h) + 0.5
        x_s = np.arange(0.0, w) + 0.5
        y_s = (y_s - box[1]) / (box[3] - box[1]) * mask.shape[0]
        x_s = (x_s - box[0]) / (box[2] - box[0]) * mask.shape[1]
        # Waste a lot of compute since most indices are out-of-border
        res = mask_continuous(x_s, y_s)
        return (res >= 0.5).astype("uint8")

    x_0, y_0 = list(map(int, box[:2] + 0.5))
    x_1, y_1 = list(map(int, box[2:] - 0.5))  # inclusive
    x_1 = max(x_0, x_1)  # require at least 1x1
    y_1 = max(y_0, y_1)

    w = x_1 + 1 - x_0
    h = y_1 + 1 - y_0

    # rounding errors could happen here, because masks were not originally computed for this shape.
    # but it's hard to do better, because the network does not know the "original" scale

    mask = (cv2.resize(mask, (w, h)) > 0.5).astype("uint8")
    ret = np.zeros(shape, dtype="uint8")
    ret[y_0 : y_1 + 1, x_0 : x_1 + 1] = mask
    return ret


def tp_predict_image(
    np_img: np.ndarray,
    predictor: OfflinePredictor,
    preproc_short_edge_size: int,
    preproc_max_size: int,
    mrcnn_accurate_paste: bool,
) -> List[DetectionResult]:
    """
    Run detection on one image, using the TF callable. This function should handle the preprocessing internally.

    :param np_img: ndarray
    :param predictor: A tensorpack predictor
    :param preproc_short_edge_size: the short edge to resize to
    :param preproc_max_size: upper bound of one edge when resizing
    :param mrcnn_accurate_paste: whether to paste accurately
    :return: list of DetectionResult
    """
    orig_shape = np_img.shape[:2]
    resizer = InferenceResize(preproc_short_edge_size, preproc_max_size)
    resized_img = resizer.get_transform(np_img).apply_image(np_img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / np_img.shape[0] * resized_img.shape[1] / np_img.shape[1])
    boxes, score, labels, *masks = predictor(resized_img)

    # Some slow numpy postprocessing:
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true anymore.
    boxes = clip_boxes(boxes, orig_shape)
    if masks:
        full_masks = [_paste_mask(box, mask, orig_shape, mrcnn_accurate_paste) for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)
    labels = labels.tolist()
    results = [
        DetectionResult(box=args[0], score=args[1], class_id=args[2]) for args in zip(boxes, score, labels, masks)
    ]
    return results
