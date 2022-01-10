# -*- coding: utf-8 -*-
# File: common.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/common.py
"""


import cv2
import numpy as np
import pycocotools.mask as coco_mask

from tensorpack.dataflow.imgaug import ImageAugmentor, ResizeTransform


def polygons_to_mask(polys, height, width, intersect=False):
    """
    Convert polygons to binary masks.

    :param polys: a list of nx2 float array. Each array contains many (x, y) coordinates.#
    :param intersect: intersect
    :param width: width
    :param height: height
    :return: A binary matrix of (height, width)
    """

    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    rles = coco_mask.frPyObjects(polys, height, width)
    rle = coco_mask.merge(rles, intersect)
    return coco_mask.decode(rle)


def clip_boxes(boxes, shape):
    """
    :param boxes: (...)x4, float
    :param shape: h, w
    """

    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def filter_boxes_inside_shape(boxes, shape):
    """
    :param boxes: (nx4), float
    :param shape: (h, w)

    :return: indices: (k, )
            selection: (kx4)
    """

    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where((boxes[:, 0] >= 0) & (boxes[:, 1] >= 0) & (boxes[:, 2] <= w) & (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]


# Much faster than utils/np_box_ops
def np_iou(box_a, box_b):
    """
    np iou
    """

    def to_xywh(box):
        box = box.copy()
        box[:, 2] -= box[:, 0]
        box[:, 3] -= box[:, 1]
        return box

    ret = coco_mask.iou(to_xywh(box_a), to_xywh(box_b), np.zeros((len(box_b),), dtype=np.bool))
    # can accelerate even more, if using float32
    return ret.astype("float32")
