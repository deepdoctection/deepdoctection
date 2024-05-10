# -*- coding: utf-8 -*-
# File: common.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/common.py>
"""


import numpy as np
from lazy_imports import try_import

with try_import() as import_guard:
    from tensorpack.dataflow.imgaug import ImageAugmentor, ResizeTransform  # pylint: disable=E0401

with try_import() as cc_import_guard:
    import pycocotools.mask as coco_mask

if not import_guard.is_successful():
    from ....utils.mocks import ImageAugmentor


class CustomResize(ImageAugmentor):
    """
    Try resizing the shortest edge to a certain number while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, max_size, interp=1):
        """
        :param short_edge_length: a [min, max] interval from which to sample the shortest edge length.
        :param max_size: maximum allowed longest edge length.
        :param interp: Interpolation mode. We use Tensorpack's internal `ResizeTransform`, that always requires OpenCV
        """
        super().__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self.short_edge_length = short_edge_length
        self.max_size = max_size
        self.interp = interp

    def get_transform(self, img):
        """
        get transform
        """
        h, w = img.shape[:2]
        size = self.rng.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size
        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
            new_h = new_h * scale
            new_w = new_w * scale
        new_w = int(new_w + 0.5)
        new_h = int(new_h + 0.5)
        return ResizeTransform(h, w, new_h, new_w, self.interp)


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

    ret = coco_mask.iou(to_xywh(box_a), to_xywh(box_b), np.zeros((len(box_b),), dtype=bool))
    # can accelerate even more, if using float32
    return ret.astype("float32")
