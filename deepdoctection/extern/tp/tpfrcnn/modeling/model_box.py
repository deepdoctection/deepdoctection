# -*- coding: utf-8 -*-
# File: model_box.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_box.py>
"""
from collections import namedtuple

import numpy as np
from lazy_imports import try_import

with try_import() as import_guard:
    # pylint: disable=import-error
    import tensorflow as tf
    from tensorpack.tfutils.scope_utils import under_name_scope

    # pylint: enable=import-error

if not import_guard.is_successful():
    from .....utils.mocks import under_name_scope


@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    clip boxes

    :param boxes: nx4, xyxy
    :param window: [h, w]
    :param name: (str)
    :return:
    """
    boxes = tf.maximum(boxes, 0.0)
    mat = tf.tile(tf.reverse(window, [0]), [2])  # (4,)
    boxes = tf.minimum(boxes, tf.cast(mat, tf.float32), name=name)
    return boxes


@under_name_scope()
def decode_bbox_target(box_predictions, anchors, preproc_max_size):
    """
    Decode bbox target

    :param box_predictions: (..., 4), logits
    :param anchors: (..., 4), float box. Must have the same shape
    :param preproc_max_size: int
    :return: (..., 4), float32. With the same shape.
    """

    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = np.log(preproc_max_size / 16.0)
    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5  # (...)x1x2
    out = tf.concat([x1y1, x2y2], axis=-2)  # pylint: disable=E1123
    return tf.reshape(out, orig_shape)


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Encode bbox target

    :param boxes: (..., 4), float32
    :param anchors: (..., 4), float32
    :return: (..., 4), float32 with the same shape.
    """

    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    txty = (xbyb - xaya) / waha
    twth = tf.math.log(wbhb / waha)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)  # pylint: disable=E1123
    return tf.reshape(encoded, tf.shape(boxes))


@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Crop and resize

    :param image: NCHW
    :param boxes: nx4, x1y1x2y2
    :param box_ind: (n,)
    :param crop_size: (int)
    :param pad_border:  bool
    :return:   n,C,size,size
    """

    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode="SYMMETRIC")
        boxes += 1

    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bi linear sample.

        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bi linear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        :param boxes:  nx4, x1y1x2y2
        :param image_shape: shape
        :param crop_shape: crop shape
        :return:  y1x1y2x2
        """

        x_0, y_0, x_1, y_1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x_1 - x_0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y_1 - y_0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x_0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y_0 + spacing_h / 2 - 0.5) / imshape[0]

        n_w = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        n_h = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + n_h, nx0 + n_w], axis=1)  # pylint: disable=E1123

    image_shape = tf.shape(image)[2:]

    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])  # nhwc
    ret = tf.image.crop_and_resize(image, boxes, tf.cast(box_ind, tf.int32), crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])  # ncss
    return ret


@under_name_scope()
def roi_align(feature_map, boxes, resolution):
    """
    Roi align

    :param feature_map: 1xCxHxW
    :param boxes:  Nx4 float box
    :param resolution: output spatial resolution
    :return: Roi aligned tf.Tensor
    """

    # sample 4 locations per roi bin
    ret = crop_and_resize(feature_map, boxes, tf.zeros(tf.shape(boxes)[0], dtype=tf.int32), resolution * 2)
    try:
        avg_pool = tf.nn.avg_pool2d
    except AttributeError:
        avg_pool = tf.nn.avg_pool
    ret = avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding="SAME", data_format="NCHW")
    return ret


class RPNAnchors(namedtuple("_RPNAnchors", ["boxes", "gt_labels", "gt_boxes"])):
    """
    boxes (FS x FS x NA x 4): The anchor boxes.
    gt_labels (FS x FS x NA):
    gt_boxes (FS x FS x NA x 4): Ground-truth boxes corresponding to each anchor.
    """

    def encoded_gt_boxes(self):
        """
        encoded ground truth boxes
        """
        return encode_bbox_target(self.gt_boxes, self.boxes)

    def decode_logits(self, logits, preproc_max_size):
        """
        Decode logits

        :param logits: logits
        :param preproc_max_size: preprocess to max size
        """
        return decode_bbox_target(logits, self.boxes, preproc_max_size)

    @under_name_scope()
    def narrow_to(self, featuremap):
        """
        Slice anchors to the spatial size of this feature map.
        """
        shape2d = tf.shape(featuremap)[2:]  # h,w
        slice3d = tf.concat([shape2d, [-1]], axis=0)  # pylint: disable=E1123
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)  # pylint: disable=E1123
        boxes = tf.slice(self.boxes, [0, 0, 0, 0], slice4d)
        gt_labels = tf.slice(self.gt_labels, [0, 0, 0], slice3d)
        gt_boxes = tf.slice(self.gt_boxes, [0, 0, 0, 0], slice4d)
        return RPNAnchors(boxes, gt_labels, gt_boxes)
