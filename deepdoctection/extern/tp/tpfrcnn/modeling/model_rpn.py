# -*- coding: utf-8 -*-
# File: model_rpn.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_rpn.py>
"""

import numpy as np
from lazy_imports import try_import

from .model_box import clip_boxes

with try_import() as import_guard:
    # pylint: disable=import-error
    import tensorflow as tf
    from tensorpack import tfv1
    from tensorpack.models import Conv2D, layer_register
    from tensorpack.tfutils.argscope import argscope
    from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope, under_name_scope
    from tensorpack.tfutils.summary import add_moving_summary
    from tensorpack.utils.argtools import memoized

    # pylint: enable=import-error

if not import_guard.is_successful():
    from .....utils.mocks import auto_reuse_variable_scope, layer_register, memoized, under_name_scope


@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(feature_map, channel, num_anchors):
    """
    RPN head

    :return: label_logits: fHxfWxNA
             box_logits: fHxfWxNAx4
    """

    with argscope(Conv2D, data_format="channels_first", kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
        hidden = Conv2D("conv0", feature_map, channel, 3, activation=tf.nn.relu)

        label_logits = Conv2D("class", hidden, num_anchors, 1)
        box_logits = Conv2D("box", hidden, 4 * num_anchors, 1)
        # 1, NA(*4), im/16, im/16 (NCHW)

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], num_anchors, 4]))  # fHxfWxNAx4
    return label_logits, box_logits


@under_name_scope()
def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits, rpn_batch_per_im):
    """
    RPN losses

    :param anchor_labels: fHxfWxNA
    :param anchor_boxes: fHxfWxNAx4, encoded
    :param label_logits: fHxfWxNA
    :param box_logits: fHxfWxNAx4
    :param rpn_batch_per_im: RPN batch per image

    :return: label_loss, box_loss
    """

    with tf.device("/cpu:0"):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
        nr_valid = tf.stop_gradient(tfv1.count_nonzero(valid_mask, dtype=tf.int32), name="num_valid_anchor")
        nr_pos = tf.identity(tfv1.count_nonzero(pos_mask, dtype=tf.int32), name="num_pos_anchor")
        # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.

        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    with tf.name_scope("label_metrics"):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        summaries = []
        with tf.device("/cpu:0"):
            for thresh in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > thresh, tf.int32)
                nr_pos_prediction = tf.reduce_sum(valid_prediction, name="num_pos_prediction")
                pos_prediction_corr = tfv1.count_nonzero(
                    tf.logical_and(valid_label_prob > thresh, tf.equal(valid_prediction, valid_anchor_labels)),
                    dtype=tf.int32,
                )
                placeholder = 0.5  # A small value will make summaries appear lower.
                recall = tf.cast(tf.truediv(pos_prediction_corr, nr_pos), tf.float32)
                recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name=f"recall_th{thresh}")
                precision = tf.cast(tf.truediv(pos_prediction_corr, nr_pos_prediction), tf.float32)
                precision = tf.where(
                    tf.equal(nr_pos_prediction, 0), placeholder, precision, name=f"precision_th{thresh}"
                )
                summaries.extend([precision, recall])
        add_moving_summary(*summaries)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total RPN loss will be fine.  TODO make the summary op smarter
    placeholder = 0.0
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(valid_anchor_labels, tf.float32), logits=valid_label_logits
    )
    label_loss = tf.reduce_sum(label_loss) * (1.0 / rpn_batch_per_im)
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name="label_loss")

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = (
        tfv1.losses.huber_loss(pos_anchor_boxes, pos_box_logits, delta=delta, reduction=tfv1.losses.Reduction.SUM)
        / delta
    )
    box_loss = box_loss * (1.0 / rpn_batch_per_im)
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name="box_loss")

    add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return [label_loss, box_loss]


@under_name_scope()
def generate_rpn_proposals(
    boxes, scores, img_shape, rpn_min_size, rpn_proposal_nms_thres, pre_nms_top_k, post_nms_top_k=None
):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    :param post_nms_top_k: (int): See above.
    :param pre_nms_top_k: (int): See above.
    :param img_shape: [h, w]
    :param scores: n float, the logits
    :param boxes: nx4 float dtype, the proposal boxes. Decoded to float box already

    :return: boxes: kx4 float
            scores: k logits

    """
    assert boxes.shape.ndims == 2, boxes.shape
    if post_nms_top_k is None:
        post_nms_top_k = pre_nms_top_k

    top_k = tf.minimum(pre_nms_top_k, tf.size(scores))
    top_k_scores, top_k_indices = tf.nn.top_k(scores, k=top_k, sorted=False)
    top_k_boxes = tf.gather(boxes, top_k_indices)
    top_k_boxes = clip_boxes(top_k_boxes, img_shape)

    if rpn_min_size > 0:
        top_k_boxes_x1y1x2y2 = tf.reshape(top_k_boxes, (-1, 2, 2))
        top_k_boxes_x1y1, top_k_boxes_x2y2 = tf.split(top_k_boxes_x1y1x2y2, 2, axis=1)
        # nx1x2 each
        wbhb = tf.squeeze(top_k_boxes_x2y2 - top_k_boxes_x1y1, axis=1)
        valid = tf.reduce_all(wbhb > rpn_min_size, axis=1)  # n,
        top_k_valid_boxes = tf.boolean_mask(top_k_boxes, valid)
        top_k_valid_scores = tf.boolean_mask(top_k_scores, valid)
    else:
        top_k_valid_boxes = top_k_boxes
        top_k_valid_scores = top_k_scores

    nms_indices = tf.image.non_max_suppression(
        top_k_valid_boxes, top_k_valid_scores, max_output_size=post_nms_top_k, iou_threshold=rpn_proposal_nms_thres
    )

    proposal_boxes = tf.gather(top_k_valid_boxes, nms_indices)
    proposal_scores = tf.gather(top_k_valid_scores, nms_indices)
    tf.sigmoid(proposal_scores, name="probs")  # for visualization
    return tf.stop_gradient(proposal_boxes, name="boxes"), tf.stop_gradient(proposal_scores, name="scores")


@memoized
def get_all_anchors(*, stride, sizes, ratios, max_size):
    """
    Get all anchors in the largest possible image, shifted, float box

    :param stride: the stride of anchors.
    :param sizes: the sizes (sqrt area) of anchors
    :param ratios: the aspect ratios of anchors
    :param max_size: maximum size of input image

    :return: SxSxNUM_ANCHOR x 4, where S == ceil(MAX_SIZE/STRIDE), float box

    The layout in the NUM_ANCHOR dim is NUM_RATIO x NUM_SIZE.
    """
    # Generates a NAx4 matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    # are centered on 0, have sqrt areas equal to the specified sizes, and aspect ratios as given.
    anchors = []
    for size in sizes:
        for ratio in ratios:
            w = np.sqrt(size * size / ratio)
            h = ratio * w
            anchors.append([-w, -h, w, h])
    cell_anchors = np.asarray(anchors) * 0.5

    field_size = int(np.ceil(max_size / stride))
    shifts = (np.arange(0, field_size) * stride).astype("float32")
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # Kx4, K = field_size * field_size
    K = shifts.shape[0]  # pylint: disable =C0103

    A = cell_anchors.shape[0]  # pylint: disable =C0103
    field_of_anchors = cell_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    field_of_anchors = field_of_anchors.reshape((field_size, field_size, A, 4))
    # FSxFSxAx4
    # Many rounding happens inside the anchor code anyway
    # assert np.all(field_of_anchors == field_of_anchors.astype('int32'))
    field_of_anchors = field_of_anchors.astype("float32")
    return field_of_anchors
