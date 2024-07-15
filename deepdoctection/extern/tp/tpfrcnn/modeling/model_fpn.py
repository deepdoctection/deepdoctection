# -*- coding: utf-8 -*-
# File: model_fpn.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_fpn.py>
"""

import itertools

import numpy as np
from lazy_imports import try_import

from ..utils.box_ops import area as tf_area
from .backbone import GroupNorm
from .model_box import roi_align
from .model_rpn import generate_rpn_proposals, get_all_anchors, rpn_losses

with try_import() as import_guard:
    # pylint: disable=import-error
    import tensorflow as tf
    from tensorpack import tfv1
    from tensorpack.models import Conv2D, FixedUnPooling, MaxPooling, layer_register
    from tensorpack.tfutils.argscope import argscope
    from tensorpack.tfutils.scope_utils import under_name_scope
    from tensorpack.tfutils.summary import add_moving_summary
    from tensorpack.tfutils.tower import get_current_tower_context
    from tensorpack.utils.argtools import memoized

    # pylint: enable=import-error

if not import_guard.is_successful():
    from .....utils.mocks import layer_register, memoized, under_name_scope


@layer_register(log_shape=True)
def fpn_model(features, fpn_num_channels, fpn_norm):
    """
    Feature Pyramid Network model

    :param features: ResNet features c2-c5 [tf.Tensor]
    :param fpn_num_channels: FPN number of channels
    :param fpn_norm: FPN norm
    :return: FPN features p2-p6 [tf.Tensor]
    """
    assert len(features) == 4, features
    num_channel = fpn_num_channels

    use_gn = fpn_norm == "GN"

    def upsample2x(name, x):
        """
        Twofold up-sample

        :param name: name
        :param x: tf.Tensor
        :return: tf.Tensor
        """
        try:
            resize = tfv1.image.resize_images
            with tf.name_scope(name):
                shp2d = tf.shape(x)[2:]
                x = tf.transpose(x, [0, 2, 3, 1])
                x = resize(x, shp2d * 2, "nearest")
                x = tf.transpose(x, [0, 3, 1, 2])
                return x
        except AttributeError:
            return FixedUnPooling(
                name, x, 2, unpool_mat=np.ones((2, 2), dtype="float32"), data_format="channels_first"
            )  # pylint: disable=E1124

    with argscope(
        Conv2D,
        data_format="channels_first",
        activation=tf.identity,
        use_bias=not use_gn,
        kernel_initializer=tfv1.variance_scaling_initializer(scale=1.0),
    ):
        lat_2345 = [Conv2D(f"lateral_1x1_c{i + 2}", c, num_channel, 1) for i, c in enumerate(features)]
        if use_gn:
            lat_2345 = [GroupNorm(f"gn_c{i + 2}", c) for i, c in enumerate(lat_2345)]
        lat_sum_5432 = []
        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + upsample2x(f"upsample_lat{6 - idx}", lat_sum_5432[-1])
                lat_sum_5432.append(lat)
        p2345 = [Conv2D(f"posthoc_3x3_p{i + 2}", c, num_channel, 3) for i, c in enumerate(lat_sum_5432[::-1])]
        if use_gn:
            p2345 = [GroupNorm(f"gn_p{i + 2}", c) for i, c in enumerate(p2345)]
        p6 = MaxPooling(
            "maxpool_p6", p2345[-1], pool_size=1, strides=2, data_format="channels_first", padding="VALID"
        )  # pylint: disable=E1124
        return p2345 + [p6]


@under_name_scope()
def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5. Be careful that the returned tensor could be empty.

    :param boxes: (nx4)
    :return: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level. [tf.Tensor]
             4 tensors, the gathered boxes in each level. [tf.Tensor]
    """

    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.cast(tf.floor(4 + tf.math.log(sqrtarea * (1.0 / 224) + 1e-6) * (1.0 / np.log(2))), tf.int32)

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),  # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5),
    ]
    level_ids = [tf.reshape(x, [-1], name=f"roi_level{i + 2}_id") for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name=f"num_roi_level{i + 2}") for i, x in enumerate(level_ids)]
    add_moving_summary(*num_in_levels)

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


@under_name_scope()
def multilevel_roi_align(features, rcnn_boxes, resolution, fpn_anchor_strides):
    """
    multilevel roi align


    :param resolution: output spatial resolution  (int)
    :param rcnn_boxes: nx4 boxes (tf.Tensor)
    :param features: 4 FPN feature level 2-5 ([tf.Tensor])
    :param fpn_anchor_strides: FPN anchor strides
    :return: NxC x res x res
    """

    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope(f"roi_level{i + 2}"):
            boxes_on_featuremap = boxes * (1.0 / fpn_anchor_strides[i])
            all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

    # this can fail if using TF<=1.8 with MKL build
    all_rois = tf.concat(all_rois, axis=0)  # NCHW    # pylint: disable=E1123
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N   # pylint: disable=E1123
    level_id_invert_perm = tf.math.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm, name="output")
    return all_rois


def multilevel_rpn_losses(
    multilevel_anchors, multilevel_label_logits, multilevel_box_logits, rpn_batch_per_im, fpn_anchor_strides
):
    """
    multilevel rpn losses

    :param multilevel_anchors: #lvl RPNAnchors
    :param multilevel_label_logits: #lvl tensors of shape HxWxA
    :param multilevel_box_logits: #lvl tensors of shape HxWxAx4
    :param rpn_batch_per_im: RPN batch per image
    :param fpn_anchor_strides: FPN anchor strides
    :return: label_loss, box_loss
    """

    num_lvl = len(fpn_anchor_strides)
    assert len(multilevel_anchors) == num_lvl
    assert len(multilevel_label_logits) == num_lvl
    assert len(multilevel_box_logits) == num_lvl

    losses = []
    with tf.name_scope("rpn_losses"):
        for lvl in range(num_lvl):
            anchors = multilevel_anchors[lvl]
            label_loss, box_loss = rpn_losses(
                anchors.gt_labels,
                anchors.encoded_gt_boxes(),
                multilevel_label_logits[lvl],
                multilevel_box_logits[lvl],
                rpn_batch_per_im,
                name_scope=f"level{lvl + 2}",
            )
            losses.extend([label_loss, box_loss])

        total_label_loss = tf.add_n(losses[::2], name="label_loss")
        total_box_loss = tf.add_n(losses[1::2], name="box_loss")
        add_moving_summary(total_label_loss, total_box_loss)
    return [total_label_loss, total_box_loss]


@under_name_scope()
def generate_fpn_proposals(
    multilevel_pred_boxes,
    multilevel_label_logits,
    image_shape2d,
    fpn_anchor_strides,
    fpn_proposal_mode,
    rpn_train_per_level_nms_topk,
    rpn_per_level_nms_topk,
    rpn_min_size,
    rpn_proposal_nms_thresh,
    rpn_train_pre_nms_top_k,
    rpn_test_pre_nms_top_k,
    rpn_train_post_nms_top_k,
    rpn_test_post_nms_top_k,
):
    """
    generate fpn proposals

    :param multilevel_pred_boxes: #lvl HxWxAx4 boxes
    :param multilevel_label_logits: #lvl tensors of shape HxWxA
    :param image_shape2d: image shape 2d
    :param rpn_test_post_nms_top_k: RPN inference post NMS top k
    :param rpn_train_post_nms_top_k: RPN train post NMS top k
    :param rpn_test_pre_nms_top_k: RPN test pre nms top k
    :param rpn_train_pre_nms_top_k: RPN train pre NMS top k
    :param rpn_proposal_nms_thresh: RPN proposals NMS thresh
    :param rpn_min_size: RPN min size
    :param rpn_per_level_nms_topk:  RPN NMS top k per level
    :param rpn_train_per_level_nms_topk:  RPN train per level NMS top k
    :param fpn_proposal_mode: FPN proposal mode
    :param fpn_anchor_strides: FPN anchor strides
    :return: boxes: kx4 float
             scores: k logits
    """

    num_lvl = len(fpn_anchor_strides)
    assert len(multilevel_pred_boxes) == num_lvl
    assert len(multilevel_label_logits) == num_lvl

    training = get_current_tower_context().is_training
    all_boxes = []
    all_scores = []
    if fpn_proposal_mode == "Level":
        fpn_nms_top_k = rpn_train_per_level_nms_topk if training else rpn_per_level_nms_topk
        for lvl in range(num_lvl):
            with tf.name_scope(f"Lvl{lvl + 2}"):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]

                proposal_boxes, proposal_scores = generate_rpn_proposals(
                    tf.reshape(pred_boxes_decoded, [-1, 4]),
                    tf.reshape(multilevel_label_logits[lvl], [-1]),
                    image_shape2d,
                    rpn_min_size,
                    rpn_proposal_nms_thresh,
                    fpn_nms_top_k,
                )
                all_boxes.append(proposal_boxes)
                all_scores.append(proposal_scores)

        proposal_boxes = tf.concat(all_boxes, axis=0)  # nx4  # pylint: disable=E1123
        proposal_scores = tf.concat(all_scores, axis=0)  # n  # pylint: disable=E1123
        # Here we are different from Detectron.
        # Detectron picks top-k within the batch, rather than within an image, however we do not have a batch.
        proposal_topk = tf.minimum(tf.size(proposal_scores), fpn_nms_top_k)
        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices, name="all_proposals")
    else:
        for lvl in range(num_lvl):
            with tf.name_scope(f"Lvl{lvl + 2}"):
                pred_boxes_decoded = multilevel_pred_boxes[lvl]
                all_boxes.append(tf.reshape(pred_boxes_decoded, [-1, 4]))
                all_scores.append(tf.reshape(multilevel_label_logits[lvl], [-1]))
        all_boxes = tf.concat(all_boxes, axis=0)  # pylint: disable=E1123
        all_scores = tf.concat(all_scores, axis=0)  # pylint: disable=E1123
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            all_boxes,
            all_scores,
            image_shape2d,
            rpn_min_size,
            rpn_proposal_nms_thresh,
            rpn_train_pre_nms_top_k if training else rpn_test_pre_nms_top_k,
            rpn_train_post_nms_top_k if training else rpn_test_post_nms_top_k,
        )

    tf.sigmoid(proposal_scores, name="probs")  # for visualization
    return tf.stop_gradient(proposal_boxes, name="boxes"), tf.stop_gradient(proposal_scores, name="scores")


@memoized
def get_all_anchors_fpn(*, strides, sizes, ratios, max_size):
    """
    get all anchors for fpn

    :return: each anchors is a SxSx NUM_ANCHOR_RATIOS x4 array. [anchors]
    """
    assert len(strides) == len(sizes)
    foas = []
    for stride, size in zip(strides, sizes):
        foa = get_all_anchors(stride=stride, sizes=(size,), ratios=ratios, max_size=max_size)
        foas.append(foa)
    return foas
