# -*- coding: utf-8 -*-
# File: model_frcnn.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_frcnn.py>
"""

from lazy_imports import try_import

from ..utils.box_ops import pairwise_iou
from .backbone import GroupNorm
from .model_box import decode_bbox_target, encode_bbox_target

with try_import() as import_guard:
    # pylint: disable=import-error
    import tensorflow as tf
    from tensorpack import tfv1
    from tensorpack.models import Conv2D, FullyConnected, layer_register
    from tensorpack.tfutils.argscope import argscope
    from tensorpack.tfutils.common import get_tf_version_tuple
    from tensorpack.tfutils.scope_utils import under_name_scope
    from tensorpack.tfutils.summary import add_moving_summary
    from tensorpack.utils.argtools import memoized_method

    # pylint: enable=import-error

if not import_guard.is_successful():
    from .....utils.mocks import layer_register, memoized_method, under_name_scope


@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    :param iou: nxm, #proposal x #gt
    """
    # find best roi for each gt, for summary only
    best_iou = tf.reduce_max(iou, axis=0)
    mean_best_iou = tf.reduce_mean(best_iou, name="best_iou_per_gt")
    summaries = [mean_best_iou]
    with tf.device("/cpu:0"):
        for thresh in [0.3, 0.5]:
            recall = tf.truediv(
                tfv1.count_nonzero(best_iou >= thresh),
                tf.size(best_iou, out_type=tf.int64),
                name=f"recall_iou{thresh}",
            )
            summaries.append(recall)
    add_moving_summary(*summaries)


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels, frcnn_fg_thresh, frcnn_batch_per_im, frcnn_fg_ratio):
    """
    Sample some boxes from all proposals for training.
    #fg is guaranteed to be > 0, because ground truth boxes will be added as proposals.

    :param gt_labels:  m, int32
    :param gt_boxes: mx4, float box
    :param boxes: nx4 region proposals, float box
    :param frcnn_fg_ratio: front ground ratio
    :param frcnn_batch_per_im: batch per image
    :param frcnn_fg_thresh: front ground threshold

    :return: A BoxProposals instance, with:
                sampled_boxes: tx4 float box, the rois
                sampled_labels: t int64 labels, in [0, #class). Positive means foreground.
                fg_inds_wrt_gt: #fg indices, each in range [0, m-1].

                It contains the matching GT of each foreground roi.
    """

    iou = pairwise_iou(boxes, gt_boxes)  # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)  # (n+m) x 4  # pylint: disable=E1123
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)  # (n+m) x m  # pylint: disable=E1123
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        """
        sample fg bg
        """
        fg_mask = tf.cond(
            tf.shape(iou)[1] > 0,
            lambda: tf.reduce_max(iou, axis=1) >= frcnn_fg_thresh,
            lambda: tf.zeros(tf.shape(iou)[0], dtype=tf.bool),
        )

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(frcnn_batch_per_im * frcnn_fg_ratio), tf.size(fg_inds), name="num_fg")
        fg_inds = tf.random.shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(frcnn_batch_per_im - num_fg, tf.size(bg_inds), name="num_bg")
        bg_inds = tfv1.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.cond(
        tf.shape(iou)[1] > 0,
        lambda: tf.argmax(iou, axis=1),  # #proposal, each in 0~m-1
        lambda: tf.zeros(tf.shape(iou)[0], dtype=tf.int64),
    )
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)  # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)  # indices w.r.t all n+m proposal boxes  # pylint: disable=E1123
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(  # pylint: disable=E1123
        [tf.gather(gt_labels, fg_inds_wrt_gt), tf.zeros_like(bg_inds, dtype=tf.int64)],
        axis=0,
    )
    # stop the gradient -- they are meant to be training targets
    return BoxProposals(
        tf.stop_gradient(ret_boxes, name="sampled_proposal_boxes"),
        tf.stop_gradient(ret_labels, name="sampled_labels"),
        tf.stop_gradient(fg_inds_wrt_gt),
    )


@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_categories, class_agnostic_regression=False):
    """
    Fast RCNN outputs

    :param feature:(any shape)
    :param num_categories:
    :param class_agnostic_regression: if True, regression to N x 1 x 4

    :return cls_logits: N x num_class classification logits
            reg_logits: N x num_class x 4 or Nx1x4 if class agnostic
    """

    num_classes = num_categories + 1
    classification = FullyConnected(
        "class", feature, num_classes, kernel_initializer=tf.random_normal_initializer(stddev=0.01)
    )
    num_classes_for_box = 1 if class_agnostic_regression else num_classes
    box_regression = FullyConnected(
        "box", feature, num_classes_for_box * 4, kernel_initializer=tf.random_normal_initializer(stddev=0.001)
    )
    box_regression = tf.reshape(box_regression, (-1, num_classes_for_box, 4), name="output_box")
    return classification, box_regression


@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Fast RCNN losses

    :param labels: n,
    :param label_logits: nxC
    :param fg_boxes: number front ground x4, encoded
    :param fg_box_logits: number front ground xCx4 or number front ground x1x4 if class agnostic

    :return: label_loss, box_loss
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name="label_loss")

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds, out_type=tf.int64)
    empty_fg = tf.equal(num_fg, 0)
    if int(fg_box_logits.shape[1]) > 1:
        if get_tf_version_tuple() >= (1, 14):
            fg_labels = tf.expand_dims(fg_labels, axis=1)  # nfg x 1
            fg_box_logits = tf.gather(fg_box_logits, fg_labels, batch_dims=1)
        else:
            indices = tf.stack([tf.range(num_fg), fg_labels], axis=1)  # number front ground x2
            fg_box_logits = tf.gather_nd(fg_box_logits, indices)
    fg_box_logits = tf.reshape(fg_box_logits, [-1, 4])  # nfg x 4

    with tf.name_scope("label_metrics"), tf.device("/cpu:0"):
        prediction = tf.argmax(label_logits, axis=1, name="label_prediction")
        correct = tf.cast(tf.equal(prediction, labels), tf.float32)  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name="accuracy")
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int64), name="num_zero")
        false_negative = tf.where(
            empty_fg, 0.0, tf.cast(tf.truediv(num_zero, num_fg), tf.float32), name="false_negative"
        )
        fg_accuracy = tf.where(empty_fg, 0.0, tf.reduce_mean(tf.gather(correct, fg_inds)), name="fg_accuracy")

    box_loss = tf.reduce_sum(tf.abs(fg_boxes - fg_box_logits))
    box_loss = tf.truediv(box_loss, tf.cast(tf.shape(labels)[0], tf.float32), name="box_loss")

    add_moving_summary(
        label_loss, box_loss, accuracy, fg_accuracy, false_negative, tf.cast(num_fg, tf.float32, name="num_fg_label")
    )
    return [label_loss, box_loss]


@under_name_scope()
def fastrcnn_predictions(boxes, scores, output_result_score_thresh, output_results_per_im, output_frcnn_nms_thresh):
    """
    Generate final results from predictions of all proposals.

    :param scores: nx#class
    :param boxes: n#classx4 float box in float32
    :param output_frcnn_nms_thresh: output frcnn nms threshold
    :param output_results_per_im: output results per image
    :param output_result_score_thresh: output result score threshold

    :return pre_boxes: Kx4
            pre_scores: K
            pre_labels: K
    """

    assert boxes.shape[1] == scores.shape[1]
    boxes = tf.transpose(boxes, [1, 0, 2])[1:, :, :]  # #catxnx4
    scores = tf.transpose(scores[:, 1:], [1, 0])  # #catxn

    max_coord = tf.reduce_max(boxes)
    filtered_ids = tf.where(scores > output_result_score_thresh)  # Fx2
    filtered_boxes = tf.gather_nd(boxes, filtered_ids)  # Fx4
    filtered_scores = tf.gather_nd(scores, filtered_ids)  # F,
    cls_per_box = tf.slice(filtered_ids, [0, 0], [-1, 1])
    offsets = tf.cast(cls_per_box, tf.float32) * (max_coord + 1)  # F,1
    nms_boxes = filtered_boxes + offsets
    selection = tf.image.non_max_suppression(nms_boxes, filtered_scores, output_results_per_im, output_frcnn_nms_thresh)
    pre_scores = tf.gather(filtered_scores, selection, name="pre_scores")
    pre_labels = tf.add(tf.gather(cls_per_box[:, 0], selection), 1, name="pre_labels")
    pre_boxes = tf.gather(filtered_boxes, selection, name="pre_boxes")
    return pre_boxes, pre_scores, pre_labels


@under_name_scope()
def nms_post_processing(boxes, scores, labels, output_results_per_im, output_nms_thresh_class_agnostic):
    """
    Final results from Fast RCNN are calculated from performing nms per class. For layout detection overlapping boxes
    with different categories are not possible, so this post-processing steps is doing a final nms calculation over all
    classes.


    :param boxes: n#classx4 float box in float32
    :param scores: n
    :param output_nms_thresh_class_agnostic: output nms threshold, this time performing nms over all classes
    :param output_results_per_im: max prediction outputs
    :param labels: labels

    :return boxes: Kx4
            scores: K
    """

    selection = tf.image.non_max_suppression(
        boxes, scores, max_output_size=output_results_per_im, iou_threshold=output_nms_thresh_class_agnostic
    )
    final_boxes = tf.gather(boxes, selection, name="boxes")
    final_scores = tf.gather(scores, selection, name="scores")
    final_labels = tf.gather(labels, selection, name="labels")
    return final_boxes, final_scores, final_labels


@layer_register(log_shape=True)
def fastrcnn_2fc_head(feature, cfg):
    """
    FRCNN head with two fully connected heads

    :param feature: (any shape)
    :param cfg: The configuration instance as an AttrDict
    """

    dim = cfg.FPN.FRCNN_FC_HEAD_DIM
    init = tfv1.variance_scaling_initializer()
    hidden = FullyConnected(
        "fc6", feature, dim, kernel_initializer=init, activation=tf.nn.relu
    )  # pylint: disable=E1124
    hidden = FullyConnected("fc7", hidden, dim, kernel_initializer=init, activation=tf.nn.relu)  # pylint: disable=E1124
    return hidden


@layer_register(log_shape=True)
def fastrcnn_Xconv1fc_head(feature, num_convs, norm=None, **kwargs):  # pylint: disable = C0103
    """
    FRCNN head with x convolutional and one fully connected head

    :param  feature: The configuration instance as an AttrDict (NCHW)
    :param  num_convs: number of conv layers
    :param  norm: either None or 'GN'
    """

    cfg = kwargs["cfg"]
    assert norm in [None, "GN"], norm
    l = feature
    with argscope(
        Conv2D,
        data_format="channels_first",
        kernel_initializer=tfv1.variance_scaling_initializer(
            scale=2.0,
            mode="fan_out",
            distribution="untruncated_normal" if get_tf_version_tuple() >= (1, 12) else "normal",
        ),
    ):
        for k in range(num_convs):
            l = Conv2D(f"conv{k}", l, cfg.FPN.FRCNN_CONV_HEAD_DIM, 3, activation=tf.nn.relu)
            if norm is not None:
                l = GroupNorm(f"gn{k}", l)
        l = FullyConnected(  # pylint: disable=E1124
            "fc",
            l,
            cfg.FPN.FRCNN_FC_HEAD_DIM,
            kernel_initializer=tfv1.variance_scaling_initializer(),
            activation=tf.nn.relu,
        )
    return l


def fastrcnn_4conv1fc_head(*args, **kwargs):
    """
    FRCNN head with four convolutional and one fully connected head
    """
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, **kwargs)


def fastrcnn_4conv1fc_gn_head(*args, **kwargs):
    """
    FRCNN head with four convolutional and one fully connected group normalized head
    """
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, norm="GN", **kwargs)


class BoxProposals:
    """
    A structure to manage box proposals and their relations with ground truth.
    """

    def __init__(self, boxes, labels=None, fg_inds_wrt_gt=None):  # pylint: disable = W0613
        """
        :param boxes: Nx4
        :param labels: N, each in [0, #class), the true label for each input box
        :param fg_inds_wrt_gt: #fg, each in [0, M)

        The last four arguments could be None when not training.
        """

        for key, value in locals().items():
            if key != "self" and value is not None:
                setattr(self, key, value)

    @memoized_method
    def fg_inds(self):
        """
        Returns: #fg indices in [0, N-1]
        """
        return tf.reshape(tf.where(self.labels > 0), [-1], name="fg_inds")  # pylint: disable = E1101

    @memoized_method
    def fg_boxes(self):
        """
        Returns: #fg x4
        """
        return tf.gather(self.boxes, self.fg_inds(), name="fg_boxes")  # pylint: disable = E1101

    @memoized_method
    def fg_labels(self):
        """
        Returns: #fg
        """
        return tf.gather(self.labels, self.fg_inds(), name="fg_labels")  # pylint: disable = E1101


class FastRCNNHead:
    """
    A class to process & decode inputs/outputs of a fastrcnn classification+regression head.
    """

    def __init__(
        self, proposals, box_logits, label_logits, gt_boxes, bbox_regression_weights, cfg
    ):  # pylint: disable =W0613
        """
        :param proposals: BoxProposals
        :param box_logits: Nx#classx4 or Nx1x4, the output of the head
        :param label_logits: Nx#class, the output of the head
        :param gt_boxes: Mx4
        :param bbox_regression_weights: a 4 element tensor
        """

        self.cfg = cfg
        for key, value in locals().items():
            if key != "self" and value is not None:
                setattr(self, key, value)
        self._bbox_class_agnostic = int(box_logits.shape[1]) == 1
        self._num_classes = box_logits.shape[1]

    @memoized_method
    def fg_box_logits(self):
        """
        Returns: #fg x ? x 4
        """
        return tf.gather(self.box_logits, self.proposals.fg_inds(), name="fg_box_logits")  # pylint: disable =E1101

    @memoized_method
    def losses(self):
        """
        losses
        """
        encoded_fg_gt_boxes = (
            encode_bbox_target(
                tf.gather(self.gt_boxes, self.proposals.fg_inds_wrt_gt),  # pylint: disable =E1101
                self.proposals.fg_boxes(),  # pylint: disable =E1101
            )
            * self.bbox_regression_weights  # pylint: disable =E1101
        )
        return fastrcnn_losses(
            self.proposals.labels,  # pylint: disable =E1101
            self.label_logits,  # pylint: disable =E1101
            encoded_fg_gt_boxes,
            self.fg_box_logits(),
        )

    @memoized_method
    def decoded_output_boxes(self):
        """
        Returns: N x #class x 4
        """
        anchors = tf.tile(
            tf.expand_dims(self.proposals.boxes, 1), [1, self._num_classes, 1]  # pylint: disable =E1101
        )  # N x #class x 4
        decoded_boxes = decode_bbox_target(
            self.box_logits / self.bbox_regression_weights, anchors, self.cfg.PREPROC.MAX_SIZE  # pylint: disable =E1101
        )
        return decoded_boxes

    @memoized_method
    def decoded_output_boxes_for_true_label(self):
        """
        Returns: Nx4 decoded boxes
        """
        return self._decoded_output_boxes_for_label(self.proposals.labels)  # pylint: disable =E1101

    @memoized_method
    def decoded_output_boxes_for_predicted_label(self):
        """
        Returns: Nx4 decoded boxes
        """
        return self._decoded_output_boxes_for_label(self.predicted_labels())  # pylint: disable =E1101

    @memoized_method
    def decoded_output_boxes_for_label(self, labels):
        """
        Returns: Nx4 decoded boxes
        """
        assert not self._bbox_class_agnostic
        indices = tf.stack([tf.range(tf.size(labels, out_type=tf.int64)), labels])
        needed_logits = tf.gather_nd(self.box_logits, indices)  # pylint: disable =E1101
        decoded = decode_bbox_target(
            needed_logits / self.bbox_regression_weights,  # pylint: disable =E1101
            self.proposals.boxes,  # pylint: disable =E1101
            self.cfg.PREPROC.MAX_SIZE,
        )
        return decoded

    @memoized_method
    def decoded_output_boxes_class_agnostic(self):
        """
        Returns: Nx4
        """
        assert self._bbox_class_agnostic
        box_logits = tf.reshape(self.box_logits, [-1, 4])  # pylint: disable =E1101
        decoded = decode_bbox_target(
            box_logits / self.bbox_regression_weights,  # pylint: disable =E1101
            self.proposals.boxes,  # pylint: disable =E1101
            self.cfg.PREPROC.MAX_SIZE,
        )
        return decoded

    @memoized_method
    def output_scores(self, name=None):
        """
        Returns: N x #class scores, summed to one for each box.
        """
        return tf.nn.softmax(self.label_logits, name=name)  # pylint: disable =E1101

    @memoized_method
    def predicted_labels(self):
        """
        Returns: N ints
        """
        return tf.argmax(self.label_logits, axis=1, name="predicted_labels")  # pylint: disable =E1101
