# -*- coding: utf-8 -*-
# File: model_cascade.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/model_cascade.py>
"""

from lazy_imports import try_import

from ..utils.box_ops import area as tf_area
from ..utils.box_ops import pairwise_iou
from .model_box import clip_boxes
from .model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs

with try_import() as import_guard:
    # pylint: disable=import-error
    import tensorflow as tf
    from tensorpack import tfv1
    from tensorpack.tfutils import get_current_tower_context

    # pylint: enable=import-error


class CascadeRCNNHead:
    """
    Cascade RCNN Head
    """

    def __init__(
        self, proposals, roi_func, fastrcnn_head_func, gt_targets, image_shape2d, num_categories, cfg
    ):  # pylint: disable =W0613
        """
        :param proposals: BoxProposals
        :param roi_func: a function to crop features with rois
        :param fastrcnn_head_func: the fastrcnn head to apply on the cropped features
        :param gt_targets: gt_targets
        :param image_shape2d: image shape (height,width)
        :param num_categories: number of categories
        :param cfg: config
        """

        self.cfg = cfg
        for key, value in locals().items():
            if key != "self":
                setattr(self, key, value)
        self.gt_boxes, self.gt_labels = gt_targets
        del self.gt_targets  # pylint: disable =E1101

        self.num_cascade_stages = len(cfg.CASCADE.IOUS)

        self.training = get_current_tower_context().is_training
        if self.training:

            @tf.custom_gradient
            def scale_gradient(x):
                return x, lambda dy: dy * (1.0 / self.num_cascade_stages)

            self.scale_gradient = scale_gradient
        else:
            self.scale_gradient = tf.identity

        ious = cfg.CASCADE.IOUS
        # It's unclear how to do >3 stages, so it does not make sense to implement them
        assert self.num_cascade_stages == 3, "Only 3-stage cascade was implemented!"
        with tfv1.variable_scope("cascade_rcnn_stage1"):
            H1, B1 = self.run_head(self.proposals, 0)  # pylint: disable =E1101

        with tfv1.variable_scope("cascade_rcnn_stage2"):
            B1_proposal = self.match_box_with_gt(B1, ious[1])
            H2, B2 = self.run_head(B1_proposal, 1)

        with tfv1.variable_scope("cascade_rcnn_stage3"):
            B2_proposal = self.match_box_with_gt(B2, ious[2])
            H3, B3 = self.run_head(B2_proposal, 2)
        self._cascade_boxes = [B1, B2, B3]
        self._heads = [H1, H2, H3]

    def run_head(self, proposals, stage):
        """
        Run the Fast-RCNN Head

        :param proposals: BoxProposals
        :param stage: 0, 1, 2
        :return: FastRCNNHead
                 Nx4, updated boxes
        """

        reg_weights = tf.constant(self.cfg.CASCADE.BBOX_REG_WEIGHTS[stage], dtype=tf.float32)
        pooled_feature = self.roi_func(proposals.boxes)  # N,C,S,S  # pylint: disable =E1101
        pooled_feature = self.scale_gradient(pooled_feature)
        head_feature = self.fastrcnn_head_func("head", pooled_feature, cfg=self.cfg)  # pylint: disable =E1101
        label_logits, box_logits = fastrcnn_outputs(  # pylint: disable =E1124
            "outputs", head_feature, self.num_categories, class_agnostic_regression=True  # pylint: disable =E1101
        )
        head = FastRCNNHead(proposals, box_logits, label_logits, self.gt_boxes, reg_weights, self.cfg)

        refined_boxes = head.decoded_output_boxes_class_agnostic()
        refined_boxes = clip_boxes(refined_boxes, self.image_shape2d)  # pylint: disable =E1101
        if self.training:
            refined_boxes = tf.boolean_mask(refined_boxes, tf_area(refined_boxes) > 0)
        return head, tf.stop_gradient(refined_boxes, name="output_boxes")

    def match_box_with_gt(self, boxes, iou_threshold):
        """
        Match box with ground truth

        :param boxes: Nx4
        :param iou_threshold: float
        :return: BoxProposals
        """
        if self.training:
            with tf.name_scope(f"match_box_with_gt_{iou_threshold}"):
                iou = pairwise_iou(boxes, self.gt_boxes)  # NxM
                max_iou_per_box = tf.reduce_max(iou, axis=1)  # N
                best_iou_ind = tf.argmax(iou, axis=1)  # N
                labels_per_box = tf.gather(self.gt_labels, best_iou_ind)
                fg_mask = max_iou_per_box >= iou_threshold
                fg_inds_wrt_gt = tf.boolean_mask(best_iou_ind, fg_mask)
                labels_per_box = tf.stop_gradient(labels_per_box * tf.cast(fg_mask, tf.int64))
                return BoxProposals(boxes, labels_per_box, fg_inds_wrt_gt)
        else:
            return BoxProposals(boxes)

    def losses(self):
        """
        losses
        """
        ret = []
        for idx, head in enumerate(self._heads):
            with tf.name_scope(f"cascade_loss_stage{idx + 1}"):
                ret.extend(head.losses())
        return ret

    def decoded_output_boxes(self):
        """
        decoded output boxes

        :returns:  Nx#classx4
        """
        ret = self._cascade_boxes[-1]
        ret = tf.expand_dims(ret, 1)  # class-agnostic
        return tf.tile(ret, [1, self.num_categories + 1, 1])  # pylint: disable =E1101

    def output_scores(self, name=None):
        """
        :returns: Nx#class
        """
        scores = [head.output_scores(f"cascade_scores_stage{idx + 1}") for idx, head in enumerate(self._heads)]
        return tf.multiply(tf.add_n(scores), (1.0 / self.num_cascade_stages), name=name)
