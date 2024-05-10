# -*- coding: utf-8 -*-
# File: generalized_rcnn.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/generalized_rcnn.py>
"""


from lazy_imports import try_import

from ...tpcompat import ModelDescWithConfig
from ..utils.box_ops import area as tf_area
from . import model_frcnn, model_mrcnn
from .backbone import image_preprocess, resnet_fpn_backbone
from .model_box import RPNAnchors, clip_boxes, crop_and_resize
from .model_cascade import CascadeRCNNHead
from .model_fpn import (
    fpn_model,
    generate_fpn_proposals,
    get_all_anchors_fpn,
    multilevel_roi_align,
    multilevel_rpn_losses,
)
from .model_frcnn import (
    BoxProposals,
    FastRCNNHead,
    fastrcnn_outputs,
    fastrcnn_predictions,
    nms_post_processing,
    sample_fast_rcnn_targets,
)
from .model_mrcnn import maskrcnn_loss, unpackbits_masks
from .model_rpn import rpn_head

with try_import() as import_guard:
    # pylint: disable=import-error
    import tensorflow as tf
    from tensorpack import tfv1
    from tensorpack.models import l2_regularizer, regularize_cost
    from tensorpack.tfutils import optimizer
    from tensorpack.tfutils.summary import add_moving_summary

    # pylint: enable=import-error


class GeneralizedRCNN(ModelDescWithConfig):
    """
    GeneralizedRCNN
    """

    def preprocess(self, image):
        """
        Pre-processing steps

        :param image: tf.Tensor
        :return: tf.Tensor
        """
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, self.cfg)
        return tf.transpose(image, [0, 3, 1, 2])

    def optimizer(self):
        """
        optimizer
        """
        learning_rate = tfv1.get_variable("learning_rate", initializer=0.003, trainable=False)
        tf.summary.scalar("learning_rate-summary", learning_rate)

        # The learning rate in the config is set for 8 GPUs, and we use trainers with average=False.
        learning_rate = learning_rate / 8.0
        opt = tfv1.train.MomentumOptimizer(learning_rate, 0.9)
        if self.cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // self.cfg.TRAIN.NUM_GPUS)
        return opt

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        `build_graph` must create tensors of these names when called under inference context.
        :return: input names as list of strings, output names as list of strings
        """

        out = ["output/boxes", "output/scores", "output/labels"]
        if self.cfg.MODE_MASK:
            out.append("output/masks")
        return ["image"], out

    def build_graph(self, *inputs):  # pylint: disable=R1710
        """
        Build the graph

        :param inputs: args of inputs as defined in the self.inputs of subclass
        :return: total loss or None depending on training mode
        """
        inputs = dict(zip(self.input_names, inputs))
        if "gt_masks_packed" in inputs:
            gt_masks = tf.cast(unpackbits_masks(inputs.pop("gt_masks_packed")), tf.uint8, name="gt_masks")
            inputs["gt_masks"] = gt_masks

        image = self.preprocess(inputs["image"])  # 1CHW

        features = self.backbone(image)  # pylint: disable=E1101
        anchor_inputs = {k: v for k, v in inputs.items() if k.startswith("anchor_")}
        proposals, rpn_losses = self.rpn(image, features, anchor_inputs)  # pylint: disable=E1101

        targets = [inputs[k] for k in ["gt_boxes", "gt_labels", "gt_masks"] if k in inputs]
        gt_boxes_area = tf.reduce_mean(tf_area(inputs["gt_boxes"]), name="mean_gt_box_area")
        add_moving_summary(gt_boxes_area)
        head_losses = self.roi_heads(image, features, proposals, targets)  # pylint: disable=E1101

        if self.training:
            wd_cost = regularize_cost(".*/W", l2_regularizer(self.cfg.TRAIN.WEIGHT_DECAY), name="wd_cost")
            total_cost = tf.add_n(rpn_losses + head_losses + [wd_cost], "total_cost")
            add_moving_summary(total_cost, wd_cost)
            return total_cost

        # Check that the model defines the tensors it declares for inference
        # For existing models, they are defined in "fastrcnn_predictions(name_scope='output')"
        G = tf.compat.v1.get_default_graph()
        n_s = G.get_name_scope()
        for name in self.get_inference_tensor_names()[1]:
            try:
                name = "/".join([n_s, name]) if n_s else name
                G.get_tensor_by_name(name + ":0")
            except KeyError:
                raise KeyError(  # pylint: disable=W0707
                    f"Your model does not define the tensor '{name}' in inference context."
                )


class ResNetFPNModel(GeneralizedRCNN):
    """
    FPN and Cascade-RCNN with resnet/resnext backbone options
    """

    def inputs(self):
        """
        inputs

        :return: List of TensorSpec
        """
        ret = [tf.TensorSpec((None, None, 3), tf.float32, "image")]
        num_anchors = len(self.cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(self.cfg.FPN.ANCHOR_STRIDES)):
            ret.extend(
                [
                    tf.TensorSpec((None, None, num_anchors), tf.int32, f"anchor_labels_lvl{k + 2}"),
                    tf.TensorSpec((None, None, num_anchors, 4), tf.float32, f"anchor_boxes_lvl{k+2}"),
                ]
            )
        ret.extend(
            [tf.TensorSpec((None, 4), tf.float32, "gt_boxes"), tf.TensorSpec((None,), tf.int64, "gt_labels")]
        )  # all > 0
        if self.cfg.MODE_MASK:
            ret.append(tf.TensorSpec((None, None, None), tf.uint8, "gt_masks_packed"))
        return ret

    def slice_feature_and_anchors(self, p23456, anchors):
        """
        slice_feature_and_anchors

        :param p23456: output from fpn backbone
        :param anchors: anchors
        """
        for i, _ in enumerate(self.cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope(f"FPN_slice_lvl{i}"):
                anchors[i] = anchors[i].narrow_to(p23456[i])

    def backbone(self, image):
        """
        backbone

        :param image: tf.Tensor
        """
        c2345 = resnet_fpn_backbone(image, self.cfg)
        p23456 = fpn_model("fpn", c2345, self.cfg.FPN.NUM_CHANNEL, self.cfg.FPN.NORM)  # pylint: disable=E1121
        return p23456

    def rpn(self, image, features, inputs):
        """
        Region Proposal Network

        :param image: tf.Tensor
        :param features: output features
        :param inputs: input dict of anchor levels
        :return: BoxProposals and list of losses
        """
        assert len(self.cfg.RPN.ANCHOR_SIZES) == len(self.cfg.FPN.ANCHOR_STRIDES)

        image_shape2d = tf.shape(image)[2:]  # h,w
        all_anchors_fpn = get_all_anchors_fpn(
            strides=self.cfg.FPN.ANCHOR_STRIDES,
            sizes=self.cfg.RPN.ANCHOR_SIZES,
            ratios=self.cfg.RPN.ANCHOR_RATIOS,
            max_size=self.cfg.PREPROC.MAX_SIZE,
        )
        multilevel_anchors = [
            RPNAnchors(
                all_anchors_fpn[i],
                inputs[f"anchor_labels_lvl{i+2}"],
                inputs[f"anchor_boxes_lvl{i+2}"],
            )
            for i in range(len(all_anchors_fpn))
        ]
        self.slice_feature_and_anchors(features, multilevel_anchors)

        # Multi-Level RPN Proposals
        rpn_outputs = [
            rpn_head("rpn", pi, self.cfg.FPN.NUM_CHANNEL, len(self.cfg.RPN.ANCHOR_RATIOS))  # pylint: disable=E1121
            for pi in features
        ]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs]
        multilevel_pred_boxes = [
            anchor.decode_logits(logits, self.cfg.PREPROC.MAX_SIZE)
            for anchor, logits in zip(multilevel_anchors, multilevel_box_logits)
        ]

        proposal_boxes, _ = generate_fpn_proposals(
            multilevel_pred_boxes,
            multilevel_label_logits,
            image_shape2d,
            self.cfg.FPN.ANCHOR_STRIDES,
            self.cfg.FPN.PROPOSAL_MODE,
            self.cfg.RPN.TRAIN_PER_LEVEL_NMS_TOPK,
            self.cfg.RPN.PER_LEVEL_NMS_TOPK,
            self.cfg.RPN.MIN_SIZE,
            self.cfg.RPN.PROPOSAL_NMS_THRESH,
            self.cfg.RPN.TRAIN_PRE_NMS_TOPK,
            self.cfg.RPN.PRE_NMS_TOPK,
            self.cfg.RPN.TRAIN_POST_NMS_TOPK,
            self.cfg.RPN.POST_NMS_TOPK,
        )

        if self.training:
            losses = multilevel_rpn_losses(
                multilevel_anchors,
                multilevel_label_logits,
                multilevel_box_logits,
                self.cfg.RPN.BATCH_PER_IM,
                self.cfg.FPN.ANCHOR_STRIDES,
            )
        else:
            losses = []

        return BoxProposals(proposal_boxes), losses

    def roi_heads(self, image, features, proposals, targets):
        """
        Region of Interest head classifier and regressor

        :param image: tf.Tensor
        :param features: features from backbone output
        :param proposals: RPN proposals
        :param targets: targets
        :return: list of total loss when training or an empty list
        """
        image_shape2d = tf.shape(image)[2:]  # h,w
        assert len(features) == 5, "Features have to be P23456!"
        gt_boxes, gt_labels, *_ = targets

        if self.training:
            proposals = sample_fast_rcnn_targets(
                proposals.boxes,
                gt_boxes,
                gt_labels,
                self.cfg.FRCNN.FG_THRESH,
                self.cfg.FRCNN.BATCH_PER_IM,
                self.cfg.FRCNN.FG_RATIO,
            )

        fastrcnn_head_func = getattr(model_frcnn, self.cfg.FPN.FRCNN_HEAD_FUNC)
        if not self.cfg.FPN.CASCADE:
            roi_feature_fastrcnn = multilevel_roi_align(features[:4], proposals.boxes, 7, self.cfg.FPN.ANCHOR_STRIDES)

            head_feature = fastrcnn_head_func("fastrcnn", roi_feature_fastrcnn, cfg=self.cfg)
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs(
                "fastrcnn/outputs", head_feature, self.cfg.DATA.NUM_CATEGORY
            )
            fastrcnn_head = FastRCNNHead(
                proposals,
                fastrcnn_box_logits,
                fastrcnn_label_logits,
                gt_boxes,
                tf.constant(self.cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32),
                self.cfg,
            )
        else:

            def roi_func(boxes):
                return multilevel_roi_align(features[:4], boxes, 7, self.cfg.FPN.ANCHOR_STRIDES)

            fastrcnn_head = CascadeRCNNHead(
                proposals,
                roi_func,
                fastrcnn_head_func,
                (gt_boxes, gt_labels),
                image_shape2d,
                self.cfg.DATA.NUM_CATEGORY,
                self.cfg,
            )

        if self.training:
            all_losses = fastrcnn_head.losses()

            if self.cfg.MODE_MASK:
                gt_masks = targets[2]
                # maskrcnn loss
                roi_feature_maskrcnn = multilevel_roi_align(
                    features[:4],
                    proposals.fg_boxes(),
                    14,
                    self.cfg,
                    name_scope="multilevel_roi_align_mask",
                )
                maskrcnn_head_func = getattr(model_mrcnn, self.cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    "maskrcnn", roi_feature_maskrcnn, self.cfg.DATA.NUM_CATEGORY, cfg=self.cfg
                )  # #fg x #cat x 28 x 28

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks, 1), proposals.fg_boxes(), proposals.fg_inds_wrt_gt, 28, pad_border=False
                )  # fg x 1x28x28
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, "sampled_fg_mask_targets")
                all_losses.append(maskrcnn_loss(mask_logits, proposals.fg_labels(), target_masks_for_fg))
            return all_losses
        decoded_boxes = fastrcnn_head.decoded_output_boxes()
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name="fastrcnn_all_boxes")
        label_scores = fastrcnn_head.output_scores(name="fastrcnn_all_scores")
        boxes, scores, labels = fastrcnn_predictions(
            decoded_boxes,
            label_scores,
            self.cfg.OUTPUT.RESULT_SCORE_THRESH,
            self.cfg.OUTPUT.RESULTS_PER_IM,
            self.cfg.OUTPUT.FRCNN_NMS_THRESH,
            name_scope="pre_output",
        )
        final_boxes, _, final_labels = nms_post_processing(
            boxes,
            scores,
            labels,
            self.cfg.OUTPUT.RESULTS_PER_IM,
            self.cfg.OUTPUT.NMS_THRESH_CLASS_AGNOSTIC,
            name_scope="output",
        )

        if self.cfg.MODE_MASK:
            # Cascade inference needs roi transform with refined boxes.
            roi_feature_maskrcnn = multilevel_roi_align(features[:4], final_boxes, 14)  # pylint: disable=E1120
            maskrcnn_head_func = getattr(model_mrcnn, self.cfg.FPN.MRCNN_HEAD_FUNC)
            mask_logits = maskrcnn_head_func(
                "maskrcnn", roi_feature_maskrcnn, self.cfg.DATA.NUM_CATEGORY
            )  # #fg x #cat x 28 x 28
            indices = tf.stack([tf.range(tf.size(final_labels)), tf.cast(final_labels, tf.int32) - 1], axis=1)
            final_mask_logits = tf.gather_nd(mask_logits, indices)  # #resultx28x28
            tf.sigmoid(final_mask_logits, name="output/masks")
        return []
