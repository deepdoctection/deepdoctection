# -*- coding: utf-8 -*-
# File: preproc.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/1a79d595f7eda9dc9dc8428f4461680ed2222ab6/examples/FasterRCNN/data.py>
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
from lazy_imports import try_import

from ....utils.error import MalformedData
from ....utils.logger import log_once
from ....utils.transform import box_to_point4, point4_to_box
from ....utils.types import JsonDict, PixelValues
from .common import filter_boxes_inside_shape, np_iou
from .modeling.model_fpn import get_all_anchors_fpn
from .utils.np_box_ops import area as np_area
from .utils.np_box_ops import ioa as np_ioa

# pylint: disable=import-error


with try_import() as import_guard:
    from tensorpack.dataflow.imgaug import AugmentorList, ImageAugmentor
# pylint: enable=import-error


def augment(dp: JsonDict, imgaug_list: List[ImageAugmentor], add_mask: bool) -> JsonDict:
    """
    Augment an image according to a list of augmentors.

    :param dp: A dict with "image","gt_boxes","gt_labels"
    :param imgaug_list: List auf augmentors
    :param add_mask: not implemented

    :return: A dict with augmented image and gt_boxes
    """

    image = dp["image"]
    gt_boxes = dp["gt_boxes"]
    if gt_boxes.ndim == 1:
        print("stop")
    augmentations = AugmentorList(imgaug_list)
    tf_ms = augmentations.get_transform(image)
    image_aug = tf_ms.apply_image(image)
    points = box_to_point4(gt_boxes)
    points = tf_ms.apply_coords(points)
    gt_boxes = point4_to_box(points)

    dp["image"] = image_aug
    dp["gt_boxes"] = gt_boxes

    if len(gt_boxes):
        assert np.min(np_area(gt_boxes)) > 0, "some boxes have zero area"

    if add_mask:
        raise NotImplementedError()

    return dp


def anchors_and_labels(
    dp: JsonDict,
    anchor_strides: Tuple[int],
    anchor_sizes: Tuple[int],
    anchor_ratios: Tuple[int],
    max_size: int,
    batch_per_image: int,
    front_ground_ratio: float,
    positive_anchor_threshold: float,
    negative_anchor_threshold: float,
    crowd_overlap_threshold: float,
) -> Optional[JsonDict]:
    """
    Generating anchors and labels.

    :param dp: datapoint image
    :param anchor_strides: the stride between the center of neighbored anchors
    :param anchor_sizes: a list of anchor sizes: https://arxiv.org/abs/1506.01497
    :param anchor_ratios: a list of anchor ratios: https://arxiv.org/abs/1506.01497
    :param max_size: the maximum size a image entering the model can have.
    :param batch_per_image: total (across FPN levels) number of anchors that are marked valid
    :param front_ground_ratio: front ground ratio among selected RPN anchors
    :param positive_anchor_threshold: will keep all anchors with an IOU-threshold above this benchmark
    :param negative_anchor_threshold: will filter all anchors with an IOU-threshold below this benchmark
    :param crowd_overlap_threshold: Anchors which overlap with a crowd box (IOA larger than threshold)
                                    will be ignored. Setting this to a value larger than 1.0 will disable the feature
    """

    gt_boxes = dp["gt_boxes"]
    image = dp["image"]

    try:
        dummy_crowd_ind = np.zeros((gt_boxes.shape[0],), dtype="int8")

        multilevel_anchor_inputs = get_multilevel_rpn_anchor_input(
            image,
            gt_boxes,
            dummy_crowd_ind,
            anchor_strides,
            anchor_sizes,
            anchor_ratios,
            max_size,
            batch_per_image,
            front_ground_ratio,
            positive_anchor_threshold,
            negative_anchor_threshold,
            crowd_overlap_threshold,
        )
        for i, (anchor_labels, anchor_boxes) in enumerate(multilevel_anchor_inputs):
            dp[f"anchor_labels_lvl{i + 2}"] = anchor_labels
            dp[f"anchor_boxes_lvl{i + 2}"] = anchor_boxes

    except (MalformedData, IndexError) as err:
        log_once(f"input {dp['file_name']} is filtered for training: {str(err)}")
        return None

    return dp


def get_multilevel_rpn_anchor_input(
    image: np.array,
    boxes: np.array,
    is_crowd: np.array,
    anchor_strides: Tuple[int],
    anchor_sizes: Tuple[int],
    anchor_ratios: Tuple[int],
    max_size: float,
    batch_per_image: int,
    front_ground_ratio: float,
    positive_anchor_threshold: float,
    negative_anchor_threshold: float,
    crowd_overlap_threshold: float,
) -> List[Tuple[Any, Any]]:
    """
    Generates multilevel rpn anchors

    :param image: an image
    :param crowd_overlap_threshold: Anchors which overlap with a crowd box (IOA larger than threshold)
        will be ignored. Setting this to a value larger than 1.0 will disable the feature.
    :param negative_anchor_threshold: will filter all anchors with an IOU-threshold below this benchmark
    :param positive_anchor_threshold: will keep all anchors with an IOU-threshold above this benchmark
    :param front_ground_ratio: front ground ratio among selected RPN anchors
    :param batch_per_image: total (across FPN levels) number of anchors that are marked valid
    :param max_size: the maximum size an image entering the model can have.
    :param anchor_ratios: a list of anchor ratios: https://arxiv.org/abs/1506.01497
    :param anchor_sizes: a list of anchor sizes: https://arxiv.org/abs/1506.01497
    :param is_crowd: a numpy array with is_crowd indices as defined in coco, size: n,
    :param boxes: float box, gt. shouldn't be changed, size: nx4
    :param anchor_strides: the stride between the center of neighbored anchors

    :return:
        [(fm_labels, fm_boxes)]: Returns a tuple for each FPN level. Each tuple contains the anchor labels and target
                                  boxes for each pixel in the feature map.

        fm_labels: fHxfWx NUM_ANCHOR_RATIOS
        fm_boxes: fHxfWx NUM_ANCHOR_RATIOS x4

    """

    boxes = boxes.copy()
    anchors_per_level = get_all_anchors_fpn(
        strides=anchor_strides,
        sizes=anchor_sizes,
        ratios=anchor_ratios,
        max_size=max_size,
    )
    flatten_anchors_per_level = [k.reshape((-1, 4)) for k in anchors_per_level]
    all_anchors_flatten = np.concatenate(flatten_anchors_per_level, axis=0)

    inside_ind, inside_anchors = filter_boxes_inside_shape(all_anchors_flatten, image.shape[:2])

    anchor_labels, anchor_gt_boxes = get_anchor_labels(
        inside_anchors,
        boxes[is_crowd == 0],
        boxes[is_crowd == 1],
        batch_per_image,
        front_ground_ratio,
        positive_anchor_threshold,
        negative_anchor_threshold,
        crowd_overlap_threshold,
    )

    # map back to all_anchors, then split to each level
    num_all_anchors = all_anchors_flatten.shape[0]
    all_labels = -np.ones((num_all_anchors,), dtype="int32")
    all_labels[inside_ind] = anchor_labels
    all_boxes = np.zeros((num_all_anchors, 4), dtype="float32")
    all_boxes[inside_ind] = anchor_gt_boxes

    start = 0
    end = 0
    multilevel_inputs = []
    for level_anchor in anchors_per_level:
        assert level_anchor.shape[2] == len(anchor_ratios)
        anchor_shape = level_anchor.shape[:3]  # fHxfWxNUM_ANCHOR_RATIOS
        num_anchor_this_level = np.prod(anchor_shape)
        end = start + num_anchor_this_level
        multilevel_inputs.append(
            (all_labels[start:end].reshape(anchor_shape), all_boxes[start:end, :].reshape(anchor_shape + (4,)))
        )
        start = end

    assert end == num_all_anchors, f"{end} != {num_all_anchors}"

    return multilevel_inputs


def get_anchor_labels(
    anchors: PixelValues,
    gt_boxes: PixelValues,
    crowd_boxes: PixelValues,
    batch_per_image: int,
    front_ground_ratio: float,
    positive_anchor_threshold: float,
    negative_anchor_threshold: float,
    crowd_overlap_threshold: float,
) -> (PixelValues, PixelValues):
    """
    Label each anchor as fg/bg/ignore.

    :param batch_per_image: total (across FPN levels) number of anchors that are marked valid
    :param crowd_boxes: Cx4 float
    :param gt_boxes: Bx4 float, non-crowd
    :param anchors: Ax4 float
    :param crowd_overlap_threshold: Anchors which overlap with a crowd box (IOA larger than threshold)
                                    will be ignored. Setting this to a value larger than 1.0 will disable the feature.
    :param negative_anchor_threshold: will filter all anchors with an IOU-threshold below this benchmark
    :param positive_anchor_threshold: will keep all anchors with an IOU-threshold above this benchmark
    :param front_ground_ratio: front ground ratio among selected RPN anchors

    :return:
        anchor_labels: (A,) int. Each element is {-1, 0, 1}
        anchor_boxes: Ax4. Contains the target gt_box for each anchor when the anchor is fg.
    """

    # This function will modify labels and return the filtered indices
    def filter_box_label(labels: np.array, value: np.array, max_num: int) -> np.array:
        curr_inds = np.where(labels == value)[0]
        if len(curr_inds) > max_num:
            disable_inds = np.random.choice(curr_inds, size=(len(curr_inds) - max_num), replace=False)
            labels[disable_inds] = -1  # ignore them
            curr_inds = np.where(labels == value)[0]
        return curr_inds

    number_anchors, number_gt_boxes = len(anchors), len(gt_boxes)
    if number_gt_boxes == 0:
        # No ground truth. All anchors are either background or ignored.
        anchor_labels = np.zeros((number_anchors,), dtype="int32")
        filter_box_label(anchor_labels, 0, batch_per_image)
        return anchor_labels, np.zeros((number_anchors, 4), dtype="float32")

    box_ious = np_iou(anchors, gt_boxes)  # NA x NB
    ious_argmax_per_anchor = box_ious.argmax(axis=1)  # NA,
    ious_max_per_anchor = box_ious.max(axis=1)
    ious_max_per_gt = np.amax(box_ious, axis=0, keepdims=True)  # 1xNB
    # for each gt, find all those anchors (including ties) that has the max ious with it
    anchors_with_max_iou_per_gt = np.where(box_ious == ious_max_per_gt)[0]

    # Setting NA labels: 1--fg 0--bg -1--ignore
    anchor_labels = -np.ones((number_anchors,), dtype="int32")  # NA,

    # the order of setting neg/pos labels matter
    anchor_labels[anchors_with_max_iou_per_gt] = 1
    anchor_labels[ious_max_per_anchor >= positive_anchor_threshold] = 1
    anchor_labels[ious_max_per_anchor < negative_anchor_threshold] = 0

    # label all non-ignore candidate boxes which overlap crowd as ignore
    if crowd_boxes.size > 0:
        cand_inds = np.where(anchor_labels >= 0)[0]
        cand_anchors = anchors[cand_inds]
        ioas = np_ioa(crowd_boxes, cand_anchors)
        overlap_with_crowd = cand_inds[ioas.max(axis=0) > crowd_overlap_threshold]
        anchor_labels[overlap_with_crowd] = -1

    # Subsample fg labels: ignore some fg if fg is too many
    target_num_fg = int(batch_per_image * front_ground_ratio)
    fg_inds = filter_box_label(anchor_labels, 1, target_num_fg)
    # Keep an image even if there is no foreground anchors
    # if len(fg_inds) == 0:
    #     raise MalformedData("No valid foreground for RPN!")

    # Subsample bg labels. num_bg is not allowed to be too many
    old_num_bg = np.sum(anchor_labels == 0)
    if old_num_bg == 0:
        # No valid bg in this image, skip.
        raise MalformedData("No valid background for RPN!")
    target_num_bg = batch_per_image - len(fg_inds)
    filter_box_label(anchor_labels, 0, target_num_bg)  # ignore return values

    # Set anchor boxes: the best gt_box for each fg anchor
    anchor_boxes = np.zeros((number_anchors, 4), dtype="float32")
    fg_boxes = gt_boxes[ious_argmax_per_anchor[fg_inds], :]
    anchor_boxes[fg_inds, :] = fg_boxes
    # assert len(fg_inds) + np.sum(anchor_labels == 0) == self.cfg.RPN.BATCH_PER_IM
    return anchor_labels, anchor_boxes
