# -*- coding: utf-8 -*-
# File: box_ops.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Taken from

<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/utils/box_ops.py>

and

<https://github.com/tensorflow/models/blob/master/object_detection/core/box_list_ops.py>
"""

# pylint: disable=import-error

from lazy_imports import try_import

with try_import() as tf_import_guard:
    import tensorflow as tf
    from tensorpack.tfutils.scope_utils import under_name_scope

if not tf_import_guard.is_successful():
    from .....utils.mocks import under_name_scope


# pylint: enable=import-error


@under_name_scope()
def area(boxes):
    """
    :param boxes: nx4 floatbox

    :return: n
    """

    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


@under_name_scope()
def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    :param boxlist1: Nx4 floatbox
    :param boxlist2: Mx4

    :return: A tensor with shape [N, M] representing pairwise intersections
    """

    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    :param boxlist1: Nx4 floatbox
    :param boxlist2: Mx4

    :return: A tensor with shape [N, M] representing pairwise iou scores.
    """

    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections
    return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(intersections), tf.truediv(intersections, unions))
