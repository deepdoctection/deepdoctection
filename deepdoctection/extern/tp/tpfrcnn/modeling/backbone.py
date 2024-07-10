# -*- coding: utf-8 -*-
# File: backbone.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
This file is modified from
<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/modeling/backbone.py>
"""

from contextlib import ExitStack, contextmanager

import numpy as np
from lazy_imports import try_import

# pylint: disable=import-error

with try_import() as import_guard:
    import tensorflow as tf
    from tensorpack import tfv1
    from tensorpack.models import BatchNorm, Conv2D, MaxPooling, layer_register
    from tensorpack.tfutils import argscope
    from tensorpack.tfutils.varreplace import custom_getter_scope, freeze_variables

# pylint: enable=import-error

if not import_guard.is_successful():
    from .....utils.mocks import layer_register


@layer_register(log_shape=True)
def GroupNorm(x, group=32, gamma_initializer=None):
    """
    More code that reproduces the paper can be found at <https://github.com/ppwwyyxx/GroupNorm-reproduce/>.
    """
    if gamma_initializer is None:
        gamma_initializer = tf.constant_initializer(1.0)
    shape = x.get_shape().as_list()
    ndims = len(shape)
    assert ndims == 4, shape
    chan = shape[1]
    assert chan % group == 0, chan
    group_size = chan // group

    orig_shape = tf.shape(x)
    h, w = orig_shape[2], orig_shape[3]

    x = tf.reshape(x, tf.stack([-1, group, group_size, h, w]))

    mean, var = tfv1.nn.moments(x, [2, 3, 4], keep_dims=True)

    new_shape = [1, group, group_size, 1, 1]

    beta = tfv1.get_variable("beta", [chan], initializer=tf.constant_initializer())
    beta = tf.reshape(beta, new_shape)

    gamma = tfv1.get_variable("gamma", [chan], initializer=gamma_initializer)
    gamma = tf.reshape(gamma, new_shape)

    out = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5, name="output")
    return tf.reshape(out, orig_shape, name="output")


def freeze_affine_getter(getter, *args, **kwargs):
    """
    Custom getter to freeze affine params inside bn
    """

    name = args[0] if len(args) else kwargs.get("name")  # pylint: disable=C1801
    if name.endswith("/gamma") or name.endswith("/beta"):
        kwargs["trainable"] = False
        ret = getter(*args, **kwargs)
        tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, ret)  # pylint: disable=E1101
    else:
        ret = getter(*args, **kwargs)
    return ret


def maybe_reverse_pad(cfg, topleft, bottomright):
    """
    Returning chosen pad mode
    """
    if cfg.BACKBONE.TF_PAD_MODE:
        return [topleft, bottomright]
    return [bottomright, topleft]


@contextmanager
def backbone_scope(cfg, freeze):
    """
    Context scope for setting up the backbone

    :param cfg: The configuration instance as an AttrDict
    :param freeze: (bool) whether to freeze all the variables under the scope
    """

    def nonlin(x):
        x = get_norm(cfg)(x)
        return tf.nn.relu(x)

    with argscope([Conv2D, MaxPooling, BatchNorm], data_format="channels_first"), argscope(
        Conv2D,
        use_bias=False,
        activation=nonlin,
        kernel_initializer=tfv1.variance_scaling_initializer(scale=2.0, mode="fan_out"),
    ), ExitStack() as stack:
        if cfg.BACKBONE.NORM in ["FreezeBN", "SyncBN"]:
            if freeze or cfg.BACKBONE.NORM == "FreezeBN":
                stack.enter_context(argscope(BatchNorm, training=False))
            else:
                stack.enter_context(
                    argscope(BatchNorm, sync_statistics="nccl" if cfg.TRAINER == "replicated" else "horovod")
                )

        if freeze:
            stack.enter_context(freeze_variables(stop_gradient=False, skip_collection=True))
        else:
            # the layers are not completely freezed, but we may want to only freeze the affine
            if cfg.BACKBONE.FREEZE_AFFINE:
                stack.enter_context(custom_getter_scope(freeze_affine_getter))
        yield


def image_preprocess(image, cfg):
    """
    Preprocessing image by rescaling.

    :param image: tf.Tensor
    :param cfg: config
    :return: tf.Tensor
    """
    with tf.name_scope("image_preprocess"):
        if image.dtype.base_dtype != tf.float32:
            image = tf.cast(image, tf.float32)

        mean = cfg.PREPROC.PIXEL_MEAN
        std = np.asarray(cfg.PREPROC.PIXEL_STD)
        image_mean = tf.constant(mean, dtype=tf.float32)
        image_invstd = tf.constant(1.0 / std, dtype=tf.float32)
        image = (image - image_mean) * image_invstd
        return image


def get_norm(cfg, zero_init=False):
    """
    Return a norm with respect to config.

    :param cfg: config
    :param zero_init: choosing zero initializer or None
    :return: lambda func with norm applied to tf.Tensor
    """
    if cfg.BACKBONE.NORM == "None":
        return lambda x: x
    if cfg.BACKBONE.NORM == "GN":
        norm = GroupNorm
        layer_name = "gn"
    else:
        norm = BatchNorm
        layer_name = "bn"
    return lambda x: norm(layer_name, x, gamma_initializer=tf.zeros_initializer() if zero_init else None)


def resnet_shortcut(l, n_out, stride, activation=None):
    """
    Defining the skip connection in bottleneck

    :param l: tf.Tensor
    :param n_out: output dim
    :param stride: stride
    :param activation: An activation function
    :return: tf.Tensor
    """
    if activation is None:
        activation = tf.identity
    n_in = l.shape[1]
    if n_in != n_out:  # change dimension when channel is not the same
        return Conv2D("convshortcut", l, n_out, 1, strides=stride, activation=activation)  # pylint: disable=E1124
    return l


def resnet_bottleneck(l, ch_out, stride, cfg):
    """
    Defining the bottleneck for Resnet50/101 backbones

    :param l: tf.Tensor
    :param ch_out: number of output features
    :param stride: stride
    :param cfg: config
    :return: tf.Tensor
    """
    shortcut = l

    l = Conv2D("conv1", l, ch_out, 1, strides=1)  # pylint: disable=E1124
    if stride == 2:
        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(cfg, 0, 1), maybe_reverse_pad(cfg, 0, 1)])
        l = Conv2D("conv2", l, ch_out, 3, strides=2, padding="VALID")  # pylint: disable=E1124
    else:
        l = Conv2D("conv2", l, ch_out, 3, strides=stride)  # pylint: disable=E1124
    if cfg.BACKBONE.NORM != "None":
        l = Conv2D("conv3", l, ch_out * 4, 1, activation=get_norm(cfg, zero_init=True))
    else:
        l = Conv2D("conv3", l, ch_out * 4, 1, activation=tf.identity, kernel_initializer=tf.constant_initializer())
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_norm(cfg))
    return tf.nn.relu(ret, name="output")


def resnext32x4d_bottleneck(l, ch_out, stride, cfg):
    """
    Defining Resnext bottleneck <https://arxiv.org/abs/1611.05431>

    :param l: tf.Tensor
    :param ch_out: number of output features
    :param stride: stride
    :param cfg: config
    :return: tf.Tensor
    """
    shortcut = l
    l = Conv2D("conv1", l, ch_out * 2, 1, stride=1)
    l = Conv2D("conv2", l, ch_out * 2, 3, stride=stride, split=32)
    l = Conv2D("conv3", l, ch_out * 4, 1, activation=get_norm(cfg, zero_init=True))
    ret = l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_norm(cfg))
    return tf.nn.relu(ret, name="output")


def resnet_group(name, l, block_func, features, count, stride, cfg):
    """
    Defining resnet groups

    :param name: name of group
    :param l: tf.Tensor
    :param block_func: the bottleneck function
    :param features: number of features
    :param count: number of successive bottleneck function
    :param stride: stride
    :param cfg: config
    :return: tf.Tensor
    """
    with tfv1.variable_scope(name):
        for i in range(0, count):
            with tfv1.variable_scope(f"block{i}"):
                l = block_func(l, features, stride if i == 0 else 1, cfg)
    return l


def resnet_fpn_backbone(image, cfg):
    """
    Setup of the full FPN backbone.

    :param image: tf.Tensor
    :param cfg: config
    :return: tf.Tensor for level c2,c3,c4,c5
    """
    num_blocks = cfg.BACKBONE.RESNET_NUM_BLOCKS
    freeze_at = cfg.BACKBONE.FREEZE_AT
    shape2d = tf.shape(image)[2:]
    mult = float(cfg.FPN.RESOLUTION_REQUIREMENT)
    new_shape2d = tf.cast(tf.math.ceil(tf.cast(shape2d, tf.float32) / mult) * mult, tf.int32)
    pad_shape2d = new_shape2d - shape2d
    assert len(num_blocks) == 4, num_blocks
    with backbone_scope(cfg, freeze=freeze_at > 0):
        chan = image.shape[1]
        pad_base = maybe_reverse_pad(cfg, 2, 3)
        l = tf.pad(
            image,
            tf.stack(
                [
                    [0, 0],
                    [0, 0],
                    [pad_base[0], pad_base[1] + pad_shape2d[0]],
                    [pad_base[0], pad_base[1] + pad_shape2d[1]],
                ]
            ),
        )
        l.set_shape([None, chan, None, None])
        l = Conv2D("conv0", l, 64, 7, strides=2, padding="VALID")  # pylint: disable=E1124
        l = tf.pad(l, [[0, 0], [0, 0], maybe_reverse_pad(cfg, 0, 1), maybe_reverse_pad(cfg, 0, 1)])
        l = MaxPooling("pool0", l, 3, strides=2, padding="VALID")  # pylint: disable=E1124

    bottleneck = resnet_bottleneck if cfg.BACKBONE.BOTTLENECK == "resnet" else resnext32x4d_bottleneck
    with backbone_scope(cfg=cfg, freeze=freeze_at > 1):
        c2 = resnet_group("group0", l, bottleneck, 64, num_blocks[0], 1, cfg)
    with backbone_scope(cfg=cfg, freeze=False):
        c3 = resnet_group("group1", c2, bottleneck, 128, num_blocks[1], 2, cfg)
        c4 = resnet_group("group2", c3, bottleneck, 256, num_blocks[2], 2, cfg)
        c5 = resnet_group("group3", c4, bottleneck, 512, num_blocks[3], 2, cfg)

    # 32x downsampling up to now
    # size of c5: ceil(input/32)
    return c2, c3, c4, c5
