# -*- coding: utf-8 -*-
# File: tfutils.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tensorflow related utils.
"""

from tensorpack.models import disable_layer_logging


def is_tfv2() -> bool:
    """
    Returns whether TF is operating in V2 mode.
    """
    try:
        from tensorflow.python import tf2  # pylint: disable=C0415

        return tf2.enabled()
    except ImportError:
        return False


def disable_tfv2() -> bool:
    """
    Disable TF in V2 mode.
    """
    try:
        import tensorflow as tf  # pylint: disable=C0415,E0401

        tfv1 = tf.compat.v1
        if is_tfv2():
            tfv1.disable_v2_behavior()
            tfv1.disable_eager_execution()
        return True
    except ModuleNotFoundError:
        return False


def disable_tp_layer_logging() -> None:
    """
    Disables TP layer logging, if not already set
    """
    disable_layer_logging()
