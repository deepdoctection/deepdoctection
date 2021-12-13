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

import importlib.util

from ...utils.detection_types import Requirement


_TF_AVAILABLE = False

try:
    _TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
except ValueError:
    pass

_TF_ERR_MSG = "Tensorflow >=2.4.1 must be installed: https://www.tensorflow.org/install/gpu"

_TP_AVAILABLE = importlib.util.find_spec("tensorpack") is not None
_TP_ERR_MSG = "Tensorpack must be installed: >>make install-tf-dependencies"


def tf_available() -> bool:
    """
    Returns True if TF is installed
    """
    return bool(_TF_AVAILABLE)


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


def tensorpack_available() -> bool:
    """
    Returns True if Tensorpack is installed
    """
    return bool(_TP_AVAILABLE)


def get_tensorpack_requirement() -> Requirement:
    """
    Returns Tensorpack requirement
    """
    return "tensorpack", tensorpack_available(), _TP_ERR_MSG
