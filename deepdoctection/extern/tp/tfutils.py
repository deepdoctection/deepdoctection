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

from __future__ import annotations

import os
from typing import ContextManager, Optional, Union

from lazy_imports import try_import

from ...utils.env_info import ENV_VARS_TRUE

with try_import() as import_guard:
    from tensorpack.models import disable_layer_logging  # pylint: disable=E0401

with try_import() as tf_import_guard:
    import tensorflow as tf  # pylint: disable=E0401


def is_tfv2() -> bool:
    """
    Returns whether TensorFlow is operating in V2 mode.

    Returns:
        Whether TensorFlow is operating in V2 mode.

    Example:
        ```python
        is_tfv2()
        ```
    """
    try:
        from tensorflow.python import tf2  # pylint: disable=C0415

        return tf2.enabled()
    except ImportError:
        return False


def disable_tfv2() -> bool:
    """
    Disables TensorFlow V2 mode.

    Returns:
        Whether TensorFlow V2 mode was disabled.

    Example:
        ```python
        disable_tfv2()
        ```
    """

    tfv1 = tf.compat.v1
    if is_tfv2():
        tfv1.disable_v2_behavior()
        tfv1.disable_eager_execution()
        return True
    return False


def disable_tp_layer_logging() -> None:
    """
    Disables tensorpack layer logging, if not already set.

    Example:
        ```python
        disable_tp_layer_logging()
        ```
    """
    disable_layer_logging()


def get_tf_device(device: Optional[Union[str, tf.device]] = None) -> tf.device:
    """
    Selects a device on which to load a model. The selection follows a cascade of priorities:

    - If a `device` string is provided, it is used. If the string is "cuda" or "GPU", the first GPU is used.
    - If the environment variable `USE_CUDA` is set, a GPU is used. If more GPUs are available it will use the first
      one.

    Args:
        device: Device string.

    Returns:
        TensorFlow device.

    Raises:
        EnvironmentError: If `USE_CUDA` is set but no GPU device is found, or if no CPU device is found.
    """
    if device is not None:
        if isinstance(device, ContextManager):
            return device
        if isinstance(device, str):
            if device in ("cuda", "GPU"):
                device_names = [device.name for device in tf.config.list_logical_devices(device_type="GPU")]
                return tf.device(device_names[0].name)
            # The input must be something sensible
            return tf.device(device)
    if os.environ.get("USE_CUDA", "False") in ENV_VARS_TRUE:
        device_names = [device.name for device in tf.config.list_logical_devices(device_type="GPU")]
        if not device_names:
            raise EnvironmentError(
                "USE_CUDA is set but tf.config.list_logical_devices cannot find anyx device. "
                "It looks like there is an issue with your Tensorlfow installation. "
                "You can LOG_LEVEL='DEBUG' to get more information about installation."
            )
        return tf.device(device_names[0])
    device_names = [device.name for device in tf.config.list_logical_devices(device_type="CPU")]
    if not device_names:
        raise EnvironmentError(
            "Cannot find any CPU device. It looks like there is an issue with your "
            "Tensorflow installation. You can LOG_LEVEL='DEBUG' to get more information about "
            "installation."
        )
    return tf.device(device_names[0])
