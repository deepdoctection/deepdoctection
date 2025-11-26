# -*- coding: utf-8 -*-
# File: ptutils.py

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
Torch related utils
"""
from __future__ import annotations

import os
from typing import Optional, Union

from lazy_imports import try_import

from ...utils.env_info import ENV_VARS_TRUE

with try_import() as import_guard:
    import torch


def get_torch_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Select a device on which to load a model. The selection follows a cascade of priorities:

    If a device string is provided, it is used. If the environment variable `USE_CUDA` is set, a GPU is used.
    If more GPUs are available, it will use all of them unless something else is specified by `CUDA_VISIBLE_DEVICES`.

    See: <https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch>

    If an MPS device is available, it is used. Otherwise, the CPU is used.

    Args:
        device: Device either as string or torch.device.

    Returns:
        torch.device: The selected device.

    Note:
        The function checks the environment variables `USE_CUDA` and `USE_MPS` to determine device preference.
    """
    if device is not None:
        if isinstance(device, torch.device):
            return device
        if isinstance(device, str):
            return torch.device(device)
    if os.environ.get("USE_CUDA", "False") in ENV_VARS_TRUE:
        return torch.device("cuda")
    if os.environ.get("USE_MPS", "False") in ENV_VARS_TRUE:
        return torch.device("mps")
    return torch.device("cpu")
