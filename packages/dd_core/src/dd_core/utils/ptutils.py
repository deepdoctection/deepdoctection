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

from .env_info import ENV_VARS_TRUE

with try_import() as import_guard:
    import torch
    import torch.nn.functional as F


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


def get_num_gpu() -> int:
    """
    Get the number of available GPUs.

    Returns:
        int: Number of available GPUs.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


# Alias for backward compatibility
set_torch_auto_device = get_torch_device


def apply_torch_image(
    img: torch.Tensor,
    height: Union[int, float],
    width: Union[int, float],
    new_height: Union[int, float],
    new_width: Union[int, float],
    interp: str,
) -> torch.Tensor:
    """
    Apply the resize transformation to a PyTorch tensor image without using `viz_handler`.

    Args:
        img: Input image tensor of shape `[H, W]` or `[H, W, C]`, where `H` and `W` must
            match `height` and `width`. The tensor is expected to be on a valid device
            and of a floating or integer type supported by `torch.nn.functional.interpolate`.
        height: Original image height `H` that `img.shape[0]` must match.
        width: Original image width `W` that `img.shape[1]` must match.
        new_height: Target height for the resized image.
        new_width: Target width for the resized image.
        interp: Interpolation mode passed to `torch.nn.functional.interpolate`, e.g.
            `"nearest"`, `"bilinear"`, or `"bicubic"`.

    Returns:
        Resized image tensor of shape `[new_height, new_width]` or
        `[new_height, new_width, C]`, matching the channel layout of the input.

    Raises:
        AssertionError: If the input image spatial shape does not match
            `(height, width)`.
    """
    # Ensure spatial dimensions match
    assert tuple(img.shape[:2]) == (height, width)

    if img.ndim == 2:
        img = img.unsqueeze(-1)  # [H, W] -> [H, W, 1]

    # [H, W, C] -> [1, C, H, W]
    img_chw = img.permute(2, 0, 1).unsqueeze(0)

    ret = F.interpolate(
        img_chw,
        size=(new_height, new_width),
        mode=interp,
        align_corners=False if interp in {"bilinear", "bicubic"} else None,
    )
    ret = ret.squeeze(0).permute(1, 2, 0)

    if img.ndim == 3 and ret.ndim == 2:
        ret = ret.unsqueeze(-1)

    return ret
