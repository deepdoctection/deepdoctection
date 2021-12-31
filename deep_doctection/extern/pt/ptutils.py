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

import importlib.util

from ...utils.detection_types import Requirement

_PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYTORCH_ERR_MSG = "Pytorch must be installed: https://pytorch.org/get-started/locally/#linux-pip"


def pytorch_available() -> bool:
    """
    Returns True if Pytorch is installed
    """
    return bool(_PYTORCH_AVAILABLE)


def get_pytorch_requirement() -> Requirement:
    """
    Returns HF Pytorch requirement
    """
    return "torch", pytorch_available(), _PYTORCH_ERR_MSG


def set_torch_auto_device() -> "torch.device":
    """
    Returns cuda device if available, otherwise cpu
    """
    if pytorch_available():
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        raise ModuleNotFoundError
