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

from ...utils.file_utils import pytorch_available


def set_torch_auto_device() -> "torch.device":  # type: ignore
    """
    Returns cuda device if available, otherwise cpu
    """
    if pytorch_available():
        from torch import cuda, device  # pylint: disable=C0415, E0611

        return device("cuda" if cuda.is_available() else "cpu")
    raise ModuleNotFoundError
