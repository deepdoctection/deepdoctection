# -*- coding: utf-8 -*-
# File: d2utils.py

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
Detectron2 related utils
"""

import importlib.util

from ...utils.detection_types import Requirement

_DETECTRON2_AVAILABLE = importlib.util.find_spec("detectron2") is not None
_DETECTRON2_ERR_MSG = "Detectron2 must be installed: >>install-dd-pt"


def detectron2_available() -> bool:
    """
    Returns True if Detectron2 is installed
    """
    return bool(_DETECTRON2_AVAILABLE)


def get_detectron2_requirement() -> Requirement:
    """
    Returns Detectron2 requirement
    """
    return "detectron2", detectron2_available(), _DETECTRON2_ERR_MSG
