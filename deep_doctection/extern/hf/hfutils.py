# -*- coding: utf-8 -*-
# File: hfutils.py

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
HF transformers related utils
"""

import importlib.util

from ...utils.detection_types import Requirement

_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
_TRANSFORMERS_ERR_MSG = "Transformers must be installed: >>make install-transformers-dependencies"


def transformers_available() -> bool:
    """
    Returns True if HF Transformers is installed
    """
    return bool(_TRANSFORMERS_AVAILABLE)


def get_transformers_requirement() -> Requirement:
    """
    Returns HF Transformers requirement
    """
    return "transformers", transformers_available(), _TRANSFORMERS_ERR_MSG
