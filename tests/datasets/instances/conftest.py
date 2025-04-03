# -*- coding: utf-8 -*-
# File: conftest.py

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
Module fixtures
"""

from typing import Optional

import numpy as np

from deepdoctection.utils.types import PixelValues


def get_white_image(path: str) -> Optional[PixelValues]:
    """
    white image
    """
    if path:
        return np.ones((794, 596, 3), dtype=np.uint8) * 255  # type: ignore
    return None


def get_white_image_pdf(path: str, dpi: int) -> Optional[PixelValues]:  # pylint: disable=W0613
    """
    white image
    """
    if path:
        return np.ones((794, 596, 3), dtype=np.uint8) * 255  # type: ignore
    return None
