# -*- coding: utf-8 -*-
# File: __init__.py

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
External package for wrapping blocks that are imported via external packages.

It contains modules that origins from projects of other repos by cloning, but are not parts of the packages themselves
(e.g. Tensorpack Mask-RCNN). The source code was copied for this and interfaces for calling the code were adapted.

Abstract classes are also defined here which, as adapters, facilitate the transition from the original module to the
DD package.

Functions that, as wrappers, standardize the transition from external API to DD API.
"""
from ..utils.file_utils import tensorpack_available
from .base import *
from .common import *
from .d2detect import *
from .doctrocr import *
from .hflayoutlm import *
from .model import *
from .pdftext import *
from .tessocr import *
from .texocr import *  # type: ignore

if tensorpack_available():
    from .tpdetect import *
