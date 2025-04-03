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
Init module for train package
"""

from ..utils.file_utils import detectron2_available, tensorpack_available, transformers_available

if detectron2_available():
    from .d2_frcnn_train import train_d2_faster_rcnn

if transformers_available():
    from .hf_detr_train import train_hf_detr
    from .hf_layoutlm_train import train_hf_layoutlm

if tensorpack_available():
    from .tp_frcnn_train import train_faster_rcnn
