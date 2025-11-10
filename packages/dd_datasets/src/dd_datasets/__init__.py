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
dd_datasets: Dataset building and processing for deepdoctection

This package provides dataset management capabilities including:
- Dataflow: Efficient data loading and processing pipelines
- Mapper: Transformation functions for various dataset formats (COCO, Pascal VOC, etc.)
- Datasets: Built-in dataset definitions and builders
"""

__version__ = "1.0"

from .dataflow import *
from .datasets import *
from .mapper import *

