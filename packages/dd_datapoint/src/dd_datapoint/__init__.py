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
# dd_datapoint: Core Data Structures for deepdoctection

This package provides the foundational data structures and utilities for the deepdoctection ecosystem.
It includes:

- **utils**: Core utility functions and helpers
- **datapoint**: Data models for annotations, bounding boxes, images, and views

This is the minimal package needed for client-side processing of deepdoctection data structures.
"""

__version__ = "1.0"

# Import datapoint structures
from .datapoint import *

# Import key utilities
from .utils import *
