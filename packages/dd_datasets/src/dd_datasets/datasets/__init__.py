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
# Dataset concept: Building, training and evaluating datasets

Simple framework inspired by <https://huggingface.co/docs/datasets/> for creating datasets.

"""

from .adapter import *
from .base import *
from .dataflow_builder import DataFlowBaseBuilder
from .info import *
from .instances import *
from .registry import *
from .save import *
