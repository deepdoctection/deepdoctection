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
Dataset samples for pre-training and fine-tuning models

Place all datasets in a **deep**doctection's cache

    deepdoctection
    ├── datasets
    │ ├── dataset_1
    │ ├── dataset_2
    │ ├── dataset_3

If not sure:

    ```python
    print(dataset_instance.dataflow.get_workdir())
    ```
"""

from .doclaynet import *
from .fintabnet import *
from .funsd import *
from .iiitar13k import *
from .layouttest import *
from .publaynet import *
from .pubtables1m import *
from .pubtabnet import *
from .rvlcdip import *
from .xfund import *

__all__ = [
    "Publaynet",
    "Pubtabnet",
    "LayoutTest",
    "Fintabnet",
    "Xfund",
    "Funsd",
    "IIITar13K",
    "Pubtables1MDet",
    "Pubtables1MStruct",
    "Rvlcdip",
    "DocLayNet",
]
