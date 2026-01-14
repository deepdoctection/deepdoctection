# -*- coding: utf-8 -*-
# File: xxx.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
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
Dataset base classes, dataflows, adapters etc.
"""

import sys
from typing import TYPE_CHECKING

from dd_core.utils.file_utils import _LazyModule, pytorch_available

__version__ = "1.0.5"
_IMPORT_STRUCTURE = {
    "base": ["DatasetBase", "SplitDataFlow", "MergeDataset", "DatasetCard", "CustomDataset"],
    "dataflow_builder": [
        "DataFlowBaseBuilder",
    ],
    "info": ["DatasetInfo", "DatasetCategories", "get_merged_categories"],
    "registry": ["get_dataset", "print_dataset_infos"],
    "save": ["dataflow_to_json"],
    "instances": [
        "DocLayNet",
        "DocLayNetSeq",
        "Fintabnet",
        "Funsd",
        "IIITar13K",
        "LayoutTest",
        "Publaynet",
        "Pubtables1MDet",
        "Pubtables1MStruct",
        "Pubtabnet",
        "Rvlcdip",
        "Xfund",
    ],
}

if pytorch_available():
    _IMPORT_STRUCTURE["adapter"] = ["DatasetAdapter"]


if TYPE_CHECKING:
    from .adapter import *
    from .base import *
    from .dataflow_builder import *
    from .info import *
    from .instances import *
    from .registry import *
    from .save import *

else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _IMPORT_STRUCTURE,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
