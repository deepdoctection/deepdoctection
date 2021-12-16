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
Init file for mapper package. Contains everything that is related to transformation between datapoints

"""
from typing import Callable, Optional

from .cats import *
from .cocostruct import *
from .match import *
from .misc import *
from .pagestruct import *
from .prodigystruct import *
from .pubstruct import *
from .tpstruct import *
from .utils import *
from ..datapoint.image import Image

__all__ = [
    "cat_to_sub_cat",
    "filter_cat",
    "image_to_cat_id",
    "remove_cats",
    "coco_to_image",
    "image_to_coco",
    "match_anns_by_intersection",
    "to_image",
    "maybe_load_image",
    "maybe_remove_image",
    "image_ann_to_image",
    "maybe_ann_to_sub_image",
    "to_page",
    "page_dict_to_page",
    "prodigy_to_image",
    "image_to_prodigy",
    "pub_to_image",
    "image_to_tp_frcnn_training",
    "MappingContextManager",
    "DefaultMapper",
    "cur",
    "LabelSummarizer",
]

# Mapper
Mapper = Callable[[Image], Optional[Image]]
