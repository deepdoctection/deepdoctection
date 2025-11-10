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
# Core data model for single pages

A fundamental architectural decision in the **deep**doctection framework is to convert all data, be it those from
datasets or the predictions from the pipelines, into a standardized format. This procedure leads to the following
simplifications:

- Data of the training environment can be transported in the production environment (i.e. through pipelines)
  without further adjustments.
- Datasets of different origins can be merged quickly, so that training data with greater variability arise.
- Pipeline environment components can be executed one after the other without conversion measures.

The disadvantage of carrying out any redundant transformations and thus experiencing a loss of performance is accepted.
After all, the point here is not to provide an optimal processing environment.
"""

from .annotation import *
from .box import *
from .convert import *
from .image import Image, MetaAnnotation
from .view import *
