# -*- coding: utf-8 -*-
# File: settings.py

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
Module for funcs and constants that maintain general settings
"""

import os

from ..utils.metacfg import AttrDict

# naming convention for all categories and NER tags
names = AttrDict()

_N = names

_N.C.TAB = "TABLE"
_N.C.FIG = "FIGURE"
_N.C.LIST = "LIST"
_N.C.TEXT = "TEXT"
_N.C.TITLE = "TITLE"
_N.C.CELL = "CELL"
_N.C.HEAD = "HEAD"
_N.C.BODY = "BODY"
_N.C.ITEM = "ITEM"
_N.C.ROW = "ROW"
_N.C.COL = "COLUMN"
_N.C.RN = "ROW_NUMBER"
_N.C.CN = "COLUMN_NUMBER"
_N.C.RS = "ROW_SPAN"
_N.C.CS = "COLUMN_SPAN"
_N.C.NR = "NUMBER_ROWS"
_N.C.NC = "NUMBER_COLUMNS"
_N.C.NRS = "MAX_ROW_SPAN"
_N.C.NCS = "MAX_COLUMN_SPAN"
_N.C.WORD = "WORD"
_N.C.CHARS = "CHARS"
_N.C.BLOCK = "BLOCK"
_N.C.LINE = "LINE"
_N.C.CHILD = "CHILD"
_N.C.HTAB = "HTML_TABLE"
_N.C.RO = "READING_ORDER"
_N.C.LOGO = "LOGO"
_N.C.SIGN = "SIGNATURE"

_N.C.SEL = "SEMANTIC_ENTITY_LINK"
_N.C.SE = "SEMANTIC_ENTITY"
_N.C.Q = "QUESTION"
_N.C.A = "ANSWER"
_N.C.O = "OTHER"

_N.NER.TAG = "NER_TAG"
_N.NER.O = "O"
_N.NER.B = "B"
_N.NER.I = "I"

_N.freeze()

# Some path settings

# package path
FILE_PATH = os.path.split(__file__)[0]
PATH = os.path.dirname(os.path.dirname(FILE_PATH))

# model cache directory
dd_cache_home = os.path.expanduser(os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "deepdoctection"))
MODEL_DIR = os.path.join(dd_cache_home, "weights")

# configs cache directory
CONFIGS = os.path.join(dd_cache_home, "configs")

# dataset cache directory
DATASET_DIR = os.path.join(dd_cache_home, "datasets")
