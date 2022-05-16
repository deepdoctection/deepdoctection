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
_N.C.TLINE = "TEXT_LINE"
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

_N.NLP.LANG.LANG = "LANGUAGE"
_N.NLP.LANG.ENG = "eng"
_N.NLP.LANG.RUS = "rus"
_N.NLP.LANG.DEU = "deu"
_N.NLP.LANG.FRE = "fre"
_N.NLP.LANG.ITA = "ita"
_N.NLP.LANG.JPN = "jpn"
_N.NLP.LANG.SPA = "spa"
_N.NLP.LANG.CEB = "ceb"
_N.NLP.LANG.TUR = "tur"
_N.NLP.LANG.POR = "por"
_N.NLP.LANG.UKR = "ukr"
_N.NLP.LANG.EPO = "epo"
_N.NLP.LANG.POL = "pol"
_N.NLP.LANG.SWE = "swe"
_N.NLP.LANG.DUT = "dut"
_N.NLP.LANG.HEB = "heb"
_N.NLP.LANG.CHI = "chi"
_N.NLP.LANG.HUN = "hun"
_N.NLP.LANG.ARA = "ara"
_N.NLP.LANG.CAT = "cat"
_N.NLP.LANG.FIN = "fin"
_N.NLP.LANG.CZE = "cze"
_N.NLP.LANG.PER = "per"
_N.NLP.LANG.SRP = "srp"
_N.NLP.LANG.GRE = "gre"
_N.NLP.LANG.VIE = "vie"
_N.NLP.LANG.BUL = "bul"
_N.NLP.LANG.KOR = "kor"
_N.NLP.LANG.NOR = "nor"
_N.NLP.LANG.MAC = "mac"
_N.NLP.LANG.RUM = "rum"
_N.NLP.LANG.IND = "ind"
_N.NLP.LANG.THA = "tha"
_N.NLP.LANG.ARM = "arm"
_N.NLP.LANG.DAN = "dan"
_N.NLP.LANG.TAM = "tam"
_N.NLP.LANG.HIN = "hin"
_N.NLP.LANG.HRV = "hrv"
_N.NLP.LANG.BEL = "bel"
_N.NLP.LANG.GEO = "geo"
_N.NLP.LANG.TEL = "tel"
_N.NLP.LANG.KAZ = "kaz"
_N.NLP.LANG.WAR = "war"
_N.NLP.LANG.LIT = "lit"
_N.NLP.LANG.GLG = "glg"
_N.NLP.LANG.SLO = "slo"
_N.NLP.LANG.BEN = "ben"
_N.NLP.LANG.BAQ = "baq"
_N.NLP.LANG.SLV = "slv"
_N.NLP.LANG.MAL = "mal"
_N.NLP.LANG.MAR = "mar"
_N.NLP.LANG.EST = "est"
_N.NLP.LANG.AZE = "aze"
_N.NLP.LANG.ALB = "alb"
_N.NLP.LANG.LAT = "lat"
_N.NLP.LANG.BOS = "bos"
_N.NLP.LANG.NNO = "nno"
_N.NLP.LANG.URD = "urd"


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
