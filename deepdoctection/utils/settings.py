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
from typing import Union, Tuple, Any, Dict
from pathlib import Path
from enum import Enum

import catalogue  # type: ignore

from ..utils.metacfg import AttrDict


class ObjectTypes(str, Enum):

    def __repr__(self) -> str:
        return '<%s.%s>' % (self.__class__.__name__, self.name)

    @classmethod
    def from_value(cls, value: str) -> "ObjectTypes":
        for member in cls.__members__.values():
            if member.value == value:
                return member
        raise ValueError("value %s does not have corresponding member", value)


TypeOrStr = Union[ObjectTypes,str]

object_types_registry = catalogue.create("deepdoctection", "settings", entry_points=True)


@object_types_registry.register("DefaultType")
class DefaultType(ObjectTypes):
    default_type = "DEFAULT_TYPE"


@object_types_registry.register("PageType")
class PageType(ObjectTypes):
    document_type = "DOCUMENT_TYPE"  # was previously: "DOC_CLASS"
    language = "LANGUAGE"


@object_types_registry.register("SummaryType")
class SummaryType(ObjectTypes):
    summary = "SUMMARY"


@object_types_registry.register("DocumentType")
class DocumentType(ObjectTypes):
    letter = "LETTER"
    form = "FORM"
    email = "EMAIL"
    handwritten = "HANDWRITTEN"
    advertisment = "ADVERTISMENT"
    scientific_report = "SCIENTIFIC REPORT"
    scientific_publication = "SCIENTIFIC PUBLICATION"
    specification = "SPECIFICATION"
    file_folder = "FILE FOLDER"
    news_article = "NEWS ARTICLE"
    budget = "BUDGET"
    invoice = "INVOICE"
    presentation = "PRESENTATION"
    questionnaire = "QUESTIONNAIRE"
    resume = "RESUME"
    memo = "MEMO"
    financial_report = "FINANCIAL_REPORT"
    laws_and_regulations = "LAWS_AND_REGULATIONS"
    government_tenders = "GOVERNMENT_TENDERS"
    manuals = "MANUALS"
    patents = "PATENTS"


@object_types_registry.register("LayoutType")
class LayoutType(ObjectTypes):
    table = "TABLE"
    figure = "FIGURE"
    list = "LIST"
    text = "TEXT"
    title = "TITLE"  # type: ignore
    logo = "LOGO"
    signature = "SIGNATURE"
    caption = "CAPTION"
    footnote = "FOOTNOTE"
    formula = "FORMULA"
    page_footer = "PAGE-FOOTER"
    page_header = "PAGE-HEADER"
    section_header = "SECTION_HEADER"
    page = "PAGE"
    cell = "CELL"
    row = "ROW"
    column = "COLUMN"
    word = "WORD"
    line = "LINE"


@object_types_registry.register("TableType")
class TableType(ObjectTypes):
    item = "ITEM"
    number_of_rows = "NUMBER_OF_ROWS"
    number_of_columns = "NUMBER_OF_COLUMNS"
    max_row_span = "MAX_ROW_SPAN"
    max_col_span = "MAX_COL_SPAN"
    html = "HTML"


@object_types_registry.register("CellType")
class CellType(ObjectTypes):
    header = "HEADER"
    body = "BODY"
    row_number = "ROW_NUMBER"
    column_number = "COLUMN_NUMBER"
    row_span = "ROW_SPAN"
    column_span = "COLUMN_SPAN"


@object_types_registry.register("WordType")
class WordType(ObjectTypes):
    characters = "CHARACTERS"
    block = "BLOCK"
    token_class = "TOKEN_CLASS"  # was previously: "SEMANTIC_ENTITY"
    tag = "BIO_TAG"  # was previously: "NER_TAG"
    token_tag = "TOKEN_TAG"  # was previously: "NER_TOKEN"
    text_line = "TEXT_LINE"


@object_types_registry.register("TokenClasses")
class TokenClasses(ObjectTypes):
    header = "HEADER"
    question = "QUESTION"
    answer = "ANSWER"
    other = "OTHER"


@object_types_registry.register("BioTag")
class BioTag(ObjectTypes):
    begin = "B"
    inside = "I"
    outside = "O"


@object_types_registry.register("TokenClassWithTag")
class TokenClassWithTag(ObjectTypes):
    b_answer = "B-ANSWER"
    b_header = "B-HEADER"
    b_question = "B-QUESTION"
    e_answer = "E-ANSWER"
    e_header = "E-HEADER"
    e_question = "E-QUESTION"
    i_answer = "I-ANSWER"
    i_header = "I-HEADER"
    i_question = "I-QUESTION"
    s_answer = "S-ANSWER"
    s_header = "S-HEADER"
    s_question = "S-QUESTION"


_TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG={(TokenClasses.header,BioTag.begin): TokenClassWithTag.b_header,
                                        (TokenClasses.header,BioTag.inside): TokenClassWithTag.i_header,
                                        (TokenClasses.answer, BioTag.begin): TokenClassWithTag.b_answer,
                                        (TokenClasses.answer, BioTag.inside): TokenClassWithTag.i_answer,
                                        (TokenClasses.question, BioTag.begin): TokenClassWithTag.b_question,
                                        (TokenClasses.question, BioTag.inside): TokenClassWithTag.i_question,
                                        (TokenClasses.other,BioTag.outside): BioTag.outside
                                        }


def token_class_tag_to_token_class_with_tag(token: ObjectTypes,
                                            tag: ObjectTypes) -> ObjectTypes:
    if isinstance(token, TokenClasses) and isinstance(tag, BioTag):
        return _TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG[(token,tag)]
    raise TypeError("Token must be of type TokenClasses and tag must be of type BioTag")


def token_class_with_tag_to_token_class_and_tag(token_class_with_tag: ObjectTypes) -> Tuple[ObjectTypes, ObjectTypes]:
    return {val: key for key,val in _TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG.items()}[token_class_with_tag]


@object_types_registry.register("Relationships")
class Relationships(ObjectTypes):
    child = "CHILD"
    reading_order = "READING_ORDER"
    semantic_entity_link = "SEMANTIC_ENTITY_LINK"


@object_types_registry.register("Languages")
class Languages(ObjectTypes):
    english = "eng"
    russian = "rus"
    german = "deu"
    french = "fre"
    italian = "ita"
    japanese = "jpn"
    spanish = "spa"
    cebuano = "ceb"
    turkish = "tur"
    portuguese = "por"
    ukrainian = "ukr"
    esperanto = "epo"
    polish = "pol"
    swedish = "swe"
    dutch = "dut"
    hebrew = "heb"
    chinese = "chi"
    hungarian = "hun"
    arabic = "ara"
    catalan = "cat"
    finnish = "fin"
    czech = "cze"
    persian = "per"
    serbian = "srp"
    greek = "gre"
    vietnamese = "vie"
    bulgarian = "bul"
    korean = "kor"
    norwegian = "nor"
    macedonian = "mac"
    romanian = "rum"
    indonesian = "ind"
    thai = "tha"
    armenian = "arm"
    danish = "dan"
    tamil = "tam"
    hindi = "hin"
    croatian = "hrv"
    belarusian = "bel"
    georgian = "geo"
    telugu = "tel"
    kazakh = "kaz"
    waray = "war"
    lithuanian = "lit"
    scottish = "glg"
    slovak = "slo"
    benin = "ben"
    basque = "baq"
    slovenian = "slv"
    malayalam = "mal"
    marathi = "mar"
    estonian = "est"
    azerbaijani = "aze"
    albanian = "alb"
    latin = "lat"
    bosnian = "bos"
    norwegian_nynorsk = "nno"
    urdu = "urd"
    not_defined = "nn"


@object_types_registry.register("DatasetType")
class DatasetType(ObjectTypes):
    object_detection = "OBJECT_DETECTION"
    sequence_classification = "SEQUENCE_CLASSIFICATION"
    token_classification = "TOKEN_CLASSIFICATION"
    publaynet = "PUBLAYNET"


_ALL_TYPES_DICT = {}
_ALL_TYPES = set(object_types_registry.get_all().values())
for ob in _ALL_TYPES:
    _ALL_TYPES_DICT.update({e.value: e for e in ob})


def update_all_types_dict():
    maybe_new_types =  set(object_types_registry.get_all().values())
    difference = maybe_new_types - _ALL_TYPES
    for ob in difference:
        _ALL_TYPES_DICT.update({e.value: e for e in ob})


def get_type(obj_type: Union[str, ObjectTypes]) -> ObjectTypes:
    if isinstance(obj_type, ObjectTypes):
        return obj_type
    return _ALL_TYPES_DICT[obj_type]


# naming convention for all categories and NER tags
#names = AttrDict()

_N = AttrDict() #names

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
_N.C.CAP = "CAPTION"
_N.C.FOOT = "FOOTNOTE"
_N.C.FORMULA = "FORMULA"
_N.C.PFOOT = "PAGE-FOOTER"
_N.C.PHEAD = "PAGE-HEADER"
_N.C.SECH = "SECTION-HEADER"
_N.C.PAGE = "PAGE"

_N.C.SEL = "SEMANTIC_ENTITY_LINK"
_N.C.SE = "SEMANTIC_ENTITY"
_N.C.Q = "QUESTION"
_N.C.A = "ANSWER"
_N.C.O = "OTHER"

_N.C.DOC = "DOC_CLASS"

_N.C.LET = "LETTER"
_N.C.FORM = "FORM"
_N.C.EM = "EMAIL"
_N.C.HW = "HANDWRITTEN"
_N.C.AD = "ADVERTISMENT"
_N.C.SR = "SCIENTIFIC REPORT"
_N.C.SP = "SCIENTIFIC PUBLICATION"
_N.C.SPEC = "SPECIFICATION"
_N.C.FF = "FILE FOLDER"
_N.C.NA = "NEWS ARTICLE"
_N.C.BU = "BUDGET"
_N.C.INV = "INVOICE"
_N.C.PRES = "PRESENTATION"
_N.C.QUEST = "QUESTIONNAIRE"
_N.C.RES = "RESUME"
_N.C.MEM = "MEMO"
_N.C.FR = "FINANCIAL_REPORTS"
_N.C.LR = "LAWS_AND_REGULATIONS"
_N.C.GT = "GOVERNMENT_TENDERS"
_N.C.MAN = "MANUALS"
_N.C.PAT = "PATENTS"

_N.NER.TAG = "NER_TAG"
_N.NER.O = "O"
_N.NER.B = "B"
_N.NER.I = "I"

_N.NER.TOK = "NER_TOKEN"
_N.NER.B_A = "B-ANSWER"
_N.NER.B_H = "B-HEAD"
_N.NER.B_Q = "B-QUESTION"
_N.NER.I_A = "I-ANSWER"
_N.NER.I_H = "I-HEAD"
_N.NER.I_Q = "I-QUESTION"

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

_N.DS.TYPE.OBJ = "OBJECT_DETECTION"
_N.DS.TYPE.SEQ = "SEQUENCE_CLASSIFICATION"
_N.DS.TYPE.TOK = "TOKEN_CLASSIFICATION"

_N.freeze()

# Some path settings

# package path
file_path = Path(os.path.split(__file__)[0])
PATH = file_path.parent.parent

# model cache directory
dd_cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "deepdoctection"
MODEL_DIR = dd_cache_home / "weights"

# configs cache directory
CONFIGS = dd_cache_home / "configs"

# dataset cache directory
DATASET_DIR = dd_cache_home / "datasets"

FILE_PATH = os.path.split(__file__)[0]
TPATH = os.path.dirname(os.path.dirname(FILE_PATH))
