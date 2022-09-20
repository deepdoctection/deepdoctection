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
from enum import Enum
from pathlib import Path
from typing import Tuple, Union

import catalogue  # type: ignore


class ObjectTypes(str, Enum):
    """Base Class for describing objects as attributes of Enums"""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}>"

    @classmethod
    def from_value(cls, value: str) -> "ObjectTypes":
        """Getting the enum member from a given string value

        :param value: string value to get the enum member
        :return: Enum member
        """
        for member in cls.__members__.values():
            if member.value == value:
                return member
        raise ValueError(f"value {value} does not have corresponding member")


TypeOrStr = Union[ObjectTypes, str]

object_types_registry = catalogue.create("deepdoctection", "settings", entry_points=True)


# pylint: disable=invalid-name
@object_types_registry.register("DefaultType")
class DefaultType(ObjectTypes):
    """Type for default member"""

    default_type = "DEFAULT_TYPE"


@object_types_registry.register("PageType")
class PageType(ObjectTypes):
    """Type for document page properties """
    document_type = "DOCUMENT_TYPE"  # was previously: "DOC_CLASS"
    language = "LANGUAGE"


@object_types_registry.register("SummaryType")
class SummaryType(ObjectTypes):
    """Summary type member"""
    summary = "SUMMARY"


@object_types_registry.register("DocumentType")
class DocumentType(ObjectTypes):
    """Document types"""
    letter = "LETTER"
    form = "FORM"
    email = "EMAIL"
    handwritten = "HANDWRITTEN"
    advertisement = "ADVERTISeMENT"
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
    """Layout types"""
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
    """Types for table properties"""
    item = "ITEM"
    number_of_rows = "NUMBER_OF_ROWS"
    number_of_columns = "NUMBER_OF_COLUMNS"
    max_row_span = "MAX_ROW_SPAN"
    max_col_span = "MAX_COL_SPAN"
    html = "HTML"


@object_types_registry.register("CellType")
class CellType(ObjectTypes):
    """Types for cell properties"""
    header = "HEADER"
    body = "BODY"
    row_number = "ROW_NUMBER"
    column_number = "COLUMN_NUMBER"
    row_span = "ROW_SPAN"
    column_span = "COLUMN_SPAN"


@object_types_registry.register("WordType")
class WordType(ObjectTypes):
    """Types for word properties"""
    characters = "CHARACTERS"
    block = "BLOCK"
    token_class = "TOKEN_CLASS"
    tag = "BIO_TAG"
    token_tag = "TOKEN_TAG"
    text_line = "TEXT_LINE"


@object_types_registry.register("TokenClasses")
class TokenClasses(ObjectTypes):
    """Types for token classes"""
    header = "HEADER"
    question = "QUESTION"
    answer = "ANSWER"
    other = "OTHER"


@object_types_registry.register("BioTag")
class BioTag(ObjectTypes):
    """Types for tags"""
    begin = "B"
    inside = "I"
    outside = "O"


@object_types_registry.register("TokenClassWithTag")
class TokenClassWithTag(ObjectTypes):
    """Types for token classes with tags, e.g. B-ANSWER"""
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


@object_types_registry.register("Relationships")
class Relationships(ObjectTypes):
    """Types for describing relationships between types"""
    child = "CHILD"
    reading_order = "READING_ORDER"
    semantic_entity_link = "SEMANTIC_ENTITY_LINK"


@object_types_registry.register("Languages")
class Languages(ObjectTypes):
    """Language types"""
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
    """Dataset types"""
    object_detection = "OBJECT_DETECTION"
    sequence_classification = "SEQUENCE_CLASSIFICATION"
    token_classification = "TOKEN_CLASSIFICATION"
    publaynet = "PUBLAYNET"


_TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG = {
    (TokenClasses.header, BioTag.begin): TokenClassWithTag.b_header,
    (TokenClasses.header, BioTag.inside): TokenClassWithTag.i_header,
    (TokenClasses.answer, BioTag.begin): TokenClassWithTag.b_answer,
    (TokenClasses.answer, BioTag.inside): TokenClassWithTag.i_answer,
    (TokenClasses.question, BioTag.begin): TokenClassWithTag.b_question,
    (TokenClasses.question, BioTag.inside): TokenClassWithTag.i_question,
    (TokenClasses.other, BioTag.outside): BioTag.outside,
    (TokenClasses.header, BioTag.outside): BioTag.outside,
    (TokenClasses.answer, BioTag.outside): BioTag.outside,
    (TokenClasses.question, BioTag.outside): BioTag.outside,
}


def token_class_tag_to_token_class_with_tag(token: ObjectTypes, tag: ObjectTypes) -> ObjectTypes:
    """
    Mapping TokenClassWithTag enum member from token class and tag, e.g. `TokenClasses.header` and `BioTag.inside`
    maps to TokenClassWithTag.i_header.

    :param token: TokenClasses member
    :param tag: BioTag member
    :return: TokenClassWithTag member
    """
    if isinstance(token, TokenClasses) and isinstance(tag, BioTag):
        return _TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG[(token, tag)]
    raise TypeError("Token must be of type TokenClasses and tag must be of type BioTag")


def token_class_with_tag_to_token_class_and_tag(token_class_with_tag: ObjectTypes) -> Tuple[ObjectTypes, ObjectTypes]:
    """
    This is the reverse mapping from TokenClassWithTag members to TokenClasses and BioTag

    :param token_class_with_tag: TokenClassWithTag member
    :return: Tuple of TokenClasses member and BioTag member
    """
    return {val: key for key, val in _TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG.items()}[token_class_with_tag]


_ALL_TYPES_DICT = {}
_ALL_TYPES = set(object_types_registry.get_all().values())
for ob in _ALL_TYPES:
    _ALL_TYPES_DICT.update({e.value: e for e in ob})


def update_all_types_dict() -> None:
    """Updates subsequently registered object types. Useful for defining additional ObjectTypes in tests"""
    maybe_new_types = set(object_types_registry.get_all().values())
    difference = maybe_new_types - _ALL_TYPES
    for obj in difference:
        _ALL_TYPES_DICT.update({e.value: e for e in obj})


def get_type(obj_type: Union[str, ObjectTypes]) -> ObjectTypes:
    """Get an object type property from a given string. Does nothing if an ObjectType is passed

    :param: obj_type: String or ObjectTypes
    :return: ObjectType
    """
    if isinstance(obj_type, ObjectTypes):
        return obj_type
    return_value = _ALL_TYPES_DICT.get(obj_type)
    if return_value is None:
        raise KeyError(f"String {obj_type} defines not defined registered ObjectType")
    return return_value


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
