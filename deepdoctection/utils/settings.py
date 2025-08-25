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
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import catalogue  # type: ignore


class ObjectTypes(str, Enum):
    """Base Class for describing objects as attributes of Enums"""

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}>"

    @classmethod
    def from_value(cls, value: str) -> ObjectTypes:
        """Getting the enum member from a given string value

        :param value: string value to get the enum member
        :return: Enum member
        """
        for member in cls.__members__.values():
            if member.value == value:
                return member
        raise ValueError(f"value {value} does not have corresponding member")


TypeOrStr = Union[ObjectTypes, str]  # pylint: disable=C0103

object_types_registry = catalogue.create("deepdoctection", "settings", entry_points=True)


# pylint: disable=invalid-name
@object_types_registry.register("DefaultType")
class DefaultType(ObjectTypes):
    """Type for default member"""

    DEFAULT_TYPE = "default_type"


@object_types_registry.register("PageType")
class PageType(ObjectTypes):
    """Type for document page properties"""

    DOCUMENT_TYPE = "document_type"
    LANGUAGE = "language"
    ANGLE = "angle"
    SIZE = "size"
    PAD_TOP = "pad_top"
    PAD_BOTTOM = "pad_bottom"
    PAD_LEFT = "pad_left"
    PAD_RIGHT = "pad_right"


@object_types_registry.register("SummaryType")
class SummaryType(ObjectTypes):
    """Summary type member"""

    SUMMARY = "summary"
    DOCUMENT_SUMMARY = "document_summary"
    DOCUMENT_MAPPING = "document_mapping"


@object_types_registry.register("DocumentType")
class DocumentType(ObjectTypes):
    """Document types"""

    LETTER = "letter"
    FORM = "form"
    EMAIL = "email"
    HANDWRITTEN = "handwritten"
    ADVERTISEMENT = "advertisement"
    SCIENTIFIC_REPORT = "scientific_report"
    SCIENTIFIC_PUBLICATION = "scientific_publication"
    SPECIFICATION = "specification"
    FILE_FOLDER = "file_folder"
    NEWS_ARTICLE = "news_article"
    BUDGET = "budget"
    INVOICE = "invoice"
    PRESENTATION = "presentation"
    QUESTIONNAIRE = "questionnaire"
    RESUME = "resume"
    MEMO = "memo"
    FINANCIAL_REPORT = "financial_report"
    LAWS_AND_REGULATIONS = "laws_and_regulations"
    GOVERNMENT_TENDERS = "government_tenders"
    MANUALS = "manuals"
    PATENTS = "patents"
    BANK_STATEMENT = "bank_statement"


@object_types_registry.register("LayoutType")
class LayoutType(ObjectTypes):
    """Layout types"""

    TABLE = "table"
    TABLE_ROTATED = "table_rotated"
    FIGURE = "figure"
    LIST = "list"
    TEXT = "text"
    TITLE = "title"
    LOGO = "logo"
    SIGNATURE = "signature"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    FORMULA = "formula"
    PAGE_FOOTER = "page_footer"
    PAGE_HEADER = "page_header"
    SECTION_HEADER = "section_header"
    PAGE = "page"
    CELL = "cell"
    ROW = "row"
    COLUMN = "column"
    WORD = "word"
    LINE = "line"
    BACKGROUND = "background"
    PAGE_NUMBER = "page_number"
    KEY_VALUE_AREA = "key_value_area"
    LIST_ITEM = "list_item"
    MARK = "mark"


@object_types_registry.register("TableType")
class TableType(ObjectTypes):
    """Types for table properties"""

    ITEM = "item"
    NUMBER_OF_ROWS = "number_of_rows"
    NUMBER_OF_COLUMNS = "number_of_columns"
    MAX_ROW_SPAN = "max_row_span"
    MAX_COL_SPAN = "max_col_span"
    HTML = "html"


@object_types_registry.register("CellType")
class CellType(ObjectTypes):
    """Types for cell properties"""

    HEADER = "header"
    BODY = "body"
    ROW_NUMBER = "row_number"
    ROW_SPAN = "row_span"
    ROW_HEADER = "row_header"
    PROJECTED_ROW_HEADER = "projected_row_header"
    COLUMN_NUMBER = "column_number"
    COLUMN_SPAN = "column_span"
    COLUMN_HEADER = "column_header"
    SPANNING = "spanning"


@object_types_registry.register("WordType")
class WordType(ObjectTypes):
    """Types for word properties"""

    CHARACTERS = "characters"
    BLOCK = "block"
    TOKEN_CLASS = "token_class"
    TAG = "tag"
    TOKEN_TAG = "token_tag"
    TEXT_LINE = "text_line"
    CHARACTER_TYPE = "character_type"
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"


@object_types_registry.register("TokenClasses")
class TokenClasses(ObjectTypes):
    """Types for token classes"""

    HEADER = "header"
    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"


@object_types_registry.register("BioTag")
class BioTag(ObjectTypes):
    """Types for tags"""

    BEGIN = "B"
    INSIDE = "I"
    OUTSIDE = "O"
    SINGLE = "S"
    END = "E"


@object_types_registry.register("TokenClassWithTag")
class TokenClassWithTag(ObjectTypes):
    """Types for token classes with tags, e.g. B-answer"""

    B_ANSWER = "B-answer"
    B_HEADER = "B-header"
    B_QUESTION = "B-question"
    E_ANSWER = "E-answer"
    E_HEADER = "E-header"
    E_QUESTION = "E-question"
    I_ANSWER = "I-answer"
    I_HEADER = "I-header"
    I_QUESTION = "I-question"
    S_ANSWER = "S-answer"
    S_HEADER = "S-header"
    S_QUESTION = "S-question"


@object_types_registry.register("Relationships")
class Relationships(ObjectTypes):
    """Types for describing relationships between types"""

    CHILD = "child"
    READING_ORDER = "reading_order"
    LINK = "link"
    LAYOUT_LINK = "layout_link"
    SUCCESSOR = "successor"


@object_types_registry.register("Languages")
class Languages(ObjectTypes):
    """Language types"""

    ENGLISH = "eng"
    RUSSIAN = "rus"
    GERMAN = "deu"
    FRENCH = "fre"
    ITALIAN = "ita"
    JAPANESE = "jpn"
    SPANISH = "spa"
    CEBUANO = "ceb"
    TURKISH = "tur"
    PORTUGUESE = "por"
    UKRAINIAN = "ukr"
    ESPERANTO = "epo"
    POLISH = "pol"
    SWEDISH = "swe"
    DUTCH = "dut"
    HEBREW = "heb"
    CHINESE = "chi"
    HUNGARIAN = "hun"
    ARABIC = "ara"
    CATALAN = "cat"
    FINNISH = "fin"
    CZECH = "cze"
    PERSIAN = "per"
    SERBIAN = "srp"
    GREEK = "gre"
    VIETNAMESE = "vie"
    BULGARIAN = "bul"
    KOREAN = "kor"
    NORWEGIAN = "nor"
    MACEDONIAN = "mac"
    ROMANIAN = "rum"
    INDONESIAN = "ind"
    THAI = "tha"
    ARMENIAN = "arm"
    DANISH = "dan"
    TAMIL = "tam"
    HINDI = "hin"
    CROATIAN = "hrv"
    BELARUSIAN = "bel"
    GEORGIAN = "geo"
    TELUGU = "tel"
    KAZAKH = "kaz"
    WARAY = "war"
    LITHUANIAN = "lit"
    SCOTTISH = "glg"
    SLOVAK = "slo"
    BENIN = "ben"
    BASQUE = "baq"
    SLOVENIAN = "slv"
    MALAYALAM = "mal"
    MARATHI = "mar"
    ESTONIAN = "est"
    AZERBAIJANI = "aze"
    ALBANIAN = "alb"
    LATIN = "lat"
    BOSNIAN = "bos"
    NORWEGIAN_NOVOSIBIRSK = "nno"
    URDU = "urd"
    SWAHILI = "swa"
    NOT_DEFINED = "nn"


@object_types_registry.register("DatasetType")
class DatasetType(ObjectTypes):
    """Dataset types"""

    OBJECT_DETECTION = "object_detection"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    PUBLAYNET = "publaynet"
    DEFAULT = "default"


_TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG = {
    (TokenClasses.HEADER, BioTag.BEGIN): TokenClassWithTag.B_HEADER,
    (TokenClasses.HEADER, BioTag.INSIDE): TokenClassWithTag.I_HEADER,
    (TokenClasses.HEADER, BioTag.END): TokenClassWithTag.E_HEADER,
    (TokenClasses.HEADER, BioTag.SINGLE): TokenClassWithTag.S_HEADER,
    (TokenClasses.ANSWER, BioTag.BEGIN): TokenClassWithTag.B_ANSWER,
    (TokenClasses.ANSWER, BioTag.INSIDE): TokenClassWithTag.I_ANSWER,
    (TokenClasses.ANSWER, BioTag.END): TokenClassWithTag.E_ANSWER,
    (TokenClasses.ANSWER, BioTag.SINGLE): TokenClassWithTag.S_ANSWER,
    (TokenClasses.QUESTION, BioTag.BEGIN): TokenClassWithTag.B_QUESTION,
    (TokenClasses.QUESTION, BioTag.INSIDE): TokenClassWithTag.I_QUESTION,
    (TokenClasses.QUESTION, BioTag.END): TokenClassWithTag.E_QUESTION,
    (TokenClasses.QUESTION, BioTag.SINGLE): TokenClassWithTag.S_QUESTION,
    (TokenClasses.OTHER, BioTag.OUTSIDE): BioTag.OUTSIDE,
    (TokenClasses.HEADER, BioTag.OUTSIDE): BioTag.OUTSIDE,
    (TokenClasses.ANSWER, BioTag.OUTSIDE): BioTag.OUTSIDE,
    (TokenClasses.QUESTION, BioTag.OUTSIDE): BioTag.OUTSIDE,
}


def token_class_tag_to_token_class_with_tag(token: ObjectTypes, tag: ObjectTypes) -> ObjectTypes:
    """
    Maps a `TokenClassWithTag` enum member from a token class and tag, e.g. `TokenClasses.header` and `BioTag.inside`
    maps to `TokenClassWithTag.i_header`.

    Args:
        token: TokenClasses member.
        tag: BioTag member.

    Returns:
        TokenClassWithTag member.

    Raises:
        TypeError: If token is not of type TokenClasses or tag is not of type BioTag.
    """
    if isinstance(token, TokenClasses) and isinstance(tag, BioTag):
        return _TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG[(token, tag)]
    raise TypeError(
        f"Token must be of type TokenClasses, is of {type(token)} and tag " f"{type(tag)} must be of type BioTag"
    )


def token_class_with_tag_to_token_class_and_tag(
    token_class_with_tag: ObjectTypes,
) -> Optional[tuple[ObjectTypes, ObjectTypes]]:
    """
    This is the reverse mapping from TokenClassWithTag members to TokenClasses and BioTag

    Args:
        token_class_with_tag: `TokenClassWithTag` member

    Returns:
        Tuple of `TokenClasses` member and `BioTag` member
    """
    return {val: key for key, val in _TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG.items()}.get(token_class_with_tag)


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


_OLD_TO_NEW_OBJ_TYPE: dict[str, str] = {
    "DOC_CLASS": "document_type",
    "CHARS": "characters",
    "BIO_TAG": "tag",
    "B-ANSWER": "B-answer",
    "B-HEADER": "B-header",
    "B-QUESTION": "B-question",
    "E-ANSWER": "E-answer",
    "E-HEADER": "E-header",
    "E-QUESTION": "E-question",
    "I-ANSWER": "I-answer",
    "I-HEADER": "I-header",
    "I-QUESTION": "I-question",
    "S-ANSWER": "S-answer",
    "S-HEADER": "S-header",
    "S-QUESTION": "S-question",
}


def _get_new_obj_type_str(obj_type: str) -> str:
    return _OLD_TO_NEW_OBJ_TYPE.get(obj_type, obj_type)


_BLACK_LIST: list[str] = ["B", "I", "O", "E", "S"]


def _get_black_list() -> list[str]:
    return _BLACK_LIST


def update_black_list(item: str) -> None:
    """Updates the black list, i.e. set of elements that must not be lowered"""
    _BLACK_LIST.append(item)


def get_type(obj_type: Union[str, ObjectTypes]) -> ObjectTypes:
    """
    Get an object type property from a given string. Does nothing if an `ObjectType` is passed

    Args:
        obj_type: String or ObjectTypes
    Returns:
        `ObjectType`
    """
    if isinstance(obj_type, ObjectTypes):
        return obj_type
    obj_type = _get_new_obj_type_str(obj_type)
    if obj_type.startswith(("B-", "E-", "I-", "S-")):
        obj_type = obj_type[:2] + obj_type[2:].lower()
    elif obj_type not in _get_black_list():
        obj_type = obj_type.lower()
    return_value = _ALL_TYPES_DICT.get(obj_type)
    if return_value is None:
        raise KeyError(f"String {obj_type} does not correspond to a registered ObjectType")
    return return_value


# Some path settings

# package path
file_path = Path(os.path.split(__file__)[0])
PATH = file_path.parent.parent

# model cache directory
if os.environ.get("DEEPDOCTECTION_CACHE"):
    dd_cache_home = Path(os.environ["DEEPDOCTECTION_CACHE"])
else:
    dd_cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "deepdoctection"

CACHE_DIR = dd_cache_home
MODEL_DIR = dd_cache_home / "weights"

# configs cache directory
CONFIGS = dd_cache_home / "configs"

# dataset cache directory
DATASET_DIR = dd_cache_home / "datasets"

FILE_PATH = os.path.split(__file__)[0]
TPATH = os.path.dirname(os.path.dirname(FILE_PATH))
