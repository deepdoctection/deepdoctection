# -*- coding: utf-8 -*-
# File: object_types.py

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

import itertools
import re
import threading
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Sequence, Type, Union

import catalogue  # type: ignore

from .error import DuplicateObjectTypeError


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
_orig_register = object_types_registry.register


def _iter_registered_enums() -> Iterable[Type[ObjectTypes]]:
    """
    Iterate over all enum classes registered in the catalogue registry.
    """
    return object_types_registry.get_all().values()


def _wrapped_register(name: str, func: Optional[Any] = None) -> Callable[[Type[ObjectTypes]], Type[ObjectTypes]]:
    def _decorator(cls: Type[ObjectTypes]) -> Type[ObjectTypes]:
        with _TYPES_INDEX_LOCK:
            registered_cls = _orig_register(name, func=func)(cls)
            _rebuild_types_index_locked()
            return registered_cls

    return _decorator


def _upsert_dynamic_enum(name: str, members: Sequence[tuple[str, str]]) -> Type[ObjectTypes] | None:
    """
    Idempotently register or extend a dynamic ObjectTypes enum under `name`.

    Rules:
    - existing values under the same enum name are preserved
    - new values are appended
    - repeated registration of the same values is a no-op
    - values already owned by a different enum raise DuplicateObjectTypeError
    """
    with _TYPES_INDEX_LOCK:
        registered = object_types_registry.get_all()
        existing_enum = registered.get(name)

        existing_by_value: dict[str, str] = {}
        if existing_enum is not None:
            for member in existing_enum:
                existing_by_value[str(member.value)] = member.name

        merged_by_value = dict(existing_by_value)

        for proposed_member_name, raw_value in members:
            value = _normalize_object_type_value(raw_value)

            existing_member = _ALL_TYPES_DICT.get(value)

            if existing_member is not None and existing_member.__class__.__name__ != name:
                continue

            if value not in merged_by_value:
                merged_by_value[value] = proposed_member_name

        if merged_by_value == existing_by_value:
            return existing_enum

        if not merged_by_value:
            return None

        merged_members = [(member_name, value) for value, member_name in merged_by_value.items()]
        merged_enum = ObjectTypes(name, merged_members)  # type: ignore

        registered_cls = _orig_register(name)(merged_enum)
        _rebuild_types_index_locked()
        return registered_cls


def _get_black_list() -> list[str]:
    return _BLACK_LIST


def update_black_list(item: str) -> None:
    """Updates the black list, i.e. set of elements that must not be lowered"""
    _BLACK_LIST.append(item)


def _normalize_object_type_value(obj_type: str) -> str:
    """
    Canonical normalization for lookup and dynamic registration.
    This must match get_type() semantics.
    """
    obj_type = _get_new_obj_type_str(obj_type)

    if obj_type.startswith(("B-", "E-", "I-", "S-")):
        return obj_type[:2] + obj_type[2:].lower()

    if obj_type not in _get_black_list():
        return obj_type.lower()

    return obj_type


def _flatten_categories(categories_list: Sequence[str]) -> list[str]:
    """
    Flatten a possibly nested category sequence into a plain list[str].
    """
    if categories_list and isinstance(categories_list[0], (list, tuple)):
        return [str(item) for item in itertools.chain.from_iterable(categories_list)]
    return [str(item) for item in categories_list]


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    """
    Stable de-duplication preserving the first occurrence order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _sanitize_enum_member_name(value: str, used_names: set[str]) -> str:
    """
    Convert a value string into a safe enum member name.
    """
    candidate = re.sub(r"[^A-Z0-9_]", "_", value.upper())
    if not candidate:
        candidate = "TYPE"
    if candidate[0].isdigit():
        candidate = f"TYPE_{candidate}"

    name = candidate
    suffix = 2
    while name in used_names:
        name = f"{candidate}_{suffix}"
        suffix += 1

    used_names.add(name)
    return name


def _build_types_index_from_registry() -> dict[str, ObjectTypes]:
    """
    Build a fresh value->member index from the current registry contents.

    This is the only place where duplicate value detection happens.
    """
    new_index: dict[str, ObjectTypes] = {}

    for enum_cls in _iter_registered_enums():
        for member in enum_cls:
            value = str(member.value)
            existing = new_index.get(value)
            if existing is not None and existing is not member:
                raise DuplicateObjectTypeError(
                    f"Object type value '{value}' already taken by {existing!r}; cannot register {member!r}"
                )
            new_index[value] = member

    return new_index


def _rebuild_types_index_locked() -> None:
    """
    Rebuild the global lookup index from the registry.
    Caller must hold _TYPES_INDEX_LOCK.
    """
    global _ALL_TYPES_DICT  # pylint: disable=W0603
    _ALL_TYPES_DICT = _build_types_index_from_registry()


def _rebuild_types_index() -> None:
    """
    Public/internal helper to rebuild the lookup index from the registry.
    """
    with _TYPES_INDEX_LOCK:
        _rebuild_types_index_locked()


def token_class_tag_to_token_class_with_tag(token: ObjectTypes, tag: ObjectTypes) -> ObjectTypes:
    """
    Maps a `TokenClassWithTagLabel` enum member from a token class and tag, e.g. `TokenClassLabel.HEADER` and
    `BioTag.INSIDE` maps to `TTokenClassWithTagLabel.I_HEADER`.

    Args:
        token: TokenClasses member.
        tag: BioTag member.

    Returns:
        TokenClassWithTag member.

    Raises:
        TypeError: If token is not of type TokenClasses or tag is not of type BioTag.
    """
    if isinstance(token, TokenClassLabel) and isinstance(tag, BioTagLabel):
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


def register_custom_token_tag(custom_object_types: ObjectTypes, suffix: str) -> str:
    """
    Registers custom token tags for a given ObjectType with a specified suffix. The tags are created by combining
    BIO tags (B, I, E) with the custom object types.

    Args:
        custom_object_types: An instance of ObjectTypes containing the custom object types to be registered.
        suffix: A string suffix to be appended to the name of the registered object type.

    Returns:
        The name of the registered object type.

    Example:

    ```python
    from deepdoctection.utils.settings import ObjectTypes

    class CustomObjectTypesLabel(ObjectTypes):
        TOKEN_A = "token_a"
        TOKEN_B = "token_b"

    custom_object_types = CustomObjectTypes()
    register_custom_token_tag(custom_object_types, "custom_type")
    # This will register tags like "B-TOKEN_A", "I-TOKEN_A", "E-TOKEN_A", "B-TOKEN_B", "I-TOKEN_B", "E-TOKEN_B"
    ```

    """
    tag_list = [i for i in object_types_registry.get("BioTagLabel") if i in ("B", "I", "E")]
    name = f"{custom_object_types.__name__.lower()}_{suffix}"  # type: ignore

    product = [
        (
            a[0].value + "_" + a[1].value.upper(),  # type: ignore
            a[0].value + "-" + a[1].value,  # type: ignore
        )
        for a in list(itertools.product(tag_list, custom_object_types))
    ]

    _upsert_dynamic_enum(name, product)
    return name


def register_string_categories_from_list(categories_list: Sequence[str], object_type_name: str) -> None:
    """
    Idempotently register or extend string categories under a dynamic ObjectTypes enum.

    Repeated calls with the same values are a no-op.
    Repeated calls with new values extend the enum under the same name.
    """
    flattened_categories = _flatten_categories(categories_list)
    normalized_values = _dedupe_preserve_order([_normalize_object_type_value(cat) for cat in flattened_categories])

    if not normalized_values:
        return

    used_names: set[str] = set()
    members = [(_sanitize_enum_member_name(value, used_names), value) for value in normalized_values]

    _upsert_dynamic_enum(object_type_name, members)


def update_all_types_dict() -> None:
    """
    Compatibility helper retained for older code/tests that call it explicitly.
    Rebuilds the global index from the registry.
    """
    _rebuild_types_index()


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

    if not isinstance(obj_type, str):
        raise TypeError(f"get_type expects str or ObjectTypes, got {type(obj_type)}")

    normalized = _normalize_object_type_value(obj_type)

    with _TYPES_INDEX_LOCK:
        member = _ALL_TYPES_DICT.get(normalized)

    if member is None:
        raise KeyError(f"String {normalized} does not correspond to a registered ObjectType")

    return member


def _get_new_obj_type_str(obj_type: str) -> str:
    return _OLD_TO_NEW_OBJ_TYPE.get(obj_type, obj_type)


# Monkey-patch the registry to enforce duplicate detection for all modules.
object_types_registry.register = _wrapped_register

_TYPES_INDEX_LOCK = threading.RLock()
_ALL_TYPES_DICT: dict[str, ObjectTypes] = {}


_rebuild_types_index()


@object_types_registry.register("DefaultType")
class DefaultType(ObjectTypes):
    """Type for default member"""

    DEFAULT_TYPE = "default_type"


@object_types_registry.register("DocumentFileType")
class DocumentFileLabel(ObjectTypes):
    """Supported document types."""

    PDF = "pdf"
    VAR = "various"
    PNG = "png"
    JPEG = "jpeg"
    JPG = "jpg"
    TIFF = "tiff"


@object_types_registry.register("PageKey")
class PageKey(ObjectTypes):
    """Type for document page properties"""

    DOCUMENT_TYPE = "document_type"
    LANGUAGE = "language"
    ANGLE = "angle"
    SIZE = "size"


@object_types_registry.register("ImagePadKey")
class ImagePadKey(ObjectTypes):
    """Types for image padding values"""

    PAD_TOP = "top"
    PAD_BOTTOM = "bottom"
    PAD_LEFT = "left"
    PAD_RIGHT = "right"


@object_types_registry.register("SummaryKey")
class SummaryKey(ObjectTypes):
    """Summary type member"""

    SUMMARY = "summary"
    DOCUMENT_SUMMARY = "document_summary"
    DOCUMENT_MAPPING = "document_mapping"
    KEY_VALUES = "key_values"
    SPLIT_NEXT = "split_next"


@object_types_registry.register("RelationshipKey")
class RelationshipKey(ObjectTypes):
    """Relationship keys between annotations."""

    CHILD = "child"
    READING_ORDER = "reading_order"
    LINK = "link"
    LAYOUT_LINK = "layout_link"
    SUCCESSOR = "successor"


@object_types_registry.register("DocumentLabel")
class DocumentLabel(ObjectTypes):
    """Document types"""

    LETTER = "letter"
    FORM = "form"
    EMAIL = "email"
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


@object_types_registry.register("LayoutLabel")
class LayoutLabel(ObjectTypes):
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


@object_types_registry.register("TableKey")
class TableKey(ObjectTypes):
    """Types for table properties"""

    ITEM = "item"
    NUMBER_OF_ROWS = "number_of_rows"
    NUMBER_OF_COLUMNS = "number_of_columns"
    MAX_ROW_SPAN = "max_row_span"
    MAX_COL_SPAN = "max_col_span"
    HTML = "html"


@object_types_registry.register("CellLabel")
class CellLabel(ObjectTypes):
    """Cell semantic labels / roles."""

    BODY = "body"
    ROW_HEADER = "row_header"
    PROJECTED_ROW_HEADER = "projected_row_header"
    COLUMN_HEADER = "column_header"
    SPANNING = "spanning"


@object_types_registry.register("CellKey")
class CellKey(ObjectTypes):
    """Keys for cell properties."""

    ROW_NUMBER = "row_number"
    ROW_SPAN = "row_span"
    COLUMN_NUMBER = "column_number"
    COLUMN_SPAN = "column_span"


@object_types_registry.register("WordKey")
class WordKey(ObjectTypes):
    """Types for word properties"""

    CHARACTERS = "characters"
    BLOCK = "block"
    TOKEN_CLASS = "token_class"
    TAG = "tag"
    TOKEN_TAG = "token_tag"
    TEXT_LINE = "text_line"
    CHARACTER_TYPE = "character_type"


@object_types_registry.register("CharacterTypeLabel")
class CharacterTypeLabel(ObjectTypes):
    """Character type labels used under WordKey.CHARACTER_TYPE."""

    PRINTED = "printed"
    HANDWRITTEN = "handwritten"


@object_types_registry.register("TokenClassLabel")
class TokenClassLabel(ObjectTypes):
    """Types for token classes"""

    HEADER = "header"
    QUESTION = "question"
    ANSWER = "answer"
    OTHER = "other"


@object_types_registry.register("BioTagLabel")
class BioTagLabel(ObjectTypes):
    """Types for tags"""

    BEGIN = "B"
    INSIDE = "I"
    OUTSIDE = "O"
    SINGLE = "S"
    END = "E"


@object_types_registry.register("TokenClassWithTagLabel")
class TokenClassWithTagLabel(ObjectTypes):
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


@object_types_registry.register("LanguageCode")
class LanguageCode(ObjectTypes):
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


@object_types_registry.register("DatasetKind")
class DatasetKind(ObjectTypes):
    """Dataset types"""

    OBJECT_DETECTION = "object_detection"
    SEQUENCE_CLASSIFICATION = "sequence_classification"
    TOKEN_CLASSIFICATION = "token_classification"
    PUBLAYNET = "publaynet"
    DEFAULT = "default"


_TOKEN_AND_TAG_TO_TOKEN_CLASS_WITH_TAG = {
    (TokenClassLabel.HEADER, BioTagLabel.BEGIN): TokenClassWithTagLabel.B_HEADER,
    (TokenClassLabel.HEADER, BioTagLabel.INSIDE): TokenClassWithTagLabel.I_HEADER,
    (TokenClassLabel.HEADER, BioTagLabel.END): TokenClassWithTagLabel.E_HEADER,
    (TokenClassLabel.HEADER, BioTagLabel.SINGLE): TokenClassWithTagLabel.S_HEADER,
    (TokenClassLabel.ANSWER, BioTagLabel.BEGIN): TokenClassWithTagLabel.B_ANSWER,
    (TokenClassLabel.ANSWER, BioTagLabel.INSIDE): TokenClassWithTagLabel.I_ANSWER,
    (TokenClassLabel.ANSWER, BioTagLabel.END): TokenClassWithTagLabel.E_ANSWER,
    (TokenClassLabel.ANSWER, BioTagLabel.SINGLE): TokenClassWithTagLabel.S_ANSWER,
    (TokenClassLabel.QUESTION, BioTagLabel.BEGIN): TokenClassWithTagLabel.B_QUESTION,
    (TokenClassLabel.QUESTION, BioTagLabel.INSIDE): TokenClassWithTagLabel.I_QUESTION,
    (TokenClassLabel.QUESTION, BioTagLabel.END): TokenClassWithTagLabel.E_QUESTION,
    (TokenClassLabel.QUESTION, BioTagLabel.SINGLE): TokenClassWithTagLabel.S_QUESTION,
    (TokenClassLabel.OTHER, BioTagLabel.OUTSIDE): BioTagLabel.OUTSIDE,
    (TokenClassLabel.HEADER, BioTagLabel.OUTSIDE): BioTagLabel.OUTSIDE,
    (TokenClassLabel.ANSWER, BioTagLabel.OUTSIDE): BioTagLabel.OUTSIDE,
    (TokenClassLabel.QUESTION, BioTagLabel.OUTSIDE): BioTagLabel.OUTSIDE,
}


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


_BLACK_LIST: list[str] = ["B", "I", "O", "E", "S"]
