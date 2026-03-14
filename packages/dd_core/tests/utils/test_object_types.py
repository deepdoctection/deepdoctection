# -*- coding: utf-8 -*-
# File: test_object_types.py

# Copyright 2026 Dr. Janis Meyer. All rights reserved.
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
Testing the module utils.object_types
"""


from __future__ import annotations

import uuid

import pytest

from dd_core.utils.object_types import (
    ObjectTypes,
    get_type,
    object_types_registry,
    register_custom_token_tag,
    register_string_categories_from_list,
    update_all_types_dict,
)


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _registered_enum(type_name: str) -> type[ObjectTypes] | None:
    return object_types_registry.get_all().get(type_name)


def _registered_values(type_name: str) -> set[str]:
    enum_cls = _registered_enum(type_name)
    if enum_cls is None:
        return set()
    return {str(member.value) for member in enum_cls}


def _registered_member_names(type_name: str) -> set[str]:
    enum_cls = _registered_enum(type_name)
    if enum_cls is None:
        return set()
    return {member.name for member in enum_cls}


def test_register_string_categories_from_list_registers_values_and_get_type_resolves() -> None:
    """Registers values and ensures get_type resolves them."""
    type_name = _unique_name("llm_custom_doc_class")
    value_a = _unique_name("token_a")
    value_b = _unique_name("token_b")
    value_c = _unique_name("no")

    register_string_categories_from_list([value_a, value_b, value_c], type_name)

    assert _registered_values(type_name) == {
        value_a.lower(),
        value_b.lower(),
        value_c.lower(),
    }

    member_a = get_type(value_a)
    member_b = get_type(value_b)
    member_c = get_type(value_c)

    assert member_a.value == value_a.lower()
    assert member_b.value == value_b.lower()
    assert member_c.value == value_c.lower()

    assert member_a.__class__ is _registered_enum(type_name)
    assert member_b.__class__ is _registered_enum(type_name)
    assert member_c.__class__ is _registered_enum(type_name)


def test_register_string_categories_from_list_is_idempotent_for_same_name_and_same_values() -> None:
    """Repeated registration with same values is a no-op."""
    type_name = _unique_name("llm_custom_doc_class")
    value_a = _unique_name("token_a")
    value_b = _unique_name("token_b")
    value_c = _unique_name("no")

    register_string_categories_from_list([value_a, value_b, value_c], type_name)
    register_string_categories_from_list([value_a, value_b, value_c], type_name)
    register_string_categories_from_list([value_a, value_b, value_c], type_name)

    assert _registered_values(type_name) == {
        value_a.lower(),
        value_b.lower(),
        value_c.lower(),
    }

    assert get_type(value_a).value == value_a.lower()
    assert get_type(value_b).value == value_b.lower()
    assert get_type(value_c).value == value_c.lower()


def test_register_string_categories_from_list_extends_existing_dynamic_type() -> None:
    """Existing dynamic type is extended with new values."""
    type_name = _unique_name("llm_custom_doc_class")
    value_a = _unique_name("token_a")
    value_b = _unique_name("token_b")
    value_c = _unique_name("token_c")

    register_string_categories_from_list([value_a, value_b], type_name)
    register_string_categories_from_list([value_b, value_c], type_name)

    assert _registered_values(type_name) == {
        value_a.lower(),
        value_b.lower(),
        value_c.lower(),
    }

    assert get_type(value_a).value == value_a.lower()
    assert get_type(value_b).value == value_b.lower()
    assert get_type(value_c).value == value_c.lower()


def test_register_string_categories_from_list_skips_values_already_owned_by_other_type() -> None:
    """Values owned by another dynamic type are not re-registered."""
    first_type_name = _unique_name("first_custom_type")
    second_type_name = _unique_name("second_custom_type")
    shared_value = _unique_name("shared_token")

    register_string_categories_from_list([shared_value], first_type_name)
    register_string_categories_from_list([shared_value], second_type_name)

    member = get_type(shared_value)

    assert member.value == shared_value.lower()
    assert member.__class__ is _registered_enum(first_type_name)

    second_enum = _registered_enum(second_type_name)
    if second_enum is not None:
        assert shared_value.lower() not in {str(item.value) for item in second_enum}


def test_register_string_categories_from_list_normalizes_prefixed_values() -> None:
    """BIO-prefixed values are normalized by lowercasing the suffix."""
    type_name = _unique_name("bio_custom_type")

    raw_begin = f"B-{_unique_name('CUSTOM')}"
    raw_inside = f"I-{_unique_name('CUSTOM')}"
    raw_end = f"E-{_unique_name('CUSTOM')}"

    register_string_categories_from_list([raw_begin, raw_inside, raw_end], type_name)

    expected_begin = raw_begin[:2] + raw_begin[2:].lower()
    expected_inside = raw_inside[:2] + raw_inside[2:].lower()
    expected_end = raw_end[:2] + raw_end[2:].lower()

    assert _registered_values(type_name) == {
        expected_begin,
        expected_inside,
        expected_end,
    }

    assert get_type(raw_begin).value == expected_begin
    assert get_type(raw_inside).value == expected_inside
    assert get_type(raw_end).value == expected_end

    assert get_type(expected_begin).value == expected_begin
    assert get_type(expected_inside).value == expected_inside
    assert get_type(expected_end).value == expected_end


def test_get_type_normalizes_non_prefixed_strings_to_lowercase() -> None:
    """Non-prefixed strings are normalized to lowercase."""
    type_name = _unique_name("custom_case_type")
    raw_value = _unique_name("MiXeD_Case_Value")

    register_string_categories_from_list([raw_value], type_name)

    member_upper = get_type(raw_value.upper())
    member_mixed = get_type(raw_value)
    member_lower = get_type(raw_value.lower())

    assert member_upper.value == raw_value.lower()
    assert member_mixed.value == raw_value.lower()
    assert member_lower.value == raw_value.lower()


def test_register_via_registry_decorator_updates_lookup_index() -> None:
    """Registry decorator registration updates the lookup index."""
    registry_name = _unique_name("decorator_type")
    raw_value = _unique_name("decorator_value").lower()

    namespace = {
        "ObjectTypes": ObjectTypes,
        "object_types_registry": object_types_registry,
    }

    code = f"""
@object_types_registry.register("{registry_name}")
class DecoratorRegisteredType(ObjectTypes):
    VALUE = "{raw_value}"
"""
    compiled = compile(code, "<dynamic_object_types>", "exec")
    exec(compiled, namespace)  # pylint: disable=W0122

    member = get_type(raw_value)
    registered_cls = _registered_enum(registry_name)

    assert registered_cls is not None
    assert member.value == raw_value
    assert member.__class__ is registered_cls


def test_get_type_raises_key_error_for_unregistered_value() -> None:
    """Unregistered values raise KeyError."""
    missing_value = _unique_name("definitely_not_registered")

    with pytest.raises(KeyError):
        get_type(missing_value)


def test_update_all_types_dict_rebuilds_lookup_index_without_breaking_existing_entries() -> None:
    """Rebuilding the index preserves already-registered dynamic entries."""
    type_name = _unique_name("rebuild_type")
    value_a = _unique_name("token_a")
    value_b = _unique_name("token_b")

    register_string_categories_from_list([value_a, value_b], type_name)

    update_all_types_dict()

    assert get_type(value_a).value == value_a.lower()
    assert get_type(value_b).value == value_b.lower()


def test_register_string_categories_from_list_deduplicates_values_within_single_call() -> None:
    """Values within a single call are de-duplicated stably."""
    type_name = _unique_name("dedupe_type")
    value = _unique_name("token_dup")

    register_string_categories_from_list(
        [value, value, value.upper(), value.lower()],
        type_name,
    )

    assert _registered_values(type_name) == {value.lower()}
    assert len(_registered_member_names(type_name)) == 1
    assert get_type(value).value == value.lower()


def test_register_custom_token_tag_registers_bio_combinations_and_get_type_resolves() -> None:
    """Custom token tags are registered and can be resolved via get_type."""
    custom_enum_name = _unique_name("CustomTokenType")
    suffix = _unique_name("suffix")

    CustomTokenType = ObjectTypes(  # type: ignore
        custom_enum_name,
        [
            ("TOKEN_A", _unique_name("token_a").lower()),
            ("TOKEN_B", _unique_name("token_b").lower()),
        ],
    )

    registered_name = register_custom_token_tag(CustomTokenType, suffix)

    expected_name = f"{custom_enum_name.lower()}_{suffix}"
    assert registered_name == expected_name

    registered_enum = _registered_enum(expected_name)
    assert registered_enum is not None

    expected_values = {
        f"B-{CustomTokenType.TOKEN_A.value}",
        f"I-{CustomTokenType.TOKEN_A.value}",
        f"E-{CustomTokenType.TOKEN_A.value}",
        f"B-{CustomTokenType.TOKEN_B.value}",
        f"I-{CustomTokenType.TOKEN_B.value}",
        f"E-{CustomTokenType.TOKEN_B.value}",
    }

    assert _registered_values(expected_name) == expected_values

    assert get_type(f"B-{CustomTokenType.TOKEN_A.value}").value == f"B-{CustomTokenType.TOKEN_A.value}"
    assert get_type(f"I-{CustomTokenType.TOKEN_A.value}").value == f"I-{CustomTokenType.TOKEN_A.value}"
    assert get_type(f"E-{CustomTokenType.TOKEN_A.value}").value == f"E-{CustomTokenType.TOKEN_A.value}"

    assert get_type(f"B-{CustomTokenType.TOKEN_B.value}").value == f"B-{CustomTokenType.TOKEN_B.value}"
    assert get_type(f"I-{CustomTokenType.TOKEN_B.value}").value == f"I-{CustomTokenType.TOKEN_B.value}"
    assert get_type(f"E-{CustomTokenType.TOKEN_B.value}").value == f"E-{CustomTokenType.TOKEN_B.value}"


def test_register_custom_token_tag_is_idempotent_for_same_custom_enum_and_suffix() -> None:
    """Registering the same custom enum+suffix repeatedly is idempotent."""
    custom_enum_name = _unique_name("CustomTokenType")
    suffix = _unique_name("suffix")

    CustomTokenType = ObjectTypes(  # type: ignore
        custom_enum_name,
        [
            ("TOKEN_A", _unique_name("token_a").lower()),
            ("TOKEN_B", _unique_name("token_b").lower()),
        ],
    )

    registered_name_1 = register_custom_token_tag(CustomTokenType, suffix)
    registered_name_2 = register_custom_token_tag(CustomTokenType, suffix)
    registered_name_3 = register_custom_token_tag(CustomTokenType, suffix)

    expected_name = f"{custom_enum_name.lower()}_{suffix}"
    assert registered_name_1 == expected_name
    assert registered_name_2 == expected_name
    assert registered_name_3 == expected_name

    assert _registered_values(expected_name) == {
        f"B-{CustomTokenType.TOKEN_A.value}",
        f"I-{CustomTokenType.TOKEN_A.value}",
        f"E-{CustomTokenType.TOKEN_A.value}",
        f"B-{CustomTokenType.TOKEN_B.value}",
        f"I-{CustomTokenType.TOKEN_B.value}",
        f"E-{CustomTokenType.TOKEN_B.value}",
    }


def test_register_custom_token_tag_extends_existing_registered_tag_type() -> None:
    """A later registration for same name extends the existing enum."""
    custom_enum_name = _unique_name("CustomTokenType")
    suffix = _unique_name("suffix")

    CustomTokenTypeV1 = ObjectTypes(  # type: ignore
        custom_enum_name,
        [
            ("TOKEN_A", _unique_name("token_a").lower()),
        ],
    )

    CustomTokenTypeV2 = ObjectTypes(  # type: ignore
        custom_enum_name,
        [
            ("TOKEN_A", CustomTokenTypeV1.TOKEN_A.value),
            ("TOKEN_B", _unique_name("token_b").lower()),
        ],
    )

    registered_name = register_custom_token_tag(CustomTokenTypeV1, suffix)
    register_custom_token_tag(CustomTokenTypeV2, suffix)

    expected_name = f"{custom_enum_name.lower()}_{suffix}"
    assert registered_name == expected_name

    assert _registered_values(expected_name) == {
        f"B-{CustomTokenTypeV1.TOKEN_A.value}",
        f"I-{CustomTokenTypeV1.TOKEN_A.value}",
        f"E-{CustomTokenTypeV1.TOKEN_A.value}",
        f"B-{CustomTokenTypeV2.TOKEN_B.value}",
        f"I-{CustomTokenTypeV2.TOKEN_B.value}",
        f"E-{CustomTokenTypeV2.TOKEN_B.value}",
    }

    assert get_type(f"B-{CustomTokenTypeV2.TOKEN_B.value}").value == f"B-{CustomTokenTypeV2.TOKEN_B.value}"
    assert get_type(f"I-{CustomTokenTypeV2.TOKEN_B.value}").value == f"I-{CustomTokenTypeV2.TOKEN_B.value}"
    assert get_type(f"E-{CustomTokenTypeV2.TOKEN_B.value}").value == f"E-{CustomTokenTypeV2.TOKEN_B.value}"
