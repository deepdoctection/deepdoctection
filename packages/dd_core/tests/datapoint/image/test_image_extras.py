# -*- coding: utf-8 -*-
# File: test_image_extras.py

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
Testing Extras and Image extras serialization / deserialization
"""

import json
from pathlib import Path

import pytest

from dd_core.datapoint.image import Extras, Image

from ..conftest import WhiteImage


class TestExtras:
    """Unit tests for the Extras class."""

    def test_set_type_str_and_dump(self) -> None:
        """str key: dump replaces the value."""
        e = Extras()
        e.set_type("status", "str")
        e.dump("status", "pending")
        e.dump("status", "done")
        assert e._data["status"] == "done"

    def test_set_type_list_and_dump_appends(self) -> None:
        """list[str] key: dump appends to the list."""
        e = Extras()
        e.set_type("ids", "list[str]")
        e.dump("ids", "a")
        e.dump("ids", "b")
        assert e._data["ids"] == ["a", "b"]

    def test_list_key_initialised_as_empty_list(self) -> None:
        """list[str] key is initialised to [] on set_type."""
        e = Extras()
        e.set_type("ids", "list[str]")
        assert e._data["ids"] == []

    def test_set_type_conflict_raises(self) -> None:
        """Re-registering a key with a different type raises ValueError."""
        e = Extras()
        e.set_type("x", "str")
        with pytest.raises(ValueError):
            e.set_type("x", "list[str]")

    def test_dump_unconfigured_key_raises(self) -> None:
        """Dumping to an unconfigured key raises KeyError."""
        e = Extras()
        with pytest.raises(KeyError):
            e.dump("missing", "value")

    def test_dump_non_str_value_raises(self) -> None:
        """Passing a non-str value raises TypeError."""
        e = Extras()
        e.set_type("x", "str")
        with pytest.raises(TypeError):
            e.dump("x", 42)  # type: ignore[arg-type]

    def test_set_type_persist_conflict_raises(self) -> None:
        """Re-registering a key with a different persist flag raises ValueError."""
        e = Extras()
        e.set_type("x", "str", persist=True)
        with pytest.raises(ValueError):
            e.set_type("x", "str", persist=False)

    def test_set_type_rejects_underscore_prefixed_name(self) -> None:
        """Registering a key starting with '_' raises ValueError."""
        e = Extras()
        with pytest.raises(ValueError):
            e.set_type("_secret", "str")


class TestExtrasAttributeAccess:
    """Tests for attribute-style read access on Extras."""

    def test_str_key_readable_as_attribute(self) -> None:
        """A dumped str value is readable via attribute access."""
        e = Extras()
        e.set_type("tag", "str")
        e.dump("tag", "hello")
        assert e.tag == "hello"

    def test_list_key_readable_as_attribute(self) -> None:
        """A dumped list[str] value is readable via attribute access."""
        e = Extras()
        e.set_type("ids", "list[str]")
        e.dump("ids", "a")
        e.dump("ids", "b")
        assert e.ids == ["a", "b"]

    def test_registered_but_undumped_str_key_is_none(self) -> None:
        """A registered but never-dumped str key reads as None via attribute access."""
        e = Extras()
        e.set_type("tag", "str")
        assert e.tag is None

    def test_unregistered_attribute_raises_attribute_error(self) -> None:
        """An unregistered name raises AttributeError, not silently returning None."""
        e = Extras()
        with pytest.raises(AttributeError):
            _ = e.nope

    def test_attribute_access_works_regardless_of_persist(self) -> None:
        """Attribute read access works for both persisted and non-persisted keys."""
        e = Extras()
        e.set_type("persisted", "str", persist=True)
        e.dump("persisted", "kept")
        e.set_type("transient", "str", persist=False)
        e.dump("transient", "gone")
        assert e.persisted == "kept"
        assert e.transient == "gone"


class TestExtrasSerialization:
    """Round-trip tests for Extras.as_dict / from_dict."""

    def test_as_dict_shape(self) -> None:
        """as_dict returns dict with _schema and _data keys."""
        e = Extras()
        e.set_type("tag", "str", persist=True)
        e.dump("tag", "hello")
        d = e.as_dict()
        assert "_schema" in d
        assert "_data" in d

    def test_as_dict_default_only_includes_persisted_keys(self) -> None:
        """as_dict() (default) filters out keys not registered with persist=True."""
        e = Extras()
        e.set_type("kept", "str", persist=True)
        e.dump("kept", "hello")
        e.set_type("dropped", "list[str]", persist=False)
        e.dump("dropped", "x")
        d = e.as_dict()
        assert d["_schema"] == {"kept": "str"}
        assert d["_data"] == {"kept": "hello"}

    def test_as_dict_add_extras_includes_everything(self) -> None:
        """as_dict(add_extras=True) returns all keys regardless of persist flag."""
        e = Extras()
        e.set_type("kept", "str", persist=True)
        e.dump("kept", "hello")
        e.set_type("dropped", "list[str]", persist=False)
        e.dump("dropped", "x")
        d = e.as_dict(add_extras=True)
        assert d["_schema"] == {"kept": "str", "dropped": "list[str]"}
        assert d["_data"] == {"kept": "hello", "dropped": ["x"]}

    def test_roundtrip_str_key(self) -> None:
        """Persisted str key survives as_dict / from_dict."""
        e = Extras()
        e.set_type("tag", "str", persist=True)
        e.dump("tag", "hello")
        restored = Extras.from_dict(e.as_dict())
        assert restored._schema == {"tag": "str"}
        assert restored._data == {"tag": "hello"}
        assert restored.tag == "hello"

    def test_roundtrip_list_key(self) -> None:
        """Persisted list[str] key survives as_dict / from_dict."""
        e = Extras()
        e.set_type("ids", "list[str]", persist=True)
        e.dump("ids", "x")
        e.dump("ids", "y")
        restored = Extras.from_dict(e.as_dict())
        assert restored._schema == {"ids": "list[str]"}
        assert restored._data == {"ids": ["x", "y"]}

    def test_roundtrip_mixed(self) -> None:
        """Both key types survive together when both are persisted."""
        e = Extras()
        e.set_type("label", "str", persist=True)
        e.dump("label", "ok")
        e.set_type("refs", "list[str]", persist=True)
        e.dump("refs", "u1")
        e.dump("refs", "u2")
        restored = Extras.from_dict(e.as_dict())
        assert restored._data["label"] == "ok"
        assert restored._data["refs"] == ["u1", "u2"]

    def test_roundtrip_full_dump_preserves_transient_keys_too(self) -> None:
        """add_extras=True round trip keeps non-persisted keys as well."""
        e = Extras()
        e.set_type("label", "str", persist=True)
        e.dump("label", "ok")
        e.set_type("transient", "str", persist=False)
        e.dump("transient", "scratch")
        restored = Extras.from_dict(e.as_dict(add_extras=True))
        assert restored._data == {"label": "ok", "transient": "scratch"}
        assert restored._persist == {"label": True, "transient": False}

    def test_from_dict_without_persist_key_defaults_to_non_persistent(self) -> None:
        """Loading a dict produced before the persist feature existed (no '_persist' key) works."""
        legacy = {"_schema": {"tag": "str"}, "_data": {"tag": "hello"}}
        restored = Extras.from_dict(legacy)
        assert not restored._persist
        assert restored.tag == "hello"
        assert restored.as_dict() == {"_schema": {}, "_persist": {}, "_data": {}}


class TestImageExtrasAsDict:
    """Tests for Image.as_dict(add_extras=...) and Image(**dict_with_extras)."""

    def test_as_dict_default_excludes_transient_extras(self, white_image: WhiteImage) -> None:
        """as_dict() without add_extras does not include non-persisted extras."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str")
        img.dump_extra("tag", "hello")
        d = img.as_dict()
        assert "_extras" not in d

    def test_as_dict_default_includes_persisted_extras(self, white_image: WhiteImage) -> None:
        """as_dict() without add_extras still includes keys registered with persist=True."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str", persist=True)
        img.dump_extra("tag", "hello")
        d = img.as_dict()
        assert d["_extras"]["_data"]["tag"] == "hello"

    def test_as_dict_add_extras_includes_extras(self, white_image: WhiteImage) -> None:
        """as_dict(add_extras=True) includes _extras."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str")
        img.dump_extra("tag", "hello")
        d = img.as_dict(add_extras=True)
        assert "_extras" in d
        assert d["_extras"]["_data"]["tag"] == "hello"

    def test_reconstruct_from_dict_with_extras(self, white_image: WhiteImage) -> None:
        """Image(**dict_with_extras) properly restores Extras."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("ids", "list[str]")
        img.dump_extra("ids", "u1")
        img.dump_extra("ids", "u2")
        d = img.as_dict(add_extras=True)
        restored = Image(**d)
        assert restored.extras._data == {"ids": ["u1", "u2"]}
        assert restored.extras._schema == {"ids": "list[str]"}

    def test_reconstruct_without_extras_gives_fresh_store(self, white_image: WhiteImage) -> None:
        """Image(**dict_without_extras) produces an empty Extras store."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str")
        img.dump_extra("tag", "hello")
        d = img.as_dict()  # add_extras=False, "tag" is not persisted
        restored = Image(**d)
        assert restored.extras._data == {}
        assert restored.extras._schema == {}

    def test_as_json_excludes_non_persisted_extras(self, white_image: WhiteImage) -> None:
        """as_json() never exposes extras registered without persist=True."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str")
        img.dump_extra("tag", "hello")
        payload = json.loads(img.as_json())
        assert "_extras" not in payload

    def test_as_json_includes_persisted_extras_and_round_trips(self, white_image: WhiteImage) -> None:
        """as_json() includes persist=True keys, and Image(**json.loads(...)) recovers them."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("persisted", "str", persist=True)
        img.dump_extra("persisted", "kept")
        img.configure_extras("transient", "str", persist=False)
        img.dump_extra("transient", "gone")

        payload = json.loads(img.as_json())
        assert payload["_extras"]["_data"] == {"persisted": "kept"}

        restored = Image(**payload)
        assert restored.extras.persisted == "kept"
        with pytest.raises(AttributeError):
            _ = restored.extras.transient

    def test_save_from_file_round_trip_preserves_persisted_extras(
        self, white_image: WhiteImage, tmp_path: Path
    ) -> None:
        """A full save() -> from_file() round trip keeps persist=True extras and drops the rest."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("message", "str", persist=True)
        img.dump_extra("message", "e-mail body")
        img.configure_extras("scratch", "str", persist=False)
        img.dump_extra("scratch", "not kept")

        saved_path = img.save(path=tmp_path, dry=False)
        assert saved_path is not None
        restored = Image.from_file(str(saved_path))

        assert restored.extras.message == "e-mail body"
        with pytest.raises(AttributeError):
            _ = restored.extras.scratch
