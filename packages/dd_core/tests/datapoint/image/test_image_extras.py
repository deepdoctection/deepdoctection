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


class TestExtrasSerialization:
    """Round-trip tests for Extras.as_dict / from_dict."""

    def test_as_dict_shape(self) -> None:
        """as_dict returns dict with _schema and _data keys."""
        e = Extras()
        e.set_type("tag", "str")
        e.dump("tag", "hello")
        d = e.as_dict()
        assert "_schema" in d
        assert "_data" in d

    def test_roundtrip_str_key(self) -> None:
        """str key survives as_dict / from_dict."""
        e = Extras()
        e.set_type("tag", "str")
        e.dump("tag", "hello")
        restored = Extras.from_dict(e.as_dict())
        assert restored._schema == {"tag": "str"}
        assert restored._data == {"tag": "hello"}

    def test_roundtrip_list_key(self) -> None:
        """list[str] key survives as_dict / from_dict."""
        e = Extras()
        e.set_type("ids", "list[str]")
        e.dump("ids", "x")
        e.dump("ids", "y")
        restored = Extras.from_dict(e.as_dict())
        assert restored._schema == {"ids": "list[str]"}
        assert restored._data == {"ids": ["x", "y"]}

    def test_roundtrip_mixed(self) -> None:
        """Both key types survive together."""
        e = Extras()
        e.set_type("label", "str")
        e.dump("label", "ok")
        e.set_type("refs", "list[str]")
        e.dump("refs", "u1")
        e.dump("refs", "u2")
        restored = Extras.from_dict(e.as_dict())
        assert restored._data["label"] == "ok"
        assert restored._data["refs"] == ["u1", "u2"]


class TestImageExtrasAsDict:
    """Tests for Image.as_dict(add_extras=...) and Image(**dict_with_extras)."""

    def test_as_dict_default_excludes_extras(self, white_image: WhiteImage) -> None:
        """as_dict() without add_extras does not include _extras."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str")
        img.dump_extra("tag", "hello")
        d = img.as_dict()
        assert "_extras" not in d

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
        d = img.as_dict()  # add_extras=False
        restored = Image(**d)
        assert restored.extras._data == {}
        assert restored.extras._schema == {}

    def test_as_json_never_includes_extras(self, white_image: WhiteImage) -> None:
        """as_json() never exposes _extras regardless of configured keys."""
        img = Image(file_name=white_image.file_name, location=white_image.location)
        img.configure_extras("tag", "str")
        img.dump_extra("tag", "hello")
        payload = json.loads(img.as_json())
        assert "_extras" not in payload
