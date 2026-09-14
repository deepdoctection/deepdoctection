# -*- coding: utf-8 -*-
# File: test_env_info.py

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
Testing the module utils.env_info
"""

from __future__ import annotations

import os
from typing import Callable

import pytest
from pydantic import SecretStr

from dd_core.utils.env_info import EnvSettings


@pytest.fixture(name="isolated_environ")
def fixture_isolated_environ(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """`os.environ` replaced by a plain copy, so exporting does not leak into the test session"""
    environ = dict(os.environ)
    monkeypatch.setattr(os, "environ", environ)
    return environ


@pytest.fixture(name="settings_factory")
def fixture_settings_factory() -> Callable[..., EnvSettings]:
    """Factory building an `EnvSettings` without reading any `.env` file"""

    def _settings(**kwargs: object) -> EnvSettings:
        return EnvSettings(_env_file=None, **kwargs)  # type: ignore

    return _settings


class TestSecretsAreNotExported:
    """Secrets must live on the settings object only"""

    @staticmethod
    def test_export_to_environ_leaves_hf_credentials_untouched(
        isolated_environ: dict[str, str], settings_factory: Callable[..., EnvSettings]
    ) -> None:
        """
        `export_to_environ` must not overwrite an externally set `HF_CREDENTIALS` with the `SecretStr` mask
        """

        isolated_environ["HF_CREDENTIALS"] = "hf_a_real_token"
        settings = settings_factory(HF_CREDENTIALS="hf_a_real_token")

        settings.export_to_environ()

        assert isolated_environ["HF_CREDENTIALS"] == "hf_a_real_token"
        assert settings.HF_CREDENTIALS is not None
        assert settings.HF_CREDENTIALS.get_secret_value() == "hf_a_real_token"

    @staticmethod
    def test_export_to_environ_rejects_secrets(
        isolated_environ: dict[str, str],  # pylint: disable=W0613
        monkeypatch: pytest.MonkeyPatch,
        settings_factory: Callable[..., EnvSettings],
    ) -> None:
        """Adding a `SecretStr` to the exported variables must fail loudly instead of exporting the mask"""

        settings = settings_factory()
        monkeypatch.setattr(settings, "MODEL_CATALOG", SecretStr("secret"))

        with pytest.raises(TypeError):
            settings.export_to_environ()


class TestEmptySecrets:
    """An empty secret must behave exactly like an unset one"""

    @staticmethod
    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_hf_credentials_become_none(value: str, settings_factory: Callable[..., EnvSettings]) -> None:
        """`HF_CREDENTIALS=` must not yield `SecretStr('')`, which would produce an empty bearer header"""

        assert settings_factory(HF_CREDENTIALS=value).HF_CREDENTIALS is None

    @staticmethod
    def test_empty_aws_credentials_become_none(settings_factory: Callable[..., EnvSettings]) -> None:
        """The same normalization applies to the AWS secrets"""

        settings = settings_factory(AWS_ACCESS_KEY_ID="", AWS_SECRET_ACCESS_KEY="")

        assert settings.AWS_ACCESS_KEY_ID is None
        assert settings.AWS_SECRET_ACCESS_KEY is None

    @staticmethod
    def test_non_empty_hf_credentials_are_kept(settings_factory: Callable[..., EnvSettings]) -> None:
        """A regular token is stored as a secret"""

        settings = settings_factory(HF_CREDENTIALS="hf_a_real_token")

        assert settings.HF_CREDENTIALS is not None
        assert settings.HF_CREDENTIALS.get_secret_value() == "hf_a_real_token"
