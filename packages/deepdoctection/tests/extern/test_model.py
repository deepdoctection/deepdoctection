# -*- coding: utf-8 -*-
# File: test_model.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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
Tests for the extended functionalities of the ModelCatalog, ModelProfile, and
ModelDownloadManager classes in the DeepDoctection framework.

The module provides unit tests for verifying the methods and behaviors of the
model catalog registration, retrieval of model metadata, model paths, and
additional edge cases using monkey patches for isolated testing of IO operations.
The tests ensure correctness and robustness of the implemented functionalities
in both typical and mocked scenarios.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import pytest

from dd_core.utils.env_info import SETTINGS
from dd_core.utils.object_types import get_type
from deepdoctection.extern.model import (
    ModelCatalog,
    ModelDownloadManager,
    ModelProfile,
    print_model_infos,
)


class TestModelCatalogExtended:
    """
    Extended tests for ModelCatalog and ModelProfile
    """

    def setup_method(self) -> None:
        """
        Setup a test profile and register it.
        """
        self.profile = ModelProfile(
            name="test_model/test_model.pt",
            description="Test model profile",
            config="test_path/dd/conf_frcnn_cell.yaml",
            preprocessor_config=None,
            size=[1234],
            hf_repo_id="fake/repo",
            hf_model_name="weights.pt",
            hf_config_file=["conf_config.yaml"],
            urls=None,
            categories={1: get_type("cell")},  # categories are not used in these assertions
            dl_library="PT",
            model_wrapper="D2FrcnnDetector",
            architecture="GeneralizedRCNN",
            padding=False,
        )
        if "test_model/test_model.pt" not in ModelCatalog.CATALOG:
            ModelCatalog.register("test_model/test_model.pt", self.profile)

    def test_profile_as_dict_contains_fields(self) -> None:
        """
        ModelProfile.as_dict returns a dict with expected keys.
        """
        d = self.profile.as_dict()
        assert d["name"] == "test_model/test_model.pt"
        assert d["hf_model_name"] == "weights.pt"

    def test_model_catalog_get_model_list_contains_registered(self) -> None:
        """
        ModelCatalog.get_model_list returns absolute paths of registered models.
        """
        models = ModelCatalog.get_model_list()
        assert (SETTINGS.MODEL_DIR / "test_model/test_model.pt").as_posix() in models

    def test_model_catalog_get_profile_list_contains_key(self) -> None:
        """
        ModelCatalog.get_profile_list returns catalog keys.
        """
        keys = ModelCatalog.get_profile_list()
        assert "test_model/test_model.pt" in keys

    def test_model_catalog_is_registered_true_and_false(self) -> None:
        """
        ModelCatalog.is_registered returns True only for registered.
        """
        assert ModelCatalog.is_registered("test_model/test_model.pt") is True
        assert ModelCatalog.is_registered("unregistered_model") is False

    def test_model_catalog_full_paths(self) -> None:
        """
        ModelCatalog returns correct full paths for weights and configs.
        """
        assert (
            ModelCatalog.get_full_path_weights("test_model/test_model.pt")
            == (SETTINGS.MODEL_DIR / "test_model/test_model.pt").as_posix()
        )
        assert (
            ModelCatalog.get_full_path_configs("test_model/test_model.pt")
            == (SETTINGS.CONFIGS_DIR / "test_path/dd/conf_frcnn_cell.yaml").as_posix()
        )

    def test_print_model_infos_runs(self, capsys: Any) -> None:
        """
        print_model_infos produces a table output without error.
        """
        print_model_infos(add_description=True, add_config=True, add_categories=True)
        out = capsys.readouterr().out
        assert "name" in out
        assert "description" in out
        assert "config" in out


class TestModelDownloadManager:
    """
    Tests for ModelDownloadManager with monkeypatches for external IO.
    """

    def test_maybe_download_weights_and_configs_hf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        HF branch: download into a temp dir, not the production cache.
        """
        # Use a temporary directory for both MODEL_DIR and CONFIGS_DIR
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model_dir = Path(tmpdir) / "models"
            tmp_configs_dir = Path(tmpdir) / "configs"
            tmp_model_dir.mkdir(parents=True, exist_ok=True)
            tmp_configs_dir.mkdir(parents=True, exist_ok=True)

            monkeypatch.setattr(SETTINGS, "MODEL_DIR", tmp_model_dir, raising=False)
            monkeypatch.setattr(SETTINGS, "CONFIGS_DIR", tmp_configs_dir, raising=False)

            name = "hf_model/weights.bin"
            profile = ModelProfile(
                name=name,
                description="HF test",
                config="hf_model/config.yaml",
                preprocessor_config=None,
                size=[10],
                hf_repo_id="fake/repo",
                hf_model_name="weights.bin",
                hf_config_file=["config.yaml"],
                urls=None,
                categories=None,
                dl_library=None,
                model_wrapper=None,
            )
            if name not in ModelCatalog.CATALOG:
                ModelCatalog.register(name, profile)

            weights_abs = (SETTINGS.MODEL_DIR / name).as_posix()
            configs_abs = (SETTINGS.CONFIGS_DIR / "hf_model/config.yaml").as_posix()
            os.makedirs(os.path.dirname(weights_abs), exist_ok=True)
            os.makedirs(os.path.dirname(configs_abs), exist_ok=True)

            def _fake_hf_hub_download(
                repo_id: str,  # pylint: disable=W0613
                file_name: str,  # pylint: disable=W0613
                local_dir: str,
                force_filename: str,
                force_download: bool,  # pylint: disable=W0613
                token: Optional[str],  # pylint: disable=W0613
            ) -> str:
                target = os.path.join(local_dir, force_filename)
                os.makedirs(local_dir, exist_ok=True)
                with open(target, "wb") as f:
                    f.write(b"x" * 10)
                return target

            monkeypatch.setattr("deepdoctection.extern.model.hf_hub_download", _fake_hf_hub_download, raising=True)

            # Ensure clean state
            if os.path.isfile(weights_abs):
                os.remove(weights_abs)
            if os.path.isfile(configs_abs):
                os.remove(configs_abs)

            out = ModelDownloadManager.maybe_download_weights_and_configs(name)
            assert out == weights_abs
            assert os.path.isfile(weights_abs)
            assert os.path.getsize(weights_abs) == 10
            assert os.path.isfile(configs_abs)

    def test_maybe_download_weights_and_configs_gd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Google Drive branch: ensure it calls download helper and returns absolute weights path.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            tmp_model_dir = tmp_root / "models"
            tmp_configs_dir = tmp_root / "configs"
            tmp_model_dir.mkdir(parents=True, exist_ok=True)
            tmp_configs_dir.mkdir(parents=True, exist_ok=True)

            monkeypatch.setattr(SETTINGS, "MODEL_DIR", tmp_model_dir, raising=False)
            monkeypatch.setattr(SETTINGS, "CONFIGS_DIR", tmp_configs_dir, raising=False)

            name = "gd_model/weights.tgz"
            profile = ModelProfile(
                name=name,
                description="GD test",
                config="gd_model/config.yaml",
                preprocessor_config=None,
                size=[20],
                hf_repo_id=None,
                hf_model_name=None,
                hf_config_file=None,
                urls=["https://drive.fake/file?id=123"],
                categories=None,
                dl_library=None,
                model_wrapper=None,
            )

            if name not in ModelCatalog.CATALOG:
                ModelCatalog.register(name, profile)

            weights_abs = (SETTINGS.MODEL_DIR / name).as_posix()
            os.makedirs(os.path.dirname(weights_abs), exist_ok=True)

            def _fake_download(
                url: str, directory: str, file_name: str, expect_size: int  # pylint: disable=W0613
            ) -> None:
                target = os.path.join(directory, file_name)
                os.makedirs(directory, exist_ok=True)
                with open(target, "wb") as f:
                    f.write(b"x" * expect_size)

            monkeypatch.setattr("deepdoctection.extern.model.download", _fake_download, raising=True)

            out = ModelDownloadManager.maybe_download_weights_and_configs(name)
            assert out == weights_abs

    def test_maybe_download_no_sources_returns_path(self) -> None:
        """
        If profile has no sources, the method returns absolute path unchanged.
        """
        name = "nosrc/model.bin"
        profile = ModelProfile(
            name=name,
            description="No source",
            config=None,
            preprocessor_config=None,
            size=[],
            hf_repo_id=None,
            hf_model_name=None,
            hf_config_file=None,
            urls=None,
            categories=None,
            dl_library=None,
            model_wrapper=None,
        )
        if name not in ModelCatalog.CATALOG:
            ModelCatalog.register(name, profile)

        out = ModelDownloadManager.maybe_download_weights_and_configs(name)
        assert out == (SETTINGS.MODEL_DIR / name).as_posix()
