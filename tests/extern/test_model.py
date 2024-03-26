# -*- coding: utf-8 -*-
# File: test_model.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Testing module extern.model
"""

from pytest import mark, raises

from deepdoctection.extern.model import ModelCatalog, ModelProfile
from deepdoctection.utils.fs import get_configs_dir_path, get_weights_dir_path
from deepdoctection.utils.settings import LayoutType


class TestModelCatalog:
    """
    Test ModelCatalog
    """

    def setup_method(self) -> None:
        """
        setup necessary components
        """

        self.profile = ModelProfile(
            name="test_model/test_model.pt",
            description="Test model profile",
            config="test_path/dd/conf_frcnn_cell.yaml",
            size=[],
            tp_model=False,
            hf_repo_id="",
            hf_model_name="",
            hf_config_file=["conf_config.yaml"],
            categories={"1": LayoutType.cell},
            dl_library="PT",
            model_wrapper="D2FrcnnDetector",
        )
        if "test_model/test_model.pt" not in ModelCatalog.CATALOG:
            ModelCatalog.register("test_model/test_model.pt", self.profile)

    @mark.basic
    def test_model_catalog_returns_correct_profile(self) -> None:
        """
        ModelCatalog returns correct profile
        """

        # Act
        profile = ModelCatalog.get_profile("test_model/test_model.pt")

        # Assert
        assert profile == self.profile


class TestModelCatalogNoSetupRequired:
    """
    Test ModelCatalog test cases that require not setup methods
    """

    @staticmethod
    @mark.basic
    def test_model_catalog_raises_error_if_model_not_found() -> None:
        """
        ModelCatalog raises error if model not found
        """

        # Act
        with raises(KeyError):
            ModelCatalog.get_profile("unregistered_model")

    @staticmethod
    @mark.basic
    def test_model_catalog_full_path_weights() -> None:
        """
        ModelCatalog returns correct full path weights
        """

        # Assert
        assert (
            ModelCatalog.get_full_path_weights("test_model/test_model.pt")
            == (get_weights_dir_path() / "test_model/test_model.pt").as_posix()
        )

    @staticmethod
    @mark.basic
    def test_model_catalog_get_some_unregistered_model_full_path_weights() -> None:
        """
        ModelCatalog returns correct full path weights even if model is not registered
        """

        # Assert
        assert (
            ModelCatalog.get_full_path_weights("test_path/other_model.pt")
            == (get_weights_dir_path() / "test_path/other_model.pt").as_posix()
        )

    @staticmethod
    @mark.basic
    def test_model_catalog_full_path_configs() -> None:
        """
        ModelCatalog returns correct full path configs
        """

        # Assert
        assert (
            ModelCatalog.get_full_path_configs("test_model/test_model.pt")
            == (get_configs_dir_path() / "test_path/dd/conf_frcnn_cell.yaml").as_posix()
        )

    @staticmethod
    @mark.basic
    def test_model_catalog_get_some_unregistered_model_full_path_configs() -> None:
        """
        ModelCatalog returns correct full path configs even if model is not registered
        """

        # Assert
        assert (
            ModelCatalog.get_full_path_configs("test_path/other_model.pt")
            == (get_configs_dir_path() / "test_path/other_model.pt").as_posix()
        )

    @staticmethod
    @mark.basic
    def test_model_catalog_is_registered() -> None:
        """
        ModelCatalog returns correct registered flag
        """

        # Assert
        assert ModelCatalog.is_registered("test_model/test_model.pt") is True
        assert ModelCatalog.is_registered("unregistered_model") is False
