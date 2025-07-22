# -*- coding: utf-8 -*-
# File: model.py

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
`ModelCatalog` and`ModelDownloadManager`
"""

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Union

import jsonlines
from huggingface_hub import hf_hub_download
from tabulate import tabulate
from termcolor import colored

from ..utils.fs import (
    download,
    get_cache_dir_path,
    get_configs_dir_path,
    get_package_path,
    get_weights_dir_path,
    maybe_copy_config_to_cache,
)
from ..utils.logger import LoggingRecord, log_once, logger
from ..utils.settings import ObjectTypes, get_type
from ..utils.types import PathLikeOrStr

__all__ = ["ModelCatalog", "ModelDownloadManager", "print_model_infos", "ModelProfile"]


@dataclass(frozen=True)
class ModelProfile:
    """
    Class for model profile. Add for each model one `ModelProfile` to the `ModelCatalog`
    """

    name: str
    description: str

    size: list[int]
    tp_model: bool = field(default=False)
    config: Optional[str] = field(default=None)
    preprocessor_config: Optional[str] = field(default=None)
    hf_repo_id: Optional[str] = field(default=None)
    hf_model_name: Optional[str] = field(default=None)
    hf_config_file: Optional[list[str]] = field(default=None)
    urls: Optional[list[str]] = field(default=None)
    categories: Optional[Mapping[int, ObjectTypes]] = field(default=None)
    categories_orig: Optional[Mapping[str, ObjectTypes]] = field(default=None)
    dl_library: Optional[str] = field(default=None)
    model_wrapper: Optional[str] = field(default=None)
    architecture: Optional[str] = field(default=None)
    padding: Optional[bool] = field(default=None)

    def as_dict(self) -> dict[str, Any]:
        """
        Returns:
            A dict of the dataclass
        """
        return asdict(self)


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well.

    To get an overview of all registered models

    Example:
        ```python
        print(ModelCatalog.get_model_list())
        ```

    To get a model card for some specific model:

    Example:
        ```python
        profile = ModelCatalog.get_profile("layout/model-800000_inf_only.data-00000-of-00001")
        print(profile.description)
        ```

    Some models will have their weights and configs stored in the cache. To instantiate predictors one will sometimes
    need their path. Use

    Example:
        ```python
        path_weights = ModelCatalog.get_full_path_configs("layout/model-800000_inf_only.data-00000-of-00001")
        path_configs = ModelCatalog.get_full_path_weights("layout/model-800000_inf_only.data-00000-of-00001")
        ```

    To register a new model

    Example:
        ```python
        ModelCatalog.get_full_path_configs("my_new_model")
        ```

    Attributes:
        CATALOG (dict[str, ModelProfile]): A dict of model profiles. The key is the model name and the value is a
            `ModelProfile` object.
    """

    CATALOG: dict[str, ModelProfile] = {}

    @staticmethod
    def get_full_path_weights(name: PathLikeOrStr) -> PathLikeOrStr:
        """
        Returns the absolute path of weights.

        Note:
            Weights are sometimes not defined by only one artifact. The returned string will only represent one
            weights artifact.

        Args:
            name: model name

        Returns:
            absolute weight path
        """
        try:
            profile = ModelCatalog.get_profile(os.fspath(name))
        except KeyError:
            logger.info(
                LoggingRecord(
                    f"Model {name} not found in ModelCatalog. Make sure, you have places model weights "
                    f"in the cache dir"
                )
            )
            profile = ModelProfile(name="", description="", size=[])
        if profile.name:
            return os.path.join(get_weights_dir_path(), profile.name)
        log_once(
            f"Model {name} is not registered. Please make sure the weights are available in the weights "
            f"cache directory or the full path you provide is correct"
        )
        if os.path.isfile(name):
            return name
        return os.path.join(get_weights_dir_path(), name)

    @staticmethod
    def get_full_path_configs(name: PathLikeOrStr) -> PathLikeOrStr:
        """
        Absolute path of configs for some given weights. Alternatively, pass a path to a config file
        (without the base path to the cache config directory).

        Note:
            Configs are sometimes not defined by only one file. The returned string will only represent one
            file.

        Args:
            name: model name

        Returns:
            Absolute path to the config
        """
        try:
            profile = ModelCatalog.get_profile(os.fspath(name))
        except KeyError:
            logger.info(
                LoggingRecord(
                    f"Model {name} not found in ModelCatalog. Make sure, you have places model "
                    f"configs in the cache dir"
                )
            )
            profile = ModelProfile(name="", description="", size=[])
        if profile.config is not None:
            return os.path.join(get_configs_dir_path(), profile.config)
        return os.path.join(get_configs_dir_path(), name)

    @staticmethod
    def get_full_path_preprocessor_configs(name: Union[str]) -> PathLikeOrStr:
        """
        Return the absolute path of preprocessor configs for some given weights. Preprocessor are occasionally provided
        by the transformer library.

        Args:
            name: model name

        Returns:
            Absolute path to the preprocessor config
        """

        try:
            profile = ModelCatalog.get_profile(name)
        except KeyError:
            profile = ModelProfile(name="", description="", size=[])
            logger.info(
                LoggingRecord(
                    f"Model {name} not found in ModelCatalog. Make sure, you have places preprocessor configs "
                    f"in the cache dir",
                )
            )
        if profile.preprocessor_config is not None:
            return os.path.join(get_configs_dir_path(), profile.preprocessor_config)
        return os.path.join(get_configs_dir_path(), name)

    @staticmethod
    def get_model_list() -> list[PathLikeOrStr]:
        """
        Returns:
            A list of absolute paths of registered models.
        """
        return [os.path.join(get_weights_dir_path(), profile.name) for profile in ModelCatalog.CATALOG.values()]

    @staticmethod
    def get_profile_list() -> list[str]:
        """
        Returns:
            A list profile keys.
        """
        return list(ModelCatalog.CATALOG.keys())

    @staticmethod
    def is_registered(path_weights: PathLikeOrStr) -> bool:
        """
        Checks if some weights belong to a registered model

        Args:
            path_weights: relative or absolute path

        Returns:
            `True` if the weights are registered in `ModelCatalog`
        """
        if (ModelCatalog.get_full_path_weights(path_weights) in ModelCatalog.get_model_list()) or (
            path_weights in ModelCatalog.get_model_list()
        ):
            return True
        return False

    @staticmethod
    def get_profile(name: str) -> ModelProfile:
        """
        Returns the profile of given model name, i.e. the config file, size and urls.

        Args:
            name: model name

        Returns:
            A dict of model/weights profiles
        """

        profile = ModelCatalog.CATALOG.get(name)
        if profile is not None:
            return profile
        raise KeyError(f"Model Profile {name} does not exist. Please make sure the model is registered")

    @staticmethod
    def register(name: str, profile: ModelProfile) -> None:
        """
        Register a model with its profile

        Args:
            name: Name of the model. We use the file name of the model along with its path (starting from the
                  weights `.cache`. e.g. `my_model/model_123.pkl`.
            profile: profile of the model
        """
        if name in ModelCatalog.CATALOG:
            raise KeyError("Model already registered")
        ModelCatalog.CATALOG[name] = profile

    @staticmethod
    def load_profiles_from_file(path: Optional[PathLikeOrStr] = None) -> None:
        """
        Load model profiles from a `jsonl` file and extend `CATALOG` with the new profiles.

        Args:
            path: Path to the file. `None` is allowed but will do nothing.
        """
        if not path:
            return
        with jsonlines.open(path) as reader:
            for obj in reader:
                if not obj["name"] in ModelCatalog.CATALOG:
                    categories = obj.get("categories") or {}
                    obj["categories"] = {int(key): get_type(val) for key, val in categories.items()}
                    ModelCatalog.register(obj["name"], ModelProfile(**obj))

    @staticmethod
    def save_profiles_to_file(target_path: PathLikeOrStr) -> None:
        """
        Save model profiles to a `jsonl` file.

        Args:
            target_path: Path to the file.
        """
        with jsonlines.open(target_path, mode="w") as writer:
            for profile in ModelCatalog.CATALOG.values():
                writer.write(profile.as_dict())
        writer.close()


# Loading default profiles
dd_profile_path = maybe_copy_config_to_cache(
    get_package_path(), get_cache_dir_path(), "deepdoctection/configs/profiles.jsonl", True
)
ModelCatalog.load_profiles_from_file(dd_profile_path)
# Additional profiles can be added
ModelCatalog.load_profiles_from_file(os.environ.get("MODEL_CATALOG", None))


def get_tp_weight_names(name: str) -> list[str]:
    """
    Given a path to some model weights it will return all file names according to TP naming convention

    Args:
        name: TP model name

    Returns:
        A list of TP file names
    """
    _, file_name = os.path.split(name)
    prefix, _ = file_name.split(".")
    weight_names = []
    for suffix in ["data-00000-of-00001", "index"]:
        weight_names.append(prefix + "." + suffix)

    return weight_names


def print_model_infos(add_description: bool = True, add_config: bool = True, add_categories: bool = True) -> None:
    """
    Prints a table with all registered model profiles and some of their attributes (name, description, config and
    categories)

    Args:
        add_description: If `True`, the description of the model will be printed
        add_config: If `True`, the config of the model will be printed
        add_categories: If `True`, the categories of the model will be printed
    """

    profiles = ModelCatalog.CATALOG.values()
    num_columns = min(6, len(profiles))
    infos = []
    for profile in profiles:
        tbl_input: list[Union[Mapping[int, ObjectTypes], str]] = [profile.name]
        if add_description:
            tbl_input.append(profile.description)
        if add_config:
            tbl_input.append(profile.config if profile.config else "")
        if add_categories:
            tbl_input.append(profile.categories if profile.categories else {})
        infos.append(tbl_input)
    tbl_header = ["name"]
    if add_description:
        tbl_header.append("description")
    if add_config:
        tbl_header.append("config")
    if add_categories:
        tbl_header.append("categories")
    table = tabulate(
        infos,
        headers=tbl_header * (num_columns // 2),
        tablefmt="fancy_grid",
        stralign="left",
        numalign="left",
    )
    print(colored(table, "cyan"))


class ModelDownloadManager:
    """
    Class for organizing downloads of config files and weights from various sources. Internally, it will use model
    profiles to know where things are stored.

    Example:
        ```python
        # if you are not sure about the model name use the ModelCatalog
        ModelDownloadManager.maybe_download_weights_and_configs("layout/model-800000_inf_only.data-00000-of-00001")
        ```
    """

    @staticmethod
    def maybe_download_weights_and_configs(name: str) -> PathLikeOrStr:
        """
        Check if some model is registered. If yes, it will check if their weights
        must be downloaded. Only weights that have not the same expected size will be downloaded again.

        Args:
            name: A path to some model weights
        Returns:
            Absolute path to model weights, if model is registered
        """

        absolute_path_weights = ModelCatalog.get_full_path_weights(name)
        file_names: list[str] = []
        if ModelCatalog.is_registered(name):
            profile = ModelCatalog.get_profile(name)
            # there is nothing to download if hf_repo_id or urls is not provided
            if not profile.hf_repo_id and not profile.urls:
                return absolute_path_weights
            # determine the right model name
            if profile.tp_model:
                file_names = get_tp_weight_names(name)
            else:
                model_name = profile.hf_model_name
                if model_name is None:
                    # second try. Check if a url is provided
                    if profile.urls is None:
                        raise ValueError("hf_model_name and urls cannot be both None")
                    for url in profile.urls:
                        file_names.append(url.split("/")[-1].split("&")[0])
                else:
                    file_names.append(model_name)
            if profile.hf_repo_id:
                if not os.path.isfile(absolute_path_weights):
                    ModelDownloadManager.load_model_from_hf_hub(profile, absolute_path_weights, file_names)
                absolute_path_configs = ModelCatalog.get_full_path_configs(name)
                if not os.path.isfile(absolute_path_configs):
                    ModelDownloadManager.load_configs_from_hf_hub(profile, absolute_path_configs)
            else:
                ModelDownloadManager._load_from_gd(profile, absolute_path_weights, file_names)

            return absolute_path_weights

        return absolute_path_weights

    @staticmethod
    def load_model_from_hf_hub(profile: ModelProfile, absolute_path: PathLikeOrStr, file_names: list[str]) -> None:
        """
        Load a model from the Huggingface hub for a given profile and saves the model at the directory of the given
        path.

        Args:
            profile: Profile according to `ModelCatalog.get_profile(path_weights)`
            absolute_path: Absolute path (incl. file name) of target file
            file_names: Optionally, replace the file name of the `ModelCatalog`. This is necessary e.g. for Tensorpack
                        models
        """
        repo_id = profile.hf_repo_id
        if repo_id is None:
            raise ValueError("hf_repo_id cannot be None")
        directory, _ = os.path.split(absolute_path)

        for expect_size, file_name in zip(profile.size, file_names):
            size = ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)
            if expect_size is not None and size != expect_size:
                logger.error(
                    LoggingRecord(
                        f"File downloaded from {repo_id} does not match the expected size! You may have downloaded"
                        " a broken file, or the upstream may have modified the file."
                    )
                )

    @staticmethod
    def _load_from_gd(profile: ModelProfile, absolute_path: PathLikeOrStr, file_names: list[str]) -> None:
        if profile.urls is None:
            raise ValueError("urls cannot be None")
        for size, url, file_name in zip(profile.size, profile.urls, file_names):
            directory, _ = os.path.split(absolute_path)
            download(str(url), directory, file_name, int(size))

    @staticmethod
    def load_configs_from_hf_hub(profile: ModelProfile, absolute_path: PathLikeOrStr) -> None:
        """
        Load config file(s) from the Huggingface hub for a given profile and saves the model at the directory of the
        given path.

        Args:
            profile: Profile according to `ModelCatalog.get_profile(path_weights)`
            absolute_path:  Absolute path (incl. file name) of target file
        """

        repo_id = profile.hf_repo_id
        if repo_id is None:
            raise ValueError("hf_repo_id cannot be None")
        directory, _ = os.path.split(absolute_path)
        if profile.hf_config_file is None:
            raise ValueError("hf_config_file cannot be None")
        for file_name in profile.hf_config_file:
            ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)

    @staticmethod
    def _load_from_hf_hub(
        repo_id: str, file_name: str, cache_directory: PathLikeOrStr, force_download: bool = False
    ) -> int:
        token = os.environ.get("HF_CREDENTIALS", None)
        f_path = hf_hub_download(
            repo_id,
            file_name,
            local_dir=cache_directory,  # type: ignore
            force_filename=file_name,
            force_download=force_download,
            token=token,
        )
        if f_path:
            stat_info = os.stat(f_path)
            size = stat_info.st_size

            assert size > 0, f"Downloaded an empty file from {f_path}!"
            return size
        raise TypeError("Returned value from cached_download cannot be Null")
