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
Module for ModelCatalog and ModelDownloadManager
"""

import os
from copy import copy
from typing import Any, Dict, List, Union

from huggingface_hub import cached_download, hf_hub_url  # type: ignore

from ..utils.fs import download
from ..utils.logger import logger
from ..utils.systools import get_configs_dir_path, get_weights_dir_path

__all__ = ["ModelCatalog", "ModelDownloadManager"]


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well.
    """

    S_PREFIX = "https://www.googleapis.com/drive/v3/files"

    MODELS: Dict[str, Any] = {
        "layout/model-800000_inf_only.data-00000-of-00001": {
            "config": "dd/tp/conf_frcnn_layout",
            "size": [274552244, 7907],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            "hf_model_name": "model-800000_inf_only",
            "hf_config_file": ["conf_frcnn_layout.yaml"],
            "tp_model": True,
        },
        "cell/model-1800000_inf_only.data-00000-of-00001": {
            "config": "dd/tp/conf_frcnn_cell",
            "size": [274503056, 8056],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            "hf_model_name": "model-1800000_inf_only",
            "hf_config_file": ["conf_frcnn_cell.yaml"],
            "tp_model": True,
        },
        "item/model-1620000_inf_only.data-00000-of-00001": {
            "config": "dd/tp/conf_frcnn_rows",
            "size": [274515344, 7904],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            "hf_model_name": "model-1620000_inf_only",
            "hf_config_file": ["conf_frcnn_rows.yaml"],
            "tp_model": True,
        },
        "item/model-1620000.data-00000-of-00001": {
            "config": "dd/tp/conf_frcnn_rows",
            "size": [823546048, 25787],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc",
            "hf_model_name": "model-1620000",
            "hf_config_file": ["conf_frcnn_rows.yaml"],
            "tp_model": True,
        },
        "layout/model-800000.data-00000-of-00001": {
            "config": "dd/tp/conf_frcnn_layout",
            "size": [823656748, 25796],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet",
            "hf_model_name": "model-800000",
            "hf_config_file": ["conf_frcnn_layout.yaml"],
            "tp_model": True,
        },
        "cell/model-1800000.data-00000-of-00001": {
            "config": "dd/tp/conf_frcnn_cell",
            "size": [823509160, 25905],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c",
            "hf_model_name": "model-1800000",
            "hf_config_file": ["conf_frcnn_cell.yaml"],
            "tp_model": True,
        },
        "layout/d2_model-800000-layout.pkl": {
            "config": "dd/d2/layout/CASCADE_RCNN_R_50_FPN_GN",
            "size": [274568239],
            "hf_repo_id": "deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            "hf_model_name": "d2_model-800000-layout.pkl",
            "hf_config_file": ["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            "tp_model": False,
        },
        "cell/d2_model-1800000-cell.pkl": {
            "config": "dd/d2/cell/CASCADE_RCNN_R_50_FPN_GN",
            "size": [274519039],
            "hf_repo_id": "deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            "hf_model_name": "d2_model-1800000-cell.pkl",
            "hf_config_file": ["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            "tp_model": False,
        },
        "item/d2_model-1620000-item.pkl": {
            "config": "dd/d2/item/CASCADE_RCNN_R_50_FPN_GN",
            "size": [274531339],
            "hf_repo_id": "deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            "hf_model_name": "d2_model-1620000-item.pkl",
            "hf_config_file": ["Base-RCNN-FPN.yaml", "CASCADE_RCNN_R_50_FPN_GN.yaml"],
            "tp_model": False,
        },
    }

    @staticmethod
    def get_full_path_weights(weights: str) -> str:
        """
        Returns the absolute path of weights.

        Note, that weights are sometimes not defined by only one file. The returned string will only represent one
        file.

        :param weights: weights
        :return: absolute weight path
        """
        return os.path.join(get_weights_dir_path(), weights)

    @staticmethod
    def get_full_path_configs(weights: str) -> str:
        """
        Return the absolute path of configs for some given weights.

        Note, that configs are sometimes not defined by only one file. The returned string will only represent one
        file.

        :param weights: weights
        :return: absolute path to the config
        """
        profile = ModelCatalog.get_profile(weights)

        return os.path.join(get_configs_dir_path(), profile["config"]) + ".yaml"

    @staticmethod
    def get_weights_names() -> List[str]:
        """
        Get a list of available weights

        :return: A list of names
        """
        return list(ModelCatalog.MODELS.keys())

    @staticmethod
    def print_weights_names() -> None:
        """
        Print a list of registered weights names
        """
        print(list(ModelCatalog.MODELS.keys()))

    @staticmethod
    def get_weights_list() -> List[str]:
        """
        Returns a list of absolute paths of registered weights.
        """
        return [os.path.join(get_weights_dir_path(), key) for key in ModelCatalog.MODELS]

    @staticmethod
    def is_registered(weights: str) -> bool:
        """
        Checks if some weights belong to a registered model

        :param weights: relative path
        :return: True if the weights are registered in :class:`ModelCatalog`
        """
        if ModelCatalog.get_full_path_weights(weights) in ModelCatalog.get_weights_list():
            return True
        return False

    @staticmethod
    def get_profile(weights: str) -> Dict[str, Any]:
        """
        Returns the profile of given weights, i.e. the config file, size and urls.

        :param weights: local weights
        :return: A dict of model/weights profiles
        """
        profile = copy(ModelCatalog.MODELS[weights])
        profile["urls"] = [ModelCatalog.S_PREFIX + "/" + str(url) for url in profile.get("urls", "")]
        return profile


def get_tp_weight_names(weights: str) -> List[str]:
    """
    Given a path to some model weights it will return all file names according to TP naming convention

    :param weights: weights
    :return: A list of TP file names
    """
    _, file_name = os.path.split(weights)
    prefix, _ = file_name.split(".")
    weight_names = []
    for suffix in ["data-00000-of-00001", "index"]:
        weight_names.append(prefix + "." + suffix)

    return weight_names


class ModelDownloadManager:  # pylint: disable=R0903
    """
    A registry for built-in models. Registered models have weights that can be downloaded and cached. Do not use
    this class for registering your own models as there are much more sophisticated tools for experimenting and
    versioning.
    """

    @staticmethod
    def maybe_download_weights_and_configs(weights: str, from_hf_hub: bool = True) -> str:
        """
        Check if some weights belong to some registered weights. If yes, it will check if their weights
        must be downloaded. Only weights that have not the same expected size will be downloaded again.

        :param weights: A path to some model weights
        :param from_hf_hub: If True, will use model download from the Huggingface hub
        :return: Absolute path to model weights if model is registered
        """

        absolute_path_weights = ModelCatalog.get_full_path_weights(weights)
        file_names: List[str] = []
        if ModelCatalog.is_registered(weights):
            profile = ModelCatalog.get_profile(weights)
            if profile["tp_model"]:
                file_names = get_tp_weight_names(weights)
            else:
                assert isinstance(profile["hf_model_name"], str)
                file_names.append(profile["hf_model_name"])
            if from_hf_hub:
                ModelDownloadManager.load_model_from_hf_hub(profile, absolute_path_weights, file_names)
                absolute_path_configs = ModelCatalog.get_full_path_configs(weights)
                ModelDownloadManager.load_configs_from_hf_hub(profile, absolute_path_configs)
            else:
                ModelDownloadManager._load_from_gd(profile, absolute_path_weights, file_names)

            return absolute_path_weights

        logger.info("Will use not registered model. Make sure path to weights is correctly set")
        return absolute_path_weights

    @staticmethod
    def load_model_from_hf_hub(profile: Dict[str, Any], absolute_path: str, file_names: List[str]) -> None:
        """
        Load a model from the Huggingface hub for a given profile and saves the model at the directory of the given
        path.

        :param profile: Profile according to :func:`ModelCatalog.get_profile(path_weights)`
        :param absolute_path: Absolute path (incl. file name) of target file
        :param file_names: Optionally, replace the file name of the ModelCatalog. This is necessary e.g. for Tensorpack
                           models
        """
        repo_id = profile["hf_repo_id"]
        directory, _ = os.path.split(absolute_path)
        if not file_names:
            file_names = profile["hf_model_name"]
        for expect_size, file_name in zip(profile["size"], file_names):
            size = ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)
            if expect_size is not None and size != expect_size:
                logger.error("File downloaded from %s does not match the expected size!", repo_id)
                logger.error("You may have downloaded a broken file, or the upstream may have modified the file.")

    @staticmethod
    def _load_from_gd(profile: Dict[str, List[Union[int, str]]], absolute_path: str, file_names: List[str]) -> None:
        for size, url, file_name in zip(profile["size"], profile["urls"], file_names):
            directory, _ = os.path.split(absolute_path)
            download(str(url), directory, file_name, int(size))

    @staticmethod
    def load_configs_from_hf_hub(profile: Dict[str, Any], absolute_path: str) -> None:
        """
        Load config file(s) from the Huggingface hub for a given profile and saves the model at the directory of the
        given path.

        :param profile: Profile according to :func:`ModelCatalog.get_profile(path_weights)`
        :param absolute_path:  Absolute path (incl. file name) of target file
        """

        repo_id = profile["hf_repo_id"]
        directory, _ = os.path.split(absolute_path)
        for file_name in profile["hf_config_file"]:
            ModelDownloadManager._load_from_hf_hub(repo_id, file_name, directory)

    @staticmethod
    def _load_from_hf_hub(repo_id: str, file_name: str, cache_directory: str, force_download: bool = False) -> int:
        url = hf_hub_url(repo_id=repo_id, filename=file_name)
        f_path = cached_download(
            url, cache_dir=cache_directory, force_filename=file_name, force_download=force_download
        )
        stat_info = os.stat(f_path)
        size = stat_info.st_size
        assert size > 0, f"Downloaded an empty file from {url}!"
        return size
