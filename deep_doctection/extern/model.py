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
from typing import Dict, List, Union, Any
from huggingface_hub import hf_hub_url, cached_download  # type: ignore
from ..utils.logger import logger
from ..utils.fs import download
from ..utils.systools import get_weights_dir_path


__all__ = ["ModelCatalog", "ModelDownloadManager"]


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well
    """

    S_PREFIX = "https://www.googleapis.com/drive/v3/files"

    MODELS: Dict[str, Any] = {
        "layout/model-800000_inf_only.data-00000-of-00001": {
            "config": "configs/dd/tp/conf_frcnn_layout",
            "size": [274552244, 7907],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            "hf_model_name": "model-800000_inf_only",
            "tp_model": True,
        },
        "cell/model-1800000_inf_only.data-00000-of-00001": {
            "config": "configs/dd/tp/conf_frcnn_cell",
            "size": [274503056, 8056],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            "hf_model_name": "model-1800000_inf_only",
            "tp_model": True,
        },
        "item/model-1370000_inf_only.data-00000-of-00001": {
            "config": "configs/dd/tp/conf_frcnn_rows",
            "size": [274515344, 7904],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            "hf_model_name": "model-1370000_inf_only",
            "tp_model": True,
        },
        "item/model-1370000.data-00000-of-00001": {
            "config": "configs/dd/tp/conf_frcnn_rows",
            "size": [823546048, 25787],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc",
            "hf_model_name": "model-1370000",
            "tp_model": True,
        },
        "layout/model-800000.data-00000-of-00001": {
            "config": "configs/dd/tp/conf_frcnn_layout",
            "size": [823656748, 25796],
            "hf_repo_id": "deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet",
            "hf_model_name": "model-800000",
            "tp_model": True,
        },
        "cell/model-1800000.data-00000-of-00001": {
            "config": "configs/dd/tp/conf_frcnn_cell",
            "size": [823509160, 25905],
            "hf_repo_id": "ddeepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c",
            "hf_model_name": "model-1800000",
            "tp_model": True,
        },
        "layout/d2_model-800000-layout.pkl": {
            "config": "configs/dd/d2/layout/CASCADE_RCNN_R_50_FPN_GN.yaml",
            "size": [274568239],
            "hf_repo_id": "deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
            "hf_model_name": "d2_model-800000-layout.pkl",
            "tp_model": False,
        },
        "cell/d2_model-1800000-cell.pkl": {
            "config": "configs/dd/d2/cell/CASCADE_RCNN_R_50_FPN_GN.yaml",
            "size": [274519039],
            "hf_repo_id": "deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_c_inference_only",
            "hf_model_name": "d2_model-1800000-cell.pkl",
            "tp_model": False,
        },
        "item/d2_model-1620000-item.pkl": {
            "config": "configs/dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml",
            "size": [274531339],
            "hf_repo_id": "deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_pubtabnet_rc_inference_only",
            "hf_model_name": "d2_model-1620000-item.pkl",
            "tp_model": False,
        },
    }

    @staticmethod
    def get_full_path_weights(path_weights: str) -> str:
        """
        Returns the absolute path of weights

        :param path_weights: relative weight path
        :return: absolute weight path
        """
        return os.path.join(get_weights_dir_path(), path_weights)

    @staticmethod
    def get_weights_names() -> List[str]:
        """
        Get a list of available weights names

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
    def is_registered(path_weights: str) -> bool:
        """
        Checks if a relative path of weights belongs to a registered model
        :param path_weights: relative path
        :return: True if the weights are registered in :class:`ModelCatalog`
        """
        if ModelCatalog.get_full_path_weights(path_weights) in ModelCatalog.get_weights_list():
            return True
        return False

    @staticmethod
    def get_profile(path_weights: str) -> Dict[str, Any]:
        """
        Returns the profile of given local weights, i.e. the config file, size and urls.

        :param path_weights: local weights
        :return: A dict of model/weights profiles
        """
        profile = copy(ModelCatalog.MODELS[path_weights])
        profile["urls"] = [ModelCatalog.S_PREFIX + "/" + str(url) for url in profile.get("urls", "")]
        return profile


def get_tp_weight_names(path_weights: str) -> List[str]:
    """
    Given a path to some model weights it will return all file names according to TP naming convention

    :param path_weights: An path to some weights
    :return: A list of TP file names
    """
    _, file_name = os.path.split(path_weights)
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
    def maybe_download_weights(path_weights: str, from_hf_hub: bool = True) -> str:
        """
        Check if the path pointing to weight points to some registered weights. If yes, it will check if their weights
        must be downloaded. Only weights that have not the same expected size will be downloaded again.

        :param path_weights: A path to some model weights
        :param from_hf_hub: If True, will use model download from the Huggingface hub
        :return: Absolute path to model weights if model is registered
        """

        absolute_path = os.path.join(get_weights_dir_path(), path_weights)
        file_names: List[str] = []
        if ModelCatalog.is_registered(path_weights):
            profile = ModelCatalog.get_profile(path_weights)
            if profile["tp_model"]:
                file_names = get_tp_weight_names(path_weights)
            else:
                assert isinstance(profile["hf_model_name"], str)
                file_names.append(profile["hf_model_name"])
            if from_hf_hub:
                ModelDownloadManager._load_from_hf_hub(profile, absolute_path, file_names)
            else:
                ModelDownloadManager._load_from_gd(profile, absolute_path, file_names)

            return absolute_path

        logger.info("Will use not registered model. Make sure path to weights is correctly set")
        return absolute_path

    @staticmethod
    def _load_from_hf_hub(
        profile: Dict[str, Any], absolute_path: str, file_names: List[str]
    ) -> None:
        repo_id = profile["hf_repo_id"]
        if not file_names:
            file_names = profile["hf_model_name"]
        for expect_size, file_name in zip(profile["size"], file_names):
            directory, _ = os.path.split(absolute_path)
            url = hf_hub_url(repo_id=repo_id, filename=file_name)
            f_path = cached_download(url, cache_dir=directory, force_filename=file_name)
            stat_info = os.stat(f_path)
            size = stat_info.st_size
            assert size > 0, f"Downloaded an empty file from {url}!"
            if expect_size is not None and size != expect_size:
                logger.error("File downloaded from %s does not match the expected size!", url)
                logger.error("You may have downloaded a broken file, or the upstream may have modified the file.")

    @staticmethod
    def _load_from_gd(profile: Dict[str, List[Union[int, str]]], absolute_path: str, file_names: List[str]) -> None:
        for size, url, file_name in zip(profile["size"], profile["urls"], file_names):
            directory, _ = os.path.split(absolute_path)
            download(str(url), directory, file_name, int(size))
