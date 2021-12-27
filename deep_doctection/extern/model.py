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

from ..utils.logger import logger
from ..utils.fs import download
from ..utils.systools import get_weights_dir_path


class ModelCatalog:
    """
    Catalog of some pre-trained models. The associated config file is available as well
    """

    S_PREFIX = "https://www.googleapis.com/drive/v3/files"

    MODELS: Dict[str, Any] = {
        "layout/model-2026500.data-00000-of-00001": {
            "config": "configs/dd/conf_frcnn_layout",
            "size": [1134895140, 58919],
            "urls": [
                "1d8_BBWRGbhMmqZs46TyIPVcO7KBb_mJW?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",  # pylint:
                # disable=C0301
                "11kfvkgwMSUf3ERUvMW03DUm3AhWi-cNj?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",  # pylint: disable=C0301
            ],
        },
        "cell/model-2840000.data-00000-of-00001": {
            "config": "configs/dd/conf_frcnn_cell",
            "size": [823690432, 26583],
            "urls": [
                "1MgaXIcrPCDmc9j1t6EOGdpEjJ4CXY3Au?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
                "1xyn1TFlSR-rdb3fgiEQ7px9c7JH1Fhlk?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
            ],
        },
        "item/model-1750000.data-00000-of-00001": {
            "config": "configs/dd/conf_frcnn_rows",
            "size": [823690432, 26567],
            "urls": [
                "1JryTMNLxigri_Q-4pzElBxfkj_AOAEQE?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
                "1rlwvCnki5gCPojA1A2f-ztXvacYQ61CJ?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
            ],
        },
        "item/model-2750000.data-00000-of-00001": {
            "config": "configs/dd/conf_frcnn_rows",
            "size": [823690432, 26583],
            "urls": [
                "1v86gz7014QzqxtWpJT7osT9kpxRYqp9c?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
                "1wdT9QahyNMHHkSm4kHRqInTEHU2Uztu5?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
            ],
        },
        "cell/model-3550000.data-00000-of-00001": {
            "config": "configs/dd/conf_frcnn_cell",
            "size": [823653532, 26583],
            "urls": [
                "1t0q8FKa7lak24M7RKT5kCNHzM3PhnTpp?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
                "1DPzfMnBsd1cg_CXQq7_EJhZxwWslkFw1?alt=media&key=AIzaSyDuoPG6naK-kRJikScR7cP_1sQBF1r3fWU",
            ],
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
    def get_profile(path_weights: str) -> Dict[str, List[Union[int, str]]]:
        """
        Returns the profile of given local weights, i.e. the config file, size and urls.

        :param path_weights: local weights
        :return: A dict of model/weights profiles
        """
        profile = copy(ModelCatalog.MODELS[path_weights])
        profile["urls"] = [ModelCatalog.S_PREFIX + "/" + str(url) for url in profile["urls"]]
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
    def maybe_download_weights(path_weights: str) -> str:
        """
        Check if the path pointing to weight points to some registered weights. If yes, it will check if their weights
        must be downloaded. Only weights that have not the same expected size will be downloaded again.

        :param path_weights: A path to some model weights
        :return: Absolute path to model weights if model is registered
        """

        absolute_path = os.path.join(get_weights_dir_path(), path_weights)
        if ModelCatalog.is_registered(path_weights):
            profile = ModelCatalog.get_profile(path_weights)
            file_names = get_tp_weight_names(path_weights)
            for size, url, file_name in zip(profile["size"], profile["urls"], file_names):
                directory, _ = os.path.split(absolute_path)
                download(str(url), directory, file_name, int(size))
            return absolute_path

        logger.info("Will use not registered model. Make sure path to weights is correctly set")
        return absolute_path
