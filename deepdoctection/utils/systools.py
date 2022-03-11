# -*- coding: utf-8 -*-
# File: systools.py

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
module for various basic functions that are needed everywhere
"""

import os

from .settings import CONFIGS, DATASET_DIR, MODEL_DIR, PATH

__all__ = ["sub_path", "get_package_path", "get_configs_dir_path", "get_weights_dir_path", "get_dataset_dir_path"]


def get_package_path() -> str:
    """
    :return: full base path of this package
    """
    return PATH


def get_weights_dir_path() -> str:
    """
    :return: full base path to the model dir
    """
    return MODEL_DIR


def get_configs_dir_path() -> str:
    """
    :return: full base path to the configs dir
    """
    return CONFIGS


def get_dataset_dir_path() -> str:
    """
    :return: full base path to the dataset dir
    """
    return DATASET_DIR


def sub_path(anchor_dir: str, *paths: str) -> str:
    """
    Generate a path from the anchor directory and various paths args.

    sub_path(/path/to,"dir1","dir2") will return /path/to/dir1/dir2

    :param anchor_dir: anchor directory
    :param paths: args of directories that should be added to path
    :return: sub_path
    """
    return os.path.join(os.path.dirname(os.path.abspath(anchor_dir)), *paths)
