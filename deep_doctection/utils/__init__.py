# -*- coding: utf-8 -*-
# File: __init__.py

"""
Init file for utils package
"""

from .fs import is_file_extension, get_load_image_func, load_bytes_from_pdf_file, load_image_from_file
from .systools import sub_path, get_package_path, get_configs_dir_path, get_weights_dir_path
from .utils import delete_keys_from_dict, string_to_dict, split_string
from .identifier import is_uuid_like, get_uuid_from_str, get_uuid
from .metacfg import *
from .settings import *

__all__ = [
    "is_file_extension",
    "sub_path",
    "get_package_path",
    "get_configs_dir_path",
    "get_weights_dir_path",
    "delete_keys_from_dict",
    "is_uuid_like",
    "get_uuid_from_str",
    "get_uuid",
    "get_load_image_func",
    "load_bytes_from_pdf_file",
    "load_image_from_file",
    "AttrDict",
    "save_config_to_yaml",
    "set_config_by_yaml",
    "config_to_cli_str",
    "string_to_dict",
    "split_string",
    "names",
    "",
]
