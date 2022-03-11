# -*- coding: utf-8 -*-
# File: maputils.py

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
Utility functions, only related to builtin objects
"""

from collections.abc import MutableMapping
from typing import Any, Dict, List, Set, Union


def delete_keys_from_dict(
    dictionary: Union[Dict[Any, Any], MutableMapping], keys: Union[str, List[str], Set[str]]  # type: ignore
) -> Dict[Any, Any]:
    """
    Removing key/value pairs from dictionary. Works for nested dicts as well.

    :param dictionary: A input dictionary
    :param keys: A single or list of keys
    :return: The modified dictionary with listed keys removed
    """

    if isinstance(keys, str):
        keys = [keys]
    keys_set = set(keys)

    modified_dict = {}
    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)
            elif isinstance(value, list):
                modified_dict[key] = []  # type: ignore
                for el in value:
                    if isinstance(el, MutableMapping):
                        modified_dict[key].append(delete_keys_from_dict(el, keys_set))  # type: ignore
                    else:
                        modified_dict[key].append(el)  # type: ignore
            else:
                modified_dict[key] = value
    return modified_dict


def split_string(input_string: str) -> List[str]:
    """
    Takes a string, splits between commas and returns a list with split components as list elements

    :param input_string: input
    """
    return input_string.split(",")


def string_to_dict(input_string: str) -> Dict[str, str]:
    """
    Takes a string of a form 'key1=val1,key2=val2' and returns the corresponding dict
    """
    items_list = input_string.split(",")
    output_dict = {}
    for pair in items_list:
        pair = pair.split("=")  # type: ignore
        output_dict[pair[0]] = pair[1]
    return output_dict


def to_bool(inputs: Union[str, bool, int]) -> bool:
    """
    Convert a string "True" or "False" to its boolean value
    :param inputs: Input string
    :return: boolean value
    """
    if isinstance(inputs, bool):
        return inputs
    if inputs == "False":
        return False
    return True
