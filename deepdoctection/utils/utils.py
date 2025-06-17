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
import functools
import inspect
import os
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any, Callable, Sequence, Union

import numpy as np

from .types import PathLikeOrStr


def delete_keys_from_dict(
    dictionary: Union[dict[Any, Any], MutableMapping], keys: Union[str, list[str], set[str]]  # type: ignore
) -> dict[Any, Any]:
    """
    Removes key/value pairs from a `dictionary`. Works for nested dictionaries as well.

    Args:
        dictionary: An input dictionary.
        keys: A single key or a list of keys.

    Returns:
        The modified dictionary with the specified keys removed.
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


def split_string(input_string: str) -> list[str]:
    """
    Splits an `input_string` by commas and returns a list of the split components.

    Args:
        input_string: The input string.

    Returns:
        A list of string components.
    """
    return input_string.split(",")


def string_to_dict(input_string: str) -> dict[str, str]:
    """
    Converts an `input_string` of the form `key1=val1,key2=val2` into a dictionary.

    Args:
        input_string: The input string.

    Returns:
        The corresponding dictionary.
    """
    items_list = input_string.split(",")
    output_dict = {}
    for pair in items_list:
        pair = pair.split("=")  # type: ignore
        output_dict[pair[0]] = pair[1]
    return output_dict


def to_bool(inputs: Union[str, bool, int]) -> bool:
    """
    Converts a string "True" or "False" to its boolean value.

    Args:
        inputs: Input string, boolean, or integer.

    Returns:
        The boolean value.
    """
    if isinstance(inputs, bool):
        return inputs
    if inputs == "False":
        return False
    return True


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")


def call_only_once(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorates a method or property of a class so that it can only be called once for every instance.
    Calling it more than once will result in an exception.

    Args:
        func: The method or property to decorate.

    Returns:
        The decorated function.

    Note:
        Use `call_only_once` only on methods or properties.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore
        self = args[0]
        assert func.__name__ in dir(self), "call_only_once can only be used on method or property!"

        if not hasattr(self, "_CALL_ONLY_ONCE_CACHE"):
            cache = self._CALL_ONLY_ONCE_CACHE = set()
        else:
            cache = self._CALL_ONLY_ONCE_CACHE  # pylint: disable=W0212

        cls = type(self)
        # cannot use ismethod(), because decorated method becomes a function
        is_method = inspect.isfunction(getattr(cls, func.__name__))
        assert func not in cache, (
            f"{'Method' if is_method else 'Property'} {cls.__name__}.{func.__name__} "
            f"can only be called once per object!"
        )
        cache.add(func)

        return func(*args, **kwargs)

    return wrapper


# taken from https://github.com/tensorpack/dataflow/blob/master/dataflow/utils/utils.py
def get_rng(obj: Any = None) -> np.random.RandomState:
    """
    Gets a good random number generator seeded with time, process id, and the object.

    Args:
        obj: Some object to use to generate the random seed.

    Returns:
        The random number generator.
    """
    seed = (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)


def is_file_extension(file_name: PathLikeOrStr, extension: Union[str, Sequence[str]]) -> bool:
    """
    Checks if a given `file_name` has a given `extension`.

    Args:
        file_name: The file name, either full path or standalone.
        extension: The extension of the file. Must include a dot (e.g., `.txt`).

    Returns:
        True if the file has the given extension, False otherwise.
    """
    if isinstance(extension, str):
        return os.path.splitext(file_name)[-1].lower() == extension
    return os.path.splitext(file_name)[-1].lower() in extension


def partition_list(base_list: list[str], stop_value: str) -> list[list[str]]:
    """
    Partitions a list of strings into sublists, where each sublist starts with the first occurrence of the `stop_value`.
    Consecutive `stop_value` elements are grouped together in the same sublist.

    Args:
        base_list: The list of strings to be partitioned.
        stop_value: The string value that indicates the start of a new partition.

    Returns:
        A list of lists, where each sublist is a partition of the original list.

    Example:
        ```python
        strings = ['a', 'a', 'c', 'c', 'b', 'd', 'c', 'c', 'a', 'b', 'a', 'b', 'a', 'a']
        stop_string = 'a'
        partition_list(strings, stop_string)
        # Output: [['a', 'a', 'c', 'c', 'b', 'd', 'c', 'c'], ['a', 'b'], ['a', 'b'], ['a', 'a']]
        ```
    """

    partitions = []
    current_partition: list[str] = []
    stop_found = False

    for s in base_list:
        if s == stop_value:
            if not stop_found and current_partition:
                partitions.append(current_partition)
                current_partition = []
            current_partition.append(s)
            stop_found = True
        else:
            current_partition.append(s)
            stop_found = False

    if current_partition:
        partitions.append(current_partition)

    return partitions
