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
from __future__ import annotations

import functools
import inspect
import os
from collections import defaultdict
from collections.abc import Mapping, MutableMapping
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


def as_string(value: Any) -> str:
    """
    Converts a leaf value into exactly one whitespace-normalized string.

    `None`, empty sequences and empty strings all map to the empty string. Nested token sequences are
    joined recursively with a single blank so that one field remains one unit, i.e.
    `["DE95", "5135"]` becomes `"DE95 5135"` and not two separate values.

    Args:
        value: A leaf value, i.e. a scalar, a string or an arbitrarily nested sequence of those.

    Returns:
        The string representation of the leaf.

    Example:
        ```python
        as_string(["DE95", ["5135", "0000"]])
        # Output: 'DE95 5135 0000'
        ```
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple)):
        return " ".join(part for part in map(as_string, value) if part)
    return str(value)


def is_leaf(value: Any) -> bool:
    """
    Checks if a `value` does not decompose into further records.

    A mapping is never a leaf. A sequence is a leaf if and only if it contains no mapping. A sequence
    holding at least one mapping is an array of records and has to be unfolded. Plain token sequences
    are leaves.

    Args:
        value: The value to check.

    Returns:
        True if the value is a leaf, False otherwise.

    Example:
        ```python
        is_leaf(["DE95", "5135"])
        # Output: True

        is_leaf([{"postingType": ["tax"]}])
        # Output: False
        ```
    """
    if isinstance(value, Mapping):
        return False
    if isinstance(value, (list, tuple)):
        return not any(isinstance(item, Mapping) for item in value)
    return True


def _flatten_walk(value: Any, path: str, flat: defaultdict[str, list[str]]) -> None:
    """
    Recursively collects the leaves of `value` into `flat`, keyed by their dot separated field path.

    Args:
        value: The current node of the record.
        path: The dot separated field path of the current node. Empty for the root node.
        flat: The accumulator that maps a field path to all leaf values found under it.
    """
    if isinstance(value, Mapping):
        for key, child in value.items():
            _flatten_walk(child, f"{path}.{key}" if path else str(key), flat)
    elif is_leaf(value):
        text = as_string(value)
        if text:
            flat[path].append(text)
    else:
        for item in value:
            _flatten_walk(item, path, flat)


def flatten(record: Any) -> dict[str, list[str]]:
    """
    Flattens a nested `record` into a mapping `{field_path: [values]}` with list indices discarded.

    Two records sharing the same field path end up as two entries of the same list. This is what makes
    a subsequent comparison independent of the ordering. Empty leaves are discarded so that
    "empty on both sides" does not count as a match.

    Args:
        record: An arbitrarily nested structure of mappings, sequences and scalars.

    Returns:
        A dictionary mapping the dot separated field path to the list of non-empty leaf values found
        under it.

    Example:
        ```python
        record = {
            "amount": ["197,66-"],
            "postings": [
                {"postingType": ["tax"], "postingAmount": ["31,55-"]},
                {"postingType": ["net"], "postingAmount": ["166,11-"]},
            ],
        }
        flatten(record)
        # Output: {'amount': ['197,66-'],
        #          'postings.postingType': ['tax', 'net'],
        #          'postings.postingAmount': ['31,55-', '166,11-']}
        ```
    """
    flat: defaultdict[str, list[str]] = defaultdict(list)
    _flatten_walk(record, "", flat)
    return dict(flat)


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
            cache = self._CALL_ONLY_ONCE_CACHE

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
