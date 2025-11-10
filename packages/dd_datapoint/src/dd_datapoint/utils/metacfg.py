# -*- coding: utf-8 -*-
# File: metacfg.py

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
Class `AttrDict` for maintaining configs and some functions for generating and saving `AttrDict` instances to
`.yaml` files
"""
from __future__ import annotations

import pprint
from typing import Any

import yaml

from .types import PathLikeOrStr


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")
class AttrDict:
    """
    Class `AttrDict` for maintaining configs and some functions for generating and saving `AttrDict` instances to
    `.yaml` files.

    Info:
        This module provides a class for storing key-value pairs as attributes and functions for serializing and
        deserializing configurations.
    """

    _freezed = False

    # Avoid accidental creation of new hierarchies.

    def __getattr__(self, name: str) -> Any:
        """
        Returns the attribute value for `name`. If the attribute does not exist and the instance is not frozen, a new
        `AttrDict` is created and assigned.

        Args:
            name: The name of the attribute.

        Returns:
            The value of the attribute.

        Raises:
            AttributeError: If the instance is frozen or the attribute name starts with `_`.
        """
        if self._freezed:
            raise AttributeError(name)
        if name.startswith("_"):
            # Do not mess with internals. Otherwise, copy/pickle will fail
            raise AttributeError(name)
        ret = AttrDict()
        setattr(self, name, ret)
        return ret

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute `name` to `value`.

        Args:
            name: The name of the attribute.
            value: The value to set.

        Raises:
            AttributeError: If the instance is frozen and `name` is not `_freezed`.
        """
        if self._freezed and name != "_freezed":
            raise AttributeError(f"Config was freezed! Unknown config: {name}")
        super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        Returns a pretty-printed string representation of the configuration.

        Returns:
            A string representation of the configuration.
        """
        return pprint.pformat(self.to_dict(), width=100, compact=True)

    __repr__ = __str__

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to a nested dict.

        Returns:
            A dictionary representation of the configuration.
        """
        return {
            k: v.to_dict() if isinstance(v, AttrDict) else v for k, v in self.__dict__.items() if not k.startswith("_")
        }

    def from_dict(self, d: dict[str, Any]) -> None:  # pylint: disable=C0103
        """
        Generate an instance from a dict.

        Args:
            d: The dictionary to load values from.
        """
        if isinstance(d, dict):
            self.freeze(False)
            for k, v in d.items():  # pylint: disable=C0103
                self_v = getattr(self, k)
                if isinstance(v, dict):
                    self_v.from_dict(v)
                else:
                    setattr(self, k, v)

    def update_args(self, args: list[str]) -> None:
        """
        Update from command line args.

        Args:
            args: A list of command line arguments in the form `key1.key2=val`.
        """
        for cfg in args:
            keys, v = cfg.split("=", maxsplit=1)  # pylint: disable=C0103
            key_list = keys.split(".")

            dic = self
            for _, k in enumerate(key_list[:-1]):
                assert k in dir(dic), f"Unknown config key: {keys}"
                dic = getattr(dic, k)
            key = key_list[-1]

            old_v = getattr(dic, key)
            if not isinstance(old_v, str):
                v = eval(v)  # pylint: disable=C0103, W0123
            setattr(dic, key, v)

    def overwrite_config(self, other_config: AttrDict) -> None:
        """
        Overwrite the current config with values from another config.

        Args:
            other_config: The other `AttrDict` instance to copy values from.

        Raises:
            AttributeError: If the config is frozen.
        """
        if self._freezed:
            raise AttributeError("Config was freezed! Cannot overwrite config.")
        self.from_dict(other_config.to_dict())

    def freeze(self, freezed: bool = True) -> None:
        """
        Freeze or unfreeze the instance, so that no attributes can be added or changed.

        Args:
            freezed: Whether to freeze the instance.
        """
        self._freezed = freezed
        for v in self.__dict__.values():  # pylint: disable=C0103
            if isinstance(v, AttrDict):
                v.freeze(freezed)

    # avoid silent bugs
    def __eq__(self, _: Any) -> bool:
        raise NotImplementedError()

    def __ne__(self, _: Any) -> bool:
        raise NotImplementedError()


def set_config_by_yaml(path_yaml: PathLikeOrStr) -> AttrDict:
    """
    Initialize the config class from a YAML file.

    Args:
        path_yaml: The path to the YAML file.

    Returns:
        An `AttrDict` instance initialized from the YAML file.
    """
    config = AttrDict()
    _C = config  # pylint: disable=C0103
    _C.freeze(freezed=False)

    with open(path_yaml, "r") as file:  # pylint: disable=W1514
        _C.from_dict(yaml.load(file, Loader=yaml.Loader))

    _C.freeze()
    return config


def save_config_to_yaml(config: AttrDict, path_yaml: PathLikeOrStr) -> None:
    """
    Save the configuration instance as a YAML file.

    Example:
        ```python
        save_config_to_yaml(config, "config.yaml")
        ```

    Args:
        config: The configuration instance as an `AttrDict`.
        path_yaml: The path to save the YAML file to.

    """

    with open(path_yaml, "w") as file:  # pylint: disable=W1514
        yaml.dump(config.to_dict(), file)


def config_to_cli_str(config: AttrDict, *exclude: str) -> str:
    """
    Transform an `AttrDict` to a string that can be passed to a CLI. Optionally exclude keys from the string.

    Example:
        ```python
        config_to_cli_str(config, "key1", "key2")
        ```

    Args:
        config: An `AttrDict`.
        *exclude: Keys of the `AttrDict` to exclude.

    Returns:
        A string that can be passed to a CLI.
    """

    config_dict = config.to_dict()
    for key in exclude:
        config_dict.pop(key)

    output_str = ""
    for key, val in config_dict.items():
        output_str += f"--{key} {val} "

    return output_str
