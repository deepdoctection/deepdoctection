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
Class AttrDict for maintaining configs and some functions for generating and saving AttrDict instances to .yaml files
"""

import pprint
from typing import Any, Dict, List

import yaml


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")
class AttrDict:
    """
    Class for storing key,values as instance with attributes and values.
    """

    _freezed = False

    # Avoid accidental creation of new hierarchies.

    def __getattr__(self, name: str) -> Any:
        """
        __getattr__
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
        __setattr__
        """
        if self._freezed and name not in self.__dict__:
            raise AttributeError(f"Config was freezed! Unknown config: {name}")
        super().__setattr__(name, value)

    def __str__(self) -> str:
        """
        __str__
        """
        return pprint.pformat(self.to_dict(), width=100, compact=True)

    __repr__ = __str__

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a nested dict."""
        return {
            k: v.to_dict() if isinstance(v, AttrDict) else v
            for k, v in self.__dict__.items()  # pylint: disable=C0103
            if not k.startswith("_")
        }

    def from_dict(self, d: Dict[str, Any]) -> None:  # pylint: disable=C0103
        """
        Generate an instance from a dict
        """
        if isinstance(d, dict):
            self.freeze(False)
            for k, v in d.items():  # pylint: disable=C0103
                self_v = getattr(self, k)
                if isinstance(v, dict):
                    self_v.from_dict(v)
                else:
                    setattr(self, k, v)

    def update_args(self, args: List[str]) -> None:
        """
        Update from command line args.
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

    def freeze(self, freezed: bool = True) -> None:
        """
        :param freezed: freeze the instance, so that no attributes can be added or changed
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


def set_config_by_yaml(path_yaml: str) -> AttrDict:
    """
    Use to initialize the config class for tensorpack faster rcnn

    :param path_yaml: The path to the file
    """
    config = AttrDict()
    _C = config  # pylint: disable=C0103
    _C.freeze(freezed=False)

    with open(path_yaml, "r") as file:  # pylint: disable=W1514
        _C.from_dict(yaml.load(file, Loader=yaml.Loader))

    _C.freeze()
    return config


def save_config_to_yaml(config: AttrDict, path_yaml: str) -> None:
    """
    :param config: The configuration instance as an AttrDict
    :param path_yaml: Save the config class for tensorpack faster rcnn
    :return: yaml_path: The path to save the file to
    """

    with open(path_yaml, "w") as file:  # pylint: disable=W1514
        yaml.dump(config.to_dict(), file)


def config_to_cli_str(config: AttrDict, *exclude: str) -> str:
    """
    Transform an AttrDict to a string that can be passed to a cli. Add optionally keys of the config that should not be
    added to the string.

    :param config: An :class:`AttrDict`
    :param exclude: keys of the AttrDict
    :return: A string that can be passed to a cli
    """

    config_dict = config.to_dict()
    for key in exclude:
        config_dict.pop(key)

    output_str = ""
    for key, val in config_dict.items():
        output_str += f"--{key} {val} "

    return output_str
