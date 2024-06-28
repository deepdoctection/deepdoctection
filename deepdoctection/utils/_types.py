# -*- coding: utf-8 -*-
# File: _types.py

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
Typing sheet for the whole package
"""

import os
import queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Type, TypeVar, Union

import numpy.typing as npt
import tqdm
from numpy import uint8
from typing_extensions import TypeAlias


# Type for a general dataclass
class IsDataclass(Protocol):  # pylint: disable=R0903
    """
    type hint for general dataclass
    """

    __dataclass_fields__: dict[Any, Any]


# Numpy image type
PixelValues = npt.NDArray[uint8]
# b64 encoded image as string
B64Str: TypeAlias = str
# b64 encoded image in bytes
B64: TypeAlias = bytes

# Typing for curry decorator
DP = TypeVar("DP")
S = TypeVar("S")
T = TypeVar("T")

# Some type hints that must be distinguished when running mypy and linters
if TYPE_CHECKING:
    QueueType = queue.Queue[Any]  # pylint: disable=E1136
    TqdmType = tqdm.tqdm[Any]  # pylint: disable=E1136
    BaseExceptionType = Type[BaseException]

else:
    BaseExceptionType = bool
    QueueType = queue.Queue
    TqdmType = tqdm.tqdm

# A dict converted from a generic JSON object
JsonDict = dict[str, Any]

# A path to a file, directory etc. can be given as a string or Path object
StrOrPathLike: TypeAlias = Union[str, os.PathLike]

# mainly used in utils
# Type for requirements. A requirement is a Tuple of string and a callable that returns True if the requirement is
# available
PackageAvailable: TypeAlias = bool
ErrorMsg: TypeAlias = str
Requirement = tuple[str, PackageAvailable, ErrorMsg]

BGR: TypeAlias = tuple[int, int, int]

# A type to collect key val pairs of environ information. Mainly used in env_info.py
KeyValEnvInfos: TypeAlias = list[tuple[str, str]]
#Pathlike = Union[str, Path]
