# -*- coding: utf-8 -*-
# File: detection_types.py

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

import queue
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Protocol, Tuple, Type, TypeVar, Union

import numpy.typing as npt
import tqdm
from numpy import uint8


# Type for a general dataclass
class IsDataclass(Protocol):  # pylint: disable=R0903
    """
    type hint for general dataclass
    """

    __dataclass_fields__: Dict[Any, Any]


# Type for category dict of the DatasetCategories class
KeyValue = Union[str, int]

# Numpy image type
ImageType = npt.NDArray[uint8]

# typing for curry decorator
DP = TypeVar("DP")
S = TypeVar("S")
T = TypeVar("T")

# Some type hints that must be distinguished when runnning mypy and linters
if TYPE_CHECKING:
    QueueType = queue.Queue[Any]  # pylint: disable=E1136
    TqdmType = tqdm.tqdm[Any]  # pylint: disable=E1136
    BaseExceptionType = Type[BaseException]
else:
    BaseExceptionType = bool
    QueueType = queue.Queue
    TqdmType = tqdm.tqdm

JsonDict = Dict[str, Any]

# Type for requirements. A requirement is a Tuple of string and a callable that returns True if the requirement is
# available
Requirement = Tuple[str, bool, str]

# Pathlike, use this typing for everything where a path (absolute/relative) is involved
Pathlike = Union[str, Path]
