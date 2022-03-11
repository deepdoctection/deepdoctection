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
Typing for the whole package
"""

from typing import TYPE_CHECKING, Any, Callable, Dict, Protocol, Tuple, Type, Union

import numpy.typing as npt
from numpy import float32


# Type for a general dataclass
class IsDataclass(Protocol):  # pylint: disable=R0903
    """
    type hint for general dataclass
    """

    __dataclass_fields__: Dict[Any, Any]


# Type for category dict of the DatasetCategories class
KeyValue = Union[str, int]

# Numpy image type
ImageType = npt.NDArray[float32]

#
MapFunc = Callable[[Union[Any, Tuple[Any, Any]]], Any]
if TYPE_CHECKING:
    BaseExceptionType = Type[BaseException]
else:
    BaseExceptionType = bool

JsonDict = Dict[str, Any]

# Type for requirements. A requirement is a Tuple of string and a callable that returns True if the requirement is
# available
Requirement = Tuple[str, bool, str]
