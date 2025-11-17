# -*- coding: utf-8 -*-
# File: collector.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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


from typing import Iterator, Optional, TypeVar, Protocol, runtime_checkable, overload, List
from itertools import islice

T = TypeVar("T")

@runtime_checkable
class SupportsResetState(Protocol):
    def reset_state(self) -> None: ...

class ResettableIterator(SupportsResetState, Iterator[T], Protocol):
    pass

@overload
def collect_datapoint_from_dataflow(df: ResettableIterator[T], max_datapoints: Optional[int] = ...) -> List[T]: ...
@overload
def collect_datapoint_from_dataflow(df: Iterator[T], max_datapoints: Optional[int] = ...) -> List[T]: ...

def collect_datapoint_from_dataflow(
    df: Iterator[T],
    max_datapoints: Optional[int] = None
) -> List[T]:
    """
    Collect elements from a generator/iterator, optionally calling its reset_state method.
    :param df: A generator/iterator that may define reset_state\(\)
    :param max_datapoints: Maximum number of items to collect
    :return: A list of collected items
    """
    if isinstance(df, SupportsResetState):
        df.reset_state()

    return list(islice(df, max_datapoints)) if max_datapoints is not None else list(df)
