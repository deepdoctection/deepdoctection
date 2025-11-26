# -*- coding: utf-8 -*-
# File: custom.py

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
Some custom dataflow classes. Some ideas have been taken from

<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/common.py>
"""
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np

from ..utils.error import DataFlowResetStateNotCalledError
from ..utils.logger import LoggingRecord, logger
from ..utils.tqdm import get_tqdm
from ..utils.utils import get_rng
from .base import DataFlow, DataFlowReentrantGuard, ProxyDataFlow
from .serialize import DataFromIterable, DataFromList

__all__ = ["CacheData", "CustomDataFromList", "CustomDataFromIterable"]


class CacheData(ProxyDataFlow):
    """
    Completely cache the first pass of a DataFlow in memory,
    and produce from the cache thereafter.

    Note:
        The user should not stop the iterator before it has reached the end.
        Otherwise, the cache may be incomplete.

    Example:
        ```python
        df_list = CacheData(df).get_cache() # Buffers the whole dataflow and return a list of all datapoints
        ```

    """

    def __init__(self, df: DataFlow, shuffle: bool = False) -> None:
        """
        Args:
            df: input DataFlow.
            shuffle: whether to shuffle the cache before yielding from it.
        """
        self.shuffle = shuffle
        self.buffer: list[Any] = []
        self._guard: Optional[DataFlowReentrantGuard] = None
        self.rng = get_rng(self)
        super().__init__(df)

    def reset_state(self) -> None:
        super().reset_state()
        self._guard = DataFlowReentrantGuard()
        self.buffer = []

    def __iter__(self) -> Iterator[Any]:
        if self._guard is None:
            raise DataFlowResetStateNotCalledError()

        with self._guard:
            if self.buffer:
                if self.shuffle:
                    self.rng.shuffle(self.buffer)
                yield from self.buffer
            else:
                for dp in self.df:
                    yield dp
                    self.buffer.append(dp)

    def get_cache(self) -> list[Any]:
        """
        Get the cache of the whole dataflow as a list.

        Returns:
            list of datapoints
        """
        self.reset_state()
        with get_tqdm() as status_bar:
            for _ in self:
                status_bar.update()
            if self.shuffle:
                self.rng.shuffle(self.buffer)
            return self.buffer


class CustomDataFromList(DataFromList):
    """
    Wraps a list of datapoints to a dataflow. Compared to `Tensorpack.DataFlow.DataFromList`
    implementation you can specify a number of datapoints after that the iteration stops.
    You can also pass a re-balance function that filters on that list.

    Example:

        ```python
        def filter_first(lst):
            return lst.pop(0)

        df = CustomDataFromList(lst=[["a","b"],["c","d"]], rebalance_func=filter_first)
        df.reset_state()

        will yield:
            ["c","d"]
        ```

    """

    def __init__(
        self,
        lst: list[Any],
        shuffle: bool = False,
        max_datapoints: Optional[int] = None,
        rebalance_func: Optional[Callable[[list[Any]], list[Any]]] = None,
    ):
        """
        Args:
            lst: The input list. Each element represents a datapoint.
            shuffle: Whether to shuffle the list before streaming.
            max_datapoints: The maximum number of datapoints to return before stopping the iteration.
                            If None it streams the whole dataflow.
            rebalance_func: A func that inputs a list and outputs a list. Useful, if you want to filter the passed
                            list and re-balance the sample. Only the output list of the re-balancing function will be
                            considered.
        """
        super().__init__(lst, shuffle)
        self.max_datapoints = max_datapoints
        self.rebalance_func = rebalance_func

    def __len__(self) -> int:
        if self.max_datapoints is not None:
            return min(self.max_datapoints, len(self.lst))
        return len(self.lst)

    def __iter__(self) -> Iterator[Any]:
        if self.rng is None:
            raise DataFlowResetStateNotCalledError()
        if self.rebalance_func is not None:
            lst_tmp = self.rebalance_func(self.lst)
            logger.info(LoggingRecord(f"CustomDataFromList: subset size after re-balancing: {len(lst_tmp)}"))
        else:
            lst_tmp = self.lst

        if self.shuffle:
            idxs = np.arange(len(lst_tmp))
            self.rng.shuffle(idxs)
            for idx, k in enumerate(idxs):
                if self.max_datapoints is not None:
                    if idx < self.max_datapoints:
                        yield lst_tmp[k]
                    else:
                        break
                else:
                    yield lst_tmp[k]
        else:
            if self.max_datapoints is not None:
                for k, _ in enumerate(lst_tmp):
                    if k < self.max_datapoints:
                        yield lst_tmp[k]
                    else:
                        break
            else:
                yield from lst_tmp


class CustomDataFromIterable(DataFromIterable):
    """
    Wrap an iterable of datapoints to a dataflow. Can stop iteration after max_datapoints.
    """

    def __init__(self, iterable: Iterable[Any], max_datapoints: Optional[int] = None):
        """
        Args:
            iterable: An iterable object
            max_datapoints: The maximum number of datapoints to stream. If None it iterates through the whole
                            dataflow.
        """
        super().__init__(iterable)
        self.max_datapoints = max_datapoints
        if self.max_datapoints is not None:
            self._len = self.max_datapoints

    def __iter__(self) -> Any:
        if self.max_datapoints is not None:
            for idx, dp in enumerate(self._itr):
                if idx < self.max_datapoints:
                    yield dp
                else:
                    break
        else:
            yield from self._itr
