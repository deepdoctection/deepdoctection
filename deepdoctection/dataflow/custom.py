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
Adding some functionality to dataflow classes (e.g. monkey patching, inheritance ...)
"""
from typing import Any, Callable, Iterable, List, Optional

import numpy as np
from dataflow import CacheData, DataFromIterable, DataFromList  # type: ignore

from ..utils.logger import logger

__all__ = ["CacheData", "CustomDataFromList", "CustomDataFromIterable"]


def _get_cache(self: CacheData) -> List[Any]:
    """
    get the cache of the whole dataflow as a list

    :return: list of datapoints
    """
    self.reset_state()
    for _ in self:
        pass
    return self.buffer


CacheData.get_cache = _get_cache


class CustomDataFromList(DataFromList):  # type: ignore
    """
    Wraps a list of datapoints to a dataflow. Compared to :class:`Tensorpack.DataFlow.DataFromList` implementation you
    can specify a number of datapoints after that the iteration stops. You can also pass a rebalance function that
    filters on that list.
    """

    def __init__(
        self,
        lst: List[Any],
        shuffle: Optional[bool] = False,
        max_datapoints: Optional[int] = None,
        rebalance_func: Optional[Callable[[List[Any]], List[Any]]] = None,
    ):
        """
        :param lst: the input list. Each element represents a datapoint.
        :param shuffle: Whether to shuffle the list before streaming.
        :param max_datapoints: The maximum number of datapoints to return before stopping the iteration.
                               If None it streams the whole dataflow.
        :param rebalance_func: A func that inputs a list and outputs a list. Useful, if you want to filter the passed
                               list and re-balance the sample. Only the output list of the re-balancing function will be
                               considered.
        """
        if shuffle:
            logger.info("Make sure to call .reset_state() for the dataflow otherwise an error will be raised")
        super().__init__(lst, shuffle)
        self.max_datapoints = max_datapoints
        self.rebalance_func = rebalance_func

    def __len__(self) -> int:
        if self.max_datapoints is not None:
            return min(self.max_datapoints, len(self.lst))
        return len(self.lst)

    def __iter__(self) -> Any:
        if self.rebalance_func is not None:
            lst_tmp = self.rebalance_func(self.lst)
            logger.info("subset size after re-balancing: %s", len(lst_tmp))
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


class CustomDataFromIterable(DataFromIterable):  # type: ignore
    """
    Wrap an iterable of datapoints to a dataflow. Can stop iteration after max_datapoints.
    """

    def __init__(self, iterable: Iterable[Any], max_datapoints: Optional[int] = None):
        """
        :param iterable: An iterable object
        :param max_datapoints: The maximum number of datapoints to stream. If None it iterates through the whole
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
