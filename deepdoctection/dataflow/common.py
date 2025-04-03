# -*- coding: utf-8 -*-
# File: common.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")


"""
Some DataFlow classes for transforming and processing datapoints. Many classes have been taken from

<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/common.py>
"""
import itertools
from copy import copy
from typing import Any, Callable, Iterator, Union

import tqdm

from ..utils.tqdm import get_tqdm, get_tqdm_default_kwargs
from .base import DataFlow, ProxyDataFlow


class TestDataSpeed(ProxyDataFlow):
    """Test the speed of a DataFlow"""

    def __init__(self, df: DataFlow, size: int = 5000, warmup: int = 0) -> None:
        """
        :param df: the DataFlow to test.
        :param size: number of datapoints to fetch.
        :param warmup: warmup iterations
        """
        super().__init__(df)
        self.test_size = int(size)
        self.warmup = int(warmup)
        self._reset_called = False

    def reset_state(self) -> None:
        self._reset_called = True
        super().reset_state()

    def __iter__(self) -> Iterator[Any]:
        """Will run testing at the beginning, then produce data normally."""
        self.start()
        yield from self.df

    def start(self) -> None:
        """
        Start testing with a progress bar.
        """
        if not self._reset_called:
            self.df.reset_state()
        itr = iter(self.df)
        if self.warmup:
            for _ in tqdm.trange(self.warmup, **get_tqdm_default_kwargs()):  # type: ignore
                next(itr)
        # add smoothing for speed benchmark
        with get_tqdm(total=self.test_size, leave=True, smoothing=0.2) as pbar:
            for idx, _ in enumerate(itr):
                pbar.update()
                if idx == self.test_size - 1:
                    break


class FlattenData(ProxyDataFlow):
    """
    Flatten an iterator within a datapoint. Will flatten the datapoint if it is a list or a tuple.

    **Example:**

            dp_1 = ['a','b']
            dp_2 = ['c','d']

        will yield

            ['a'], ['b'], ['c'], ['d'].
    """

    def __iter__(self) -> Any:
        for dp in self.df:
            if isinstance(dp, (list, tuple)):
                for dpp in dp:
                    yield [dpp] if isinstance(dp, list) else tuple(dpp)


class MapData(ProxyDataFlow):
    """
    Apply a mapper/filter on the datapoints of a DataFlow.
    Note:
        1. Please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, `len(MapData(ds))` will be incorrect.

    **Example:**

            df = ... # some dataflow each datapoint is [img, label]
            ds = MapData(ds, lambda dp: [dp[0] * 255, dp[1]])
    """

    def __init__(self, df: DataFlow, func: Callable[[Any], Any]) -> None:
        """
        :param df: input DataFlow
        :param func: takes a datapoint and returns a new
               datapoint. Return None to discard/skip this datapoint.
        """
        super().__init__(df)
        self.func = func

    def __iter__(self) -> Iterator[Any]:
        for dp in self.df:
            ret = self.func(copy(dp))  # shallow copy the list
            if ret is not None:
                yield ret


class MapDataComponent(MapData):
    """
    Apply a mapper/filter on a datapoint component.

    Note:
        1. This dataflow itself doesn't modify the datapoints.
           But please make sure func doesn't modify its arguments in place,
           unless you're certain it's safe.
        2. If you discard some datapoints, ``len(MapDataComponent(ds, ..))`` will be incorrect.


    **Example:**

            df = ... # some dataflow each datapoint is [img, label]
            ds = MapDataComponent(ds, lambda img: img * 255, 0)  # map the 0th component
    """

    def __init__(self, df: DataFlow, func: Callable[[Any], Any], index: Union[int, str] = 0) -> None:
        """
        :param df: input DataFlow which produces either list or dict.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                Return None to discard/skip this datapoint.
        :param index: index or key of the component.
        """
        self._index = index
        self._func = func
        super().__init__(df, self._mapper)

    def _mapper(self, dp: Any) -> Any:
        ret = self._func(dp[self._index])
        if ret is None:
            return None
        dp = copy(dp)  # shallow copy to avoid modifying the datapoint
        if isinstance(dp, tuple):
            dp = list(dp)  # to be able to modify it in the next line
        dp[self._index] = ret
        return dp


class RepeatedData(ProxyDataFlow):
    """Take data points from another DataFlow and produce them until
    it's exhausted for certain amount of times. i.e.:
    `dp1`, `dp2`, .... `dpn`, `dp1`, `dp2`, ....`dpn`.
    """

    def __init__(self, df: DataFlow, num: int) -> None:
        """
        :param df: input DataFlow
        :param num: number of times to repeat ds.
                Set to -1 to repeat ``ds`` infinite times.
        """
        self.num = num
        if self.num != -1:
            self.dfs = itertools.tee(df, self.num)
        else:
            self.dfs = ()
        super().__init__(df)

    def __len__(self) -> int:
        """
        Raises:
            `ValueError` when num == -1.
        """
        if self.num == -1:
            raise NotImplementedError("__len__() is unavailable for infinite dataflow")
        return len(self.df) * self.num

    def __iter__(self) -> Iterator[Any]:
        if self.num == -1:
            while True:
                yield from self.df
        else:
            for df in self.dfs:
                yield from df


class ConcatData(DataFlow):
    """
    Concatenate several DataFlow.
    Produce datapoints from each DataFlow and start the next when one
    DataFlow is exhausted. Use this dataflow to process several .pdf in one step.

    **Example:**

           df_1 = analyzer.analyze(path=path/to/pdf_1.pdf")
           df_2 = analyzer.analyze(path=path/to/pdf_2.pdf")
           df = ConcatData([df_1,df_2])
    """

    def __init__(self, df_lists: list[DataFlow]) -> None:
        """
        :param df_lists: a list of DataFlow.
        """
        self.df_lists = df_lists

    def reset_state(self) -> None:
        for df in self.df_lists:
            df.reset_state()

    def __len__(self) -> int:
        return sum(len(x) for x in self.df_lists)

    def __iter__(self) -> Iterator[Any]:
        for df in self.df_lists:
            yield from df


class JoinData(DataFlow):
    """
    Join the components from each DataFlow. See below for its behavior.
    Note that you can't join a DataFlow that produces lists with one that produces dicts.

    **Example:**

        df1 produces: [[c1], [c2]]
        df2 produces: [[c3], [c4]]
        joined: [[c1, c3], [c2, c4]]

        df1 produces: {"a":c1, "b":c2}
        df2 produces: {"c":c3}
        joined: {"a":c1, "b":c2, "c":c3}

    `JoinData` will stop once the first Dataflow throws a StopIteration
    """

    def __init__(self, df_lists: list[DataFlow]) -> None:
        """
        :param df_lists: a list of DataFlow. When these dataflows have different sizes, JoinData will stop when any
                        of them is exhausted.
                        The list could contain the same DataFlow instance more than once,
                        but note that in that case `__iter__` will then also be called many times.
        """
        self.df_lists = df_lists

    def reset_state(self) -> None:
        for df in set(self.df_lists):
            df.reset_state()

    def __len__(self) -> int:
        """
        Return the minimum size among all.
        """
        return min(len(k) for k in self.df_lists)

    def __iter__(self) -> Iterator[Any]:
        itrs = [k.__iter__() for k in self.df_lists]
        try:
            while True:
                all_dps = [next(itr) for itr in itrs]
                dp: Any
                if isinstance(all_dps[0], (list, tuple)):
                    dp = list(itertools.chain(*all_dps))
                else:
                    dp = {}
                    for x in all_dps:
                        dp.update(x)
                yield dp
        except StopIteration:  # some of them are exhausted
            pass


class BatchData(ProxyDataFlow):
    """
    Stack datapoints into batches. It produces datapoints of the same number of components as `df`, but
    each datapoint is now a list of datapoints.
    """

    def __init__(self, df: DataFlow, batch_size: int, remainder: bool = False) -> None:
        """
        :param df: A dataflow
        :param batch_size: batch size
        :param remainder: When the remaining datapoints in ``df`` is not enough to form a batch, whether or not to
                          also produce the remaining data as a smaller batch.
                          If set to `False`, all produced datapoints are guaranteed to have the same batch size.
                          If set to `True`, `len(ds)` must be accurate.
        """
        super().__init__(df)
        if not remainder:
            if batch_size > len(df):
                raise ValueError("Dataflow must be larger than batch_size")
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.remainder = remainder

    def __len__(self) -> int:
        df_size = len(self.df)
        div = df_size // self.batch_size
        rem = df_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def __iter__(self) -> Iterator[Any]:
        holder = []
        for data in self.df:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield holder
                holder = []
        if self.remainder and len(holder) > 0:
            yield holder
