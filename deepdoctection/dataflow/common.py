# -*- coding: utf-8 -*-
# File: common.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")


"""
Some DataFlows  for transforming and processing datapoints
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
        Args:
            df: The DataFlow to test.
            size: Number of datapoints to fetch.
            warmup: Warmup iterations.
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
    FlattenData flattens an iterator within a datapoint. Will flatten the datapoint if it is a list or a tuple.

    Example:
        ```python
        dp_1 = ['a','b']
        dp_2 = ['c','d']

        yields:
            ['a'], ['b'], ['c'], ['d']
        ```
    """

    def __iter__(self) -> Any:
        for dp in self.df:
            if isinstance(dp, (list, tuple)):
                for dpp in dp:
                    yield [dpp] if isinstance(dp, list) else tuple(dpp)


class MapData(ProxyDataFlow):
    """
    MapData applies a mapper/filter on the datapoints of a DataFlow.

    Notes:
        1. Please ensure that `func` does not modify its arguments in-place unless it is safe.
        2. If some datapoints are discarded, `len(MapData(ds))` will be incorrect.

    Example:
        ```python
        df = ... # a DataFlow where each datapoint is [img, label]
        ds = MapData(ds, lambda dp: [dp[0] * 255, dp[1]])
        ```
    """

    def __init__(self, df: DataFlow, func: Callable[[Any], Any]) -> None:
        """
        Args:
            df: input DataFlow
            func: takes a datapoint and returns a new
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
    MapDataComponent applies a mapper/filter on a component of a datapoint.

    Notes:
        1. This DataFlow itself does not modify the datapoints. Please ensure that `func` does not modify its arguments
           in-place unless it is safe.
        2. If some datapoints are discarded, `len(MapDataComponent(ds, ..))` will be incorrect.

    Example:
        ```python
        df = ... # a DataFlow where each datapoint is [img, label]
        ds = MapDataComponent(ds, lambda img: img * 255, 0)  # maps the 0th component
        ```
    """

    def __init__(self, df: DataFlow, func: Callable[[Any], Any], index: Union[int, str] = 0) -> None:
        """
        Args:
            df: input DataFlow which produces either list or dict.
                func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value for ``dp[index]``.
                Return None to discard/skip this datapoint.
            index: index or key of the component.
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
    """
    RepeatedData takes datapoints from another DataFlow and produces them until they are exhausted for a certain number
    of repetitions.

    Example:
        ```python
        dp1, dp2, .... dpn, dp1, dp2, ....dpn
        ```
    """

    def __init__(self, df: DataFlow, num: int) -> None:
        """
        Args:
            df: Input DataFlow.
            num: Number of repetitions of the DataFlow. Set `-1` to repeat the DataFlow infinitely.
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
            ValueError: when num == -1.
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
    ConcatData concatenates multiple DataFlows. Produces datapoints from each DataFlow and starts the next when one
    DataFlow is exhausted. Use this DataFlow to process multiple .pdf files in one step.

    Example:
        ```python
        df_1 = analyzer.analyze(path="path/to/pdf_1.pdf")
        df_2 = analyzer.analyze(path="path/to/pdf_2.pdf")
        df = ConcatData([df_1, df_2])
        ```


    """

    def __init__(self, df_lists: list[DataFlow]) -> None:
        """
        Args:
            df_lists: A list of DataFlows.
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
    JoinData joins the components from each DataFlow. See below for its behavior. It is not possible to join a DataFlow
    that produces lists with one that produces dictionaries.

    Example:
        ```python
        df1 produces: [[c1], [c2]]
        df2 produces: [[c3], [c4]]
        joined: [[c1, c3], [c2, c4]]

        df1 produces: {"a": c1, "b": c2}
        df2 produces: {"c": c3}
        joined: {"a": c1, "b": c2, "c": c3}
        ```

    `JoinData` stops once the first DataFlow raises a `StopIteration`.


    """

    def __init__(self, df_lists: list[DataFlow]) -> None:
        """
        Args:
            df_lists: A list of DataFlows. If these DataFlows have different sizes, `JoinData` stops when one of them is
                      exhausted. The list can contain the same DataFlow instance multiple times, but note that in this
                      case `__iter__` will also be called multiple times.
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
    BatchData stacks datapoints into batches. It produces datapoints with the same number of components as `df`, but
    each datapoint is now a list of datapoints.

    Example:
        ```python
        df produces: [[c1], [c2], [c3], [c4]]
        batch_size = 2
        yields: [[c1, c2], [c3, c4]]
        ```

    """

    def __init__(self, df: DataFlow, batch_size: int, remainder: bool = False) -> None:
        """
        Args:
            df: A DataFlow.
            batch_size: Batch size.
            remainder: If the remaining datapoints in `df` are not enough to form a batch, whether to produce the
                       remaining data as a smaller batch. If set to `False`, all produced datapoints are guaranteed to
                       have the same batch size. If set to `True`, `len(ds)` must be accurate.
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
