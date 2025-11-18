# -*- coding: utf-8 -*-
# File: serialize.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Some DataFlow classes for serialization. Many classes have been taken from

<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/raw.py>
"""

import pickle
from copy import copy
from typing import Any, Iterable, Iterator, Optional, Union

import numpy as np

from ..utils.error import DataFlowResetStateNotCalledError
from .base import DataFlow, RNGDataFlow


class DataFromList(RNGDataFlow):
    """Wrap a list of datapoints to a DataFlow"""

    def __init__(self, lst: list[Any], shuffle: bool = True) -> None:
        """
        Args:
            lst: input list. Each element is a datapoint.
            shuffle: shuffle data.
        """
        super().__init__()
        self.lst = lst
        self.shuffle = shuffle

    def __len__(self) -> int:
        return len(self.lst)

    def __iter__(self) -> Iterator[Any]:
        if not self.shuffle:
            yield from self.lst
        else:
            idxs = np.arange(len(self.lst))
            if self.rng is not None:
                self.rng.shuffle(idxs)
                for k in idxs:
                    yield self.lst[k]
            else:
                raise DataFlowResetStateNotCalledError()


class DataFromIterable(DataFlow):
    """Wrap an iterable of datapoints to a DataFlow"""

    def __init__(self, iterable: Iterable[Any]) -> None:
        """
        Args:
            iterable: an iterable object
        """
        self._itr = iterable
        self._len: Optional[int] = None
        try:
            self._len = len(iterable)  # type: ignore
        except (NotImplementedError, TypeError):
            pass

    def __len__(self) -> int:
        if self._len is None:
            raise NotImplementedError()
        return self._len

    def __iter__(self) -> Iterator[Any]:
        yield from self._itr

    def reset_state(self) -> None:
        pass


class FakeData(RNGDataFlow):
    """Generate fake data of given shapes"""

    def __init__(
        self,
        shapes: list[Union[list[Any], tuple[Any]]],
        size: int = 1000,
        random: bool = True,
        dtype: str = "float32",
        domain: tuple[Union[float, int], Union[float, int]] = (0, 1),
    ):
        """
        Args:
            shapes: a list of lists/tuples. Shapes of each component.
            size: size of this DataFlow.
            random: whether to randomly generate data every iteration.
                        Note that merely generating the data could sometimes be time-consuming!
            dtype: data type as string, or a list of data types.
            domain: (min, max) tuple, or a list of such tuples
        """

        super().__init__()
        self.shapes = shapes
        self._size = int(size)
        self.random = random
        self.dtype = [dtype] * len(shapes) if isinstance(dtype, str) else dtype
        self.domain = [domain] * len(shapes) if isinstance(domain, tuple) else domain
        if len(self.dtype) != len(self.shapes):
            raise ValueError(f"self.dtype={self.dtype} and self.shapes={self.shapes} must have same length")

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        if self.rng is None:
            raise DataFlowResetStateNotCalledError()
        if self.random:
            for _ in range(self._size):
                val = []
                for idx, _ in enumerate(self.shapes):
                    var = (
                        self.rng.rand(*self.shapes[idx]) * (self.domain[idx][1] - self.domain[idx][0])
                        + self.domain[idx][0]
                    )
                    val.append(var.astype(self.dtype[idx]))
                yield val
        else:
            val = []
            for idx, _ in enumerate(self.shapes):
                var = (
                    self.rng.rand(*self.shapes[idx]) * (self.domain[idx][1] - self.domain[idx][0]) + self.domain[idx][0]
                )
                val.append(var.astype(self.dtype[idx]))
            for _ in range(self._size):
                yield copy(val)


class PickleSerializer:
    """A Serializer to load and to dump objects"""

    @staticmethod
    def dumps(obj: Any) -> bytes:
        """
        Args:
            obj: bytes
        """
        return pickle.dumps(obj, protocol=-1)

    @staticmethod
    def loads(buf: Any) -> Any:
        """
        Args:
            buf: bytes
        """
        return pickle.loads(buf)
