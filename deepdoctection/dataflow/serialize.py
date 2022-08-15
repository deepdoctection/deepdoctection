# -*- coding: utf-8 -*-
# File: serialize.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Some DataFlow classes for serialization. Many classes have been taken from

- https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/raw.py
"""

import pickle
import copy
from typing import Any, Iterable, List

import numpy as np

from .base import DataFlow, RNGDataFlow


class DataFromList(RNGDataFlow):
    """Wrap a list of datapoints to a DataFlow"""

    def __init__(self, lst: List[Any], shuffle: bool = True):
        """
        :param lst: input list. Each element is a datapoint.
        :param shuffle: shuffle data.
        """
        super().__init__()
        self.lst = lst
        self.shuffle = shuffle

    def __len__(self):
        return len(self.lst)

    def __iter__(self):
        if not self.shuffle:
            yield from self.lst
        else:
            idxs = np.arange(len(self.lst))
            self.rng.shuffle(idxs)
            for k in idxs:
                yield self.lst[k]


class DataFromIterable(DataFlow):
    """Wrap an iterable of datapoints to a DataFlow"""

    def __init__(self, iterable: Iterable[Any]):
        """
        Args:
            iterable: an iterable object
        """
        self._itr = iterable
        try:
            self._len = len(iterable)
        except NotImplementedError:
            self._len = None

    def __len__(self):
        if self._len is None:
            raise NotImplementedError
        return self._len

    def __iter__(self):
        yield from self._itr

    def reset_state(self):
        pass


class FakeData(RNGDataFlow):
    """Generate fake data of given shapes"""

    def __init__(self, shapes, size=1000, random=True, dtype="float32", domain=(0, 1)):
        """
        Args:
            shapes (list): a list of lists/tuples. Shapes of each component.
            size (int): size of this DataFlow.
            random (bool): whether to randomly generate data every iteration.
                Note that merely generating the data could sometimes be time-consuming!
            dtype (str or list): data type as string, or a list of data types.
            domain (tuple or list): (min, max) tuple, or a list of such tuples
        """
        super().__init__()
        self.shapes = shapes
        self._size = int(size)
        self.random = random
        self.dtype = [dtype] * len(shapes) if isinstance(dtype, str) else dtype
        self.domain = [domain] * len(shapes) if isinstance(domain, tuple) else domain
        assert len(self.dtype) == len(self.shapes)
        assert len(self.domain) == len(self.domain)

    def __len__(self):
        return self._size

    def __iter__(self):
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
                yield copy.copy(val)


class PickleSerializer:
    """A Serializer to load and to dump objects"""
    @staticmethod
    def dumps(obj):
        """
        Returns:
            bytes
        """
        return pickle.dumps(obj, protocol=-1)

    @staticmethod
    def loads(buf):
        """
        Args:
            bytes
        """
        return pickle.loads(buf)
