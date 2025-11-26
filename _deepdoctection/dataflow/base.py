# -*- coding: utf-8 -*-
# File: base.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Most of the code has been taken from

<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/base.py>
"""

import threading
from abc import ABC, abstractmethod
from typing import Any, Iterator, no_type_check

from ..utils.utils import get_rng


class DataFlowReentrantGuard:
    """
    A tool to enforce non-reentrancy.
    Mostly used on DataFlow whose `get_data` is stateful,
    so that multiple instances of the iterator cannot co-exist.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        self._succ = self._lock.acquire(False)
        if not self._succ:
            raise threading.ThreadError("This dataflow is not reentrant!")

    @no_type_check
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()
        return False


class DataFlow:
    """Base class for all DataFlow"""

    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """
        - A dataflow is an iterable. The `__iter__` method should yield a list or dict each time.
          Note that dict is **partially** supported at the moment: certain dataflow does not support dict.
        - The `__iter__` method can be either finite (will stop iteration) or infinite
          (will not stop iteration). For a finite dataflow, `__iter__` can be called
          again immediately after the previous call returned.
        - For many dataflow, the `__iter__` method is non-reentrant, which means for a dataflow
          instance ``df``, `df.__iter__` cannot be called before the previous
          `df.__iter__` call has finished (iteration has stopped).
          When a dataflow is non-reentrant, `df.__iter__` should throw an exception if
          called before the previous call has finished.
          For such non-reentrant dataflows, if you need to use the same dataflow in two places,
          you need to create two dataflow instances.
        Yields:
            list/dict: The datapoint, i.e. list/dict of components.
        """

    def __len__(self) -> int:
        """
        - A dataflow can optionally implement `__len__`. If not implemented, it will
          throw `NotImplementedError`.
        - It returns an integer representing the size of the dataflow.
          The return value **may not be accurate or meaningful** at all.
          When saying the length is "accurate", it means that
          `__iter__` will always yield this many of datapoints before it stops iteration.
        - There could be many reasons why `__len__` is inaccurate.
          For example, some dataflow has dynamic size, if it throws away datapoints on the fly.
          Some dataflow mixes the datapoints between consecutive passes over
          the dataset, due to parallelism and buffering.
          In this case it does not make sense to stop the iteration anywhere.
        - Due to the above reasons, the length is only a rough guidance.
          And it's up to the user how to interpret it.
          Inside tensorpack it's only used in these places:
          + A default ``steps_per_epoch`` in training, but you probably want to customize
            it yourself, especially when using data-parallel trainer.
          + The length of progress bar when processing a dataflow.
          + Used by `InferenceRunner` to get the number of iterations in inference.
            In this case users are **responsible** for making sure that `__len__` is "accurate".
            This is to guarantee that inference is run on a fixed set of images.

        Returns:
            int: rough size of this dataflow.

        Raises:
            NotImplementedError: if this DataFlow doesn't have a size.
        """
        raise NotImplementedError

    def reset_state(self) -> None:
        """
        - The caller must guarantee that `reset_state` should be called **once and only once**
          by the **process that uses the dataflow** before `__iter__` is called.
          The caller thread of this method should stay alive to keep this dataflow alive.
        - It is meant for certain initialization that involves processes,
          e.g., initialize random number generators (RNG), create worker processes.
          Because it's very common to use RNG in data processing,
          developers of dataflow can also subclass `RNGDataFlow` to have easier access to
          a properly-initialized RNG.
        - A dataflow is not fork-safe after `reset_state` is called (because this will violate the guarantee).
          There are a few other dataflows that are not fork-safe anytime, which will be mentioned in the docs.
        - You should take the responsibility and follow the above guarantee if you're the caller of a dataflow yourself
          (either when you're using dataflow outside tensorpack, or if you're writing a wrapper dataflow).
        - Tensorpack's built-in forking dataflows (`MultiProcessRunner`, `MultiProcessMapData`, etc)
          and other component that uses dataflows (`InputSource`)
          already take care of the responsibility of calling this method.
        """
        raise NotImplementedError


class RNGDataFlow(DataFlow, ABC):
    """A DataFlow with RNG"""

    rng = None
    """
    `self.rng` is a `np.random.RandomState` instance that is initialized
    correctly (with different seeds in each process) in `RNGDataFlow.reset_state()`.
    """

    def reset_state(self) -> None:
        """Reset the RNG"""
        self.rng = get_rng(self)


class ProxyDataFlow(DataFlow):
    """Base class for DataFlow that proxies another.
    Every method is proxied to ``self.df`` unless overriden by a subclass.
    """

    def __init__(self, df: DataFlow) -> None:
        """
        Initializes the ProxyDataFlow.

        Args:
            df: DataFlow to proxy.
        """
        self.df = df

    def reset_state(self) -> None:
        """Resets the state of the proxied DataFlow."""
        self.df.reset_state()

    def __len__(self) -> int:
        """
        Returns the size of the proxied DataFlow.

        Returns:
            int: Size of the proxied DataFlow.
        """
        return self.df.__len__()

    def __iter__(self) -> Iterator[Any]:
        """
        Iterates over the proxied DataFlow.

        Returns:
            Iterator[Any]: Iterator of the proxied DataFlow.
        """
        return self.df.__iter__()
