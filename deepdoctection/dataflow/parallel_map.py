# -*- coding: utf-8 -*-
# File: parallel_map.py

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Replaces relevant parts of the Dataflow package. Most of the functions have been taken from

<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/parallel.py>
<https://github.com/tensorpack/dataflow/blob/master/dataflow/dataflow/parallel_map.py>
"""

import atexit
import errno
import multiprocessing as mp
import os
import queue
import sys
import threading
import uuid
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable, Iterator, no_type_check

import zmq

from ..utils.concurrency import StoppableThread, enable_death_signal, start_proc_mask_signal
from ..utils.error import DataFlowTerminatedError
from ..utils.logger import LoggingRecord, logger
from .base import DataFlow, DataFlowReentrantGuard, ProxyDataFlow
from .common import RepeatedData
from .serialize import PickleSerializer


@no_type_check
def del_weakref(x):
    """delete weakref"""
    instance = x()
    if instance is not None:
        del instance


@no_type_check
@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated as exc:
        logger.info(LoggingRecord(f"_zmq_catch_error: [{name}] Context terminated."))
        raise DataFlowTerminatedError() from exc
    except zmq.ZMQError as exc:
        if exc.errno == errno.ENOTSOCK:  # socket closed
            logger.info(LoggingRecord(f"_zmq_catch_error: [{name}]  Socket closed."))
            raise DataFlowTerminatedError() from exc
        raise ValueError() from exc
    except Exception as exc:
        raise ValueError() from exc


@no_type_check
def _get_pipe_name(name):
    if sys.platform.startswith("linux"):
        # linux supports abstract sockets: http://api.zeromq.org/4-1:zmq-ipc
        pipename = f"ipc://@{name}-pipe-{str(uuid.uuid1())[:8]}"
    else:
        pipedir = "."
        if not os.path.isdir(pipedir):
            raise NotADirectoryError(pipedir)
        filename = f"{pipedir.rstrip('/')}/{name}-pipe-{str(uuid.uuid1())[:6]}"
        if os.path.exists(filename):
            raise FileExistsError(filename)
        pipename = f"ipc://{filename}"
    return pipename


class _ParallelMapData(ProxyDataFlow, ABC):
    def __init__(self, df: DataFlow, buffer_size: int, strict: bool = False) -> None:
        super().__init__(df)
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be a positive number, got {buffer_size}")
        self._buffer_size = buffer_size
        self._buffer_occupancy = 0  # actual #elements in buffer, only useful in strict mode
        self._strict = strict

    def reset_state(self) -> None:
        super().reset_state()
        if not self._strict:
            df = RepeatedData(self.df, -1)
        else:
            df = self.df  # type: ignore
        self._iter = iter(df)

    @no_type_check
    @abstractmethod
    def _recv(self):
        raise NotImplementedError()

    @no_type_check
    @abstractmethod
    def _send(self, dp: Any):
        raise NotImplementedError()

    @no_type_check
    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, f"[{type(self).__name__}] map function cannot return None when strict mode is used."
        return ret

    @no_type_check
    def _fill_buffer(self, cnt=None):
        if cnt is None:
            cnt = self._buffer_size - self._buffer_occupancy
        try:
            for _ in range(cnt):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration as exce:
            raise RuntimeError(
                f"[{type(self).__name__}] buffer_size cannot be larger than the size of the dataflow when strict=True! "
                "Please use a smaller buffer_size!"
            ) from exce
        self._buffer_occupancy += cnt

    def get_data_non_strict(self) -> Any:
        """data non strict"""
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self) -> Any:
        """data strict"""
        self._fill_buffer()
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = iter(self.df)  # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            self._buffer_occupancy -= 1
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp

    def __iter__(self) -> Iterator[Any]:
        if self._strict:
            yield from self.get_data_strict()
        else:
            yield from self.get_data_non_strict()


class MultiThreadMapData(_ParallelMapData):
    """
    Same as `MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.
    The semantics of this class is **identical** to `MapData` except for the ordering.
    Threads run in parallel and can take different time to run the
    mapping function. Therefore, the order of datapoints won't be preserved.
    When `strict=True`, `MultiThreadMapData(df, ...)`
    is guaranteed to produce the exact set of data as `MapData(df, ...)`,
    if both are iterated until `StopIteration`. But the produced data will have different ordering.
    The behavior of strict mode is undefined if the given dataflow `df` is infinite.
    When `strict=False`, the data that's produced by `MultiThreadMapData(df, ...)`
    is a re-ordering of the data produced by `RepeatedData(MapData(df, ...), -1)`.
    In other words, first pass of `MultiThreadMapData.__iter__` may contain
    datapoints from the second pass of `df.__iter__`.

    Note:
        1. You should avoid starting many threads in your main process to reduce GIL contention.
           The threads will only start in the process which calls `reset_state()`.
           Therefore you can use `MultiProcessRunnerZMQ(MultiThreadMapData(...), 1)`
           to reduce GIL contention.
    """

    class _Worker(StoppableThread):
        @no_type_check
        def __init__(self, inq, outq, evt, map_func):
            super(MultiThreadMapData._Worker, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True

        @no_type_check
        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self.inq)
                    if self.stopped():
                        return
                    # cannot ignore None here. will lead to unsynced send/recv
                    obj = self.func(dp)
                    self.queue_put_stoppable(self.outq, obj)
            except Exception:  # pylint: disable=W0703
                if self.stopped():
                    pass  # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(
        self,
        df: DataFlow,
        num_thread: int,
        map_func: Callable[[Any], Any],
        *,
        buffer_size: int = 200,
        strict: bool = False,
    ):
        """
        Args:
            df: the dataflow to map
            num_thread: number of threads to use
            map_func: datapoint -> datapoint | None. Return None to
                      discard/skip the datapoint.
            buffer_size: number of datapoints in the buffer
            strict: use "strict mode", see notes above.
        """
        if strict:
            # In strict mode, buffer size cannot be larger than the total number of datapoints
            try:
                buffer_size = min(buffer_size, len(df))
            except Exception:  # pylint: disable=W0703  # df may not have a length
                pass

        super().__init__(df, buffer_size, strict)
        if not num_thread:
            raise ValueError("num_thread must be a positive number")

        self._strict = strict
        self.num_thread = num_thread
        self.map_func = map_func
        self._threads: list[Any] = []
        self._evt = None

    def reset_state(self) -> None:
        super().reset_state()
        if self._threads:
            self._threads[0].stop()
            for thr in self._threads:
                thr.join()

        self._in_queue: queue.Queue[Any] = queue.Queue()
        self._out_queue: queue.Queue[Any] = queue.Queue()
        self._evt = threading.Event()  # type: ignore
        self._threads = [
            MultiThreadMapData._Worker(self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.num_thread)
        ]
        for thr in self._threads:
            thr.start()

        self._guard = DataFlowReentrantGuard()

        # Call once at the beginning, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self) -> Any:
        return self._out_queue.get()

    def _send(self, dp: Any) -> None:
        self._in_queue.put(dp)

    def __iter__(self) -> Iterator[Any]:
        with self._guard:
            yield from super().__iter__()

    def __del__(self) -> Any:
        if self._evt is not None:
            self._evt.set()
        for thr in self._threads:
            thr.stop()
            thr.join(timeout=5.0)


class _MultiProcessZMQDataFlow(DataFlow, ABC):
    def __init__(self) -> None:
        if os.name == "nt":
            raise EnvironmentError("ZMQ IPC doesn't support windows")
        self._reset_done = False
        self._procs: list[Any] = []
        self.context = None
        self.socket = None

    def reset_state(self) -> Any:
        """
        All forked dataflows should only be reset once and only once in spawned processes.
        Subclasses should call this method with super.
        """
        assert not self._reset_done, "reset_state() was called twice! This violates the API of DataFlow!"
        self._reset_done = True

        # __del__ not guaranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

    def _start_processes(self) -> Any:
        start_proc_mask_signal(self._procs)

    @no_type_check
    def __del__(self) -> None:
        try:
            if not self._reset_done:
                return
            if not self.context.closed:
                self.socket.close(0)
                self.context.destroy(0)
            for x in self._procs:
                x.terminate()
                x.join(5)
            logger.info(LoggingRecord(f"_MultiProcessZMQDataFlow [{type(self).__name__}] successfully cleaned-up."))

        except Exception:  # pylint: disable=W0703
            pass


@no_type_check
def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(
            LoggingRecord(
                f"ZMQError in socket.bind('{name}'). Perhaps you're using pipes on a non-local file system. "
                "See documentation of MultiProcessRunnerZMQ for more information."
            )
        )

        raise


class MultiProcessMapData(_ParallelMapData, _MultiProcessZMQDataFlow):
    """
    Same as `MapData`, but start processes to run the mapping function,
    and communicate with ZeroMQ pipe.
    The semantics of this class is identical to `MapData` except for the ordering.
    Processes run in parallel and can take different time to run the
    mapping function. Therefore, the order of datapoints won't be preserved.
    When `strict=True`, `MultiProcessMapData(df, ...)`
    is guaranteed to produce the exact set of data as `MapData(df, ...)`,
    if both are iterated until `StopIteration`. But the produced data will have different ordering.
    The behavior of strict mode is undefined if the given dataflow `df` is infinite.
    When `strict=False`, the data that's produced by `MultiProcessMapData(df, ...)`
    is a reordering of the data produced by `RepeatedData(MapData(df, ...), -1)`.
    In other words, first pass of `MultiProcessMapData.__iter__` may contain
    datapoints from the second pass of `df.__iter__`.
    """

    class _Worker(mp.Process):
        @no_type_check
        def __init__(self, identity, map_func, pipename, hwm):
            super(MultiProcessMapData._Worker, self).__init__()
            self.identity = identity
            self.map_func = map_func
            self.pipename = pipename
            self.hwm = hwm

        @no_type_check
        def run(self):
            enable_death_signal(_warn=self.identity == b"0")
            ctx = zmq.Context()
            socket = ctx.socket(zmq.REP)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipename)

            while True:
                dp = PickleSerializer.loads(socket.recv(copy=False))
                dp = self.map_func(dp)
                socket.send(PickleSerializer.dumps(dp), copy=False)

    def __init__(
        self,
        df: DataFlow,
        num_proc: int,
        map_func: Callable[[Any], Any],
        *,
        buffer_size: int = 200,
        strict: bool = False,
    ) -> None:
        """
        Args:
            df: the dataflow to map
            num_proc: number of threads to use
            map_func: datapoint -> datapoint | None. Return None to
            buffer_size: number of datapoints in the buffer
            strict: use "strict mode", see notes above.
        """
        if strict:
            # In strict mode, buffer size cannot be larger than the total number of datapoints
            try:
                buffer_size = min(buffer_size, len(df))
            except Exception:  # pylint: disable=W0703  # ds may not have a length
                pass

        _ParallelMapData.__init__(self, df, buffer_size, strict)
        _MultiProcessZMQDataFlow.__init__(self)
        if num_proc <= 0:
            raise ValueError(f"num_proc must be a positive number, got {num_proc}")
        self.num_proc = num_proc
        self.map_func = map_func
        self._strict = strict
        self._procs = []

    @no_type_check
    def _create_worker(self, idx, pipename, hwm):
        return MultiProcessMapData._Worker(idx, self.map_func, pipename, hwm)

    def reset_state(self) -> None:
        _MultiProcessZMQDataFlow.reset_state(self)
        _ParallelMapData.reset_state(self)
        self._guard = DataFlowReentrantGuard()

        self.context = zmq.Context()  # type: ignore
        self.socket = self.context.socket(zmq.DEALER)  # type: ignore
        self.socket.set_hwm(self._buffer_size * 2)  # type: ignore
        pipename = _get_pipe_name("dataflow-map")
        _bind_guard(self.socket, pipename)

        self._proc_ids = [f"{k}".encode("utf-8") for k in range(self.num_proc)]
        worker_hwm = int(self._buffer_size * 2 // self.num_proc)
        self._procs = [self._create_worker(self._proc_ids[k], pipename, worker_hwm) for k in range(self.num_proc)]

        self._start_processes()
        self._fill_buffer()  # pre-fill the buffer

    @no_type_check
    def _send(self, dp: Any):
        msg = [b"", PickleSerializer.dumps(dp)]
        self.socket.send_multipart(msg, copy=False)

    @no_type_check
    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = PickleSerializer.loads(msg[1])
        return dp

    def __iter__(self) -> Iterator[Any]:
        with self._guard, _zmq_catch_error(type(self).__name__):
            yield from super().__iter__()
