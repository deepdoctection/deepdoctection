# -*- coding: utf-8 -*-
# File: concurrency.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
Some utility functions for multi threading purposes
"""

import multiprocessing as mp
import platform
import queue
import signal
import sys
import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional, no_type_check

from .logger import log_once
from .types import QueueType


# taken from https://github.com/tensorpack/dataflow/blob/master/dataflow/utils/concurrency.py
class StoppableThread(threading.Thread):
    """
    A thread that has a 'stop' event.
    """

    def __init__(self, evt: Optional[threading.Event] = None) -> None:
        """
        :param evt: if None, will create one.
        """
        super().__init__()
        if evt is None:
            evt = threading.Event()
        self._stop_evt = evt

    def stop(self) -> None:
        """Stop the thread"""
        self._stop_evt.set()

    def stopped(self) -> bool:
        """
        :param bool: whether the thread is stopped or not
        """
        return self._stop_evt.is_set()

    def queue_put_stoppable(self, q: QueueType, obj: Any) -> None:
        """Put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=5)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q: QueueType) -> Any:
        """Take obj from queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=5)
            except queue.Empty:
                pass


@contextmanager
def mask_sigint() -> Generator[Any, None, None]:
    """[Any,None,None
    :return: If called in main thread, returns a context where ``SIGINT`` is ignored, and yield True.
             Otherwise, yield False.
    """
    if threading.current_thread() == threading.main_thread():
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        yield True
        signal.signal(signal.SIGINT, sigint_handler)
    else:
        yield False


def enable_death_signal(_warn: bool = True) -> None:
    """
    Set the "death signal" of the current process, so that
    the current process will be cleaned with guarantee
    in case the parent dies accidentally.
    """
    if platform.system() != "Linux":
        return
    try:
        import prctl  # type: ignore #pylint: disable=C0415  # pip install python-prctl
    except ImportError:
        if _warn:
            log_once(
                '"import prctl" failed! Install python-prctl so that processes can be cleaned with guarantee.', "warn"
            )
        return
    assert hasattr(
        prctl, "set_pdeathsig"
    ), "prctl.set_pdeathsig does not exist! Note that you need to install 'python-prctl' instead of 'prctl'."
    # is SIGHUP a good choice?
    prctl.set_pdeathsig(signal.SIGHUP)  # pylint: disable=E1101


# taken from https://github.com/tensorpack/dataflow/blob/master/dataflow/utils/concurrency.py


@no_type_check
def start_proc_mask_signal(proc):
    """
    Start process(es) with SIGINT ignored.

    :param proc: (mp.Process or list)

    The signal mask is only applied when called from main thread.
    """
    if not isinstance(proc, list):
        proc = [proc]

    with mask_sigint():
        for pro in proc:
            if isinstance(pro, mp.Process):
                if sys.version_info < (3, 4) or mp.get_start_method() == "fork":
                    log_once(
                        """
Starting a process with 'fork' method is efficient but not safe and may cause deadlock or crash.
Use 'forkserver' or 'spawn' method instead if you run into such issues.
See https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods on how to set them.
""".replace(
                            "\n", ""
                        ),
                        "warn",
                    )  # noqa
            pro.start()
