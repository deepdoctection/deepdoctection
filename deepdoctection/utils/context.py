# -*- coding: utf-8 -*-
# File: context.py

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
Some useful contextmanagers for various tasks
"""

import subprocess
from contextlib import contextmanager
from errno import ENOENT
from glob import iglob
from os import path, remove
from tempfile import NamedTemporaryFile
from time import perf_counter as timer
from typing import Any, Generator, Iterator, Optional, Union

import numpy as np

from .logger import LoggingRecord, logger
from .types import B64, B64Str, PixelValues
from .viz import viz_handler

__all__ = ["timeout_manager", "save_tmp_file", "timed_operation"]


@contextmanager
def timeout_manager(proc, seconds: Optional[int] = None) -> Iterator[str]:  # type: ignore
    """
    Manager for time handling while some process being called

       with timeout_manager(some_process,60) as timeout:
           ...

    :param proc: process
    :param seconds: seconds to wait

    """
    try:
        if not seconds:
            yield proc.communicate()[1]
            return

        try:
            _, error_string = proc.communicate(timeout=seconds)
            yield error_string
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.kill()
            proc.returncode = -1
            raise RuntimeError(f"timeout for process id: {proc.pid}")  # pylint: disable=W0707
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


@contextmanager
def save_tmp_file(image: Union[B64Str, PixelValues, B64], prefix: str) -> Iterator[tuple[str, str]]:
    """
    Save image temporarily and handle the clean-up once not necessary anymore

        with save_tmp_file(some_np_image,"tmp") as (tmp_name, input_file_name):
            ....

    :param image: image as string or numpy array
    :param prefix: prefix of the temp file name
    """
    try:
        with NamedTemporaryFile(prefix=prefix, delete=False) as file:
            if isinstance(image, str):
                yield file.name, path.realpath(path.normpath(path.normcase(image)))
                return
            if isinstance(image, (np.ndarray, np.generic)):
                input_file_name = file.name + "_input.PNG"
                viz_handler.write_image(input_file_name, image)
                yield file.name, input_file_name
            if isinstance(image, bytes):
                input_file_name = file.name
                file.write(image)
                file.flush()
                yield file.name, input_file_name
    finally:
        for file_name in iglob(file.name + "*" if file.name else file.name):
            try:
                remove(file_name)
            except OSError as error:
                if error.errno != ENOENT:
                    raise error


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")


@contextmanager
def timed_operation(message: str, log_start: bool = False) -> Generator[Any, None, None]:
    """
    Contextmanager with a timer.

    ... code-block:: python

        with timed_operation(message="Your stdout message", log_start=True):

            with open("log.txt", "a") as file:
               ...


    :param message: a log to stdout
    :param log_start: whether to print also the beginning
    """

    if log_start:
        logger.info(LoggingRecord(f"start task: {message} ..."))
    start = timer()
    yield
    logger.info(LoggingRecord(f"{message} total: {round(timer() - start, 4)} sec."))
