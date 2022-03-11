# -*- coding: utf-8 -*-
# File: timer.py

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
Some useful stuff about timing operations and stopwatch methods. Some of them are taken from the Tensorpack library
"""

from contextlib import contextmanager
from time import perf_counter as timer
from typing import Any, Generator

from .logger import logger

__all__ = ["timed_operation"]

# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")


@contextmanager
def timed_operation(message: str, log_start: bool = False) -> Generator[Any, None, None]:
    """
    Contextmanager with a timer. Can therefore be used in a with statement.

    :param message: a log to print
    :param log_start: whether to print also the beginning
    """

    assert len(message)
    if log_start:
        logger.info(
            "start task: %s ...",
        )
    start = timer()
    yield
    logger.info("%s finished, %s sec.", message, round(timer() - start, 4))
