# -*- coding: utf-8 -*-
# File: develop.py

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


# Copyright (c) Tensorpack Contributors
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Utilities for developers only. These are not visible to users and should not appear in docs.
"""
import functools
import inspect
from collections import defaultdict
from datetime import datetime
from typing import Callable, Optional

from .logger import LoggingRecord, logger
from .types import T

__all__: list[str] = ["deprecated"]

# Copy and paste from https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/develop.py

_DEPRECATED_LOG_NUM = defaultdict(int)  # type: ignore


def log_deprecated(name: str, text: str, eos: str = "", max_num_warnings: Optional[int] = None) -> None:
    """
    Logs a deprecation warning.

    Args:
        name: Name of the deprecated item.
        text: Information about the deprecation.
        eos: End of service date such as "YYYY-MM-DD".
        max_num_warnings: The maximum number of times to print this warning.

    Note:
        Either `name` or `text` must be provided.
    """
    assert name or text
    if eos:
        eos = "after " + datetime(*map(int, eos.split("-"))).strftime("%d %b")  # type: ignore
    if name:
        if eos:
            info_msg = f"{name} will be deprecated {eos}. {text}"
        else:
            info_msg = f"{name} was deprecated. {text}"
    else:
        info_msg = text
        if eos:
            info_msg += f" Legacy period ends {eos}"

    if max_num_warnings is not None:
        if _DEPRECATED_LOG_NUM[info_msg] >= max_num_warnings:
            return
        _DEPRECATED_LOG_NUM[info_msg] += 1
    logger.info(LoggingRecord(f"[Deprecated] {info_msg}"))


def deprecated(
    text: str = "", eos: str = "", max_num_warnings: Optional[int] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator to deprecate a function.

    Example:
        ```python
        @deprecated("Explanation of what to do instead.", "2017-11-4")
        def foo(...):
            pass
        ```

    Args:
        text: Same as `log_deprecated`.
        eos: Same as `log_deprecated`.
        max_num_warnings: Same as `log_deprecated`.

    Returns:
        A decorator which deprecates the function.
    """

    def get_location() -> str:
        frame = inspect.currentframe()
        if frame:
            callstack = inspect.getouterframes(frame)[-1]
            return f"{callstack[1]}:{callstack[2]}"
        stack = inspect.stack(0)
        entry = stack[2]
        return f"{entry[1]}:{entry[2]}"

    def deprecated_inner(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def new_func(*args, **kwargs):  # type: ignore
            name = f"{func.__name__} [{get_location()}]"
            log_deprecated(name, text, eos, max_num_warnings=max_num_warnings)
            return func(*args, **kwargs)

        return new_func

    return deprecated_inner
