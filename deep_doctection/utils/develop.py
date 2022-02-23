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


import functools
from collections import defaultdict
from datetime import datetime
from typing import Optional
from . import logger

__all__ = []

# Copy and paste from https://github.com/tensorpack/tensorpack/blob/master/tensorpack/utils/develop.py

_DEPRECATED_LOG_NUM = defaultdict(int)


def log_deprecated(name: str ="", text: str ="", eos: str ="", max_num_warnings=None) -> None:
    """
    Log deprecation warning.

    :param name: name of the deprecated item.
    :param text: information about the deprecation.
    :param eos: end of service date such as "YYYY-MM-DD".
    :param max_num_warnings: the maximum number of times to print this warning
    """
    assert name or text
    if eos:
        eos = "after " + datetime(*map(int, eos.split("-"))).strftime("%d %b")
    if name:
        if eos:
            warn_msg = "%s will be deprecated %s. %s" % (name, eos, text)
        else:
            warn_msg = "%s was deprecated. %s" % (name, text)
    else:
        warn_msg = text
        if eos:
            warn_msg += " Legacy period ends %s" % eos

    if max_num_warnings is not None:
        if _DEPRECATED_LOG_NUM[warn_msg] >= max_num_warnings:
            return
        _DEPRECATED_LOG_NUM[warn_msg] += 1
    logger.warn("[Deprecated] " + warn_msg)


def deprecated(text: str = "", eos: str = "", max_num_warnings: Optional[int]=None):
    """

    :param text: same as :func:`log_deprecated`.
    :param eos: same as :func:`log_deprecated`.
    :param max_num_warnings: same as :func:`log_deprecated`.

    :return: A decorator which deprecates the function.

    **Example:**

        .. code-block:: python
            @deprecated("Explanation of what to do instead.", "2017-11-4")
            def foo(...):
                pass
    """

    def get_location():
        import inspect
        frame = inspect.currentframe()
        if frame:
            callstack = inspect.getouterframes(frame)[-1]
            return '%s:%i' % (callstack[1], callstack[2])
        else:
            stack = inspect.stack(0)
            entry = stack[2]
            return '%s:%i' % (entry[1], entry[2])

    def deprecated_inner(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            name = "{} [{}]".format(func.__name__, get_location())
            log_deprecated(name, text, eos, max_num_warnings=max_num_warnings)
            return func(*args, **kwargs)
        return new_func
    return deprecated_inner

