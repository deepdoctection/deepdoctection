# -*- coding: utf-8 -*-
# File: maputils.py

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
Utility functions related to mapping tasks
"""
from __future__ import annotations

import functools
import itertools
import traceback
from types import TracebackType
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
from tabulate import tabulate
from termcolor import colored

from ..utils.error import AnnotationError, BoundingBoxError, ImageError, UUIDError
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import ObjectTypes
from ..utils.types import DP, BaseExceptionType, S, T

__all__ = ["MappingContextManager", "DefaultMapper", "maybe_get_fake_score", "LabelSummarizer", "curry"]


class MappingContextManager:
    """
    A context for logging and catching some exceptions. Useful in a mapping function. It will remember outside the
    context if an exception has been thrown.
    """

    def __init__(
        self, dp_name: Optional[str] = None, filter_level: str = "image", **kwargs: dict[str, Optional[str]]
    ) -> None:
        """
        Args:
            dp_name: A name for the datapoint to be mapped.
            filter_level: Indicates if the `MappingContextManager` is used on datapoint level, annotation level etc.
                          `filter_level` will only be used for logging.

        Note:
            Use this context manager to catch and log exceptions during mapping.
        """
        self.dp_name = dp_name if dp_name is not None else ""
        self.filter_level = filter_level
        self.context_error = True
        self.kwargs = kwargs

    def __enter__(self) -> MappingContextManager:
        """
        Context enter.

        Returns:
            The `MappingContextManager` instance.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseExceptionType],
        exc_val: Optional[BaseExceptionType],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """
        Context exit.

        Args:
            exc_type: The exception type.
            exc_val: The exception value.
            exc_tb: The traceback object.

        Returns:
            `True` if the exception was handled, otherwise `None`.
        """
        if (
            exc_type
            in (
                KeyError,
                ValueError,
                IndexError,
                AssertionError,
                TypeError,
                FileNotFoundError,
                AttributeError,
                BoundingBoxError,
                AnnotationError,
                ImageError,
                UUIDError,
            )
            and exc_tb is not None
        ):
            frame_summary = traceback.extract_tb(exc_tb)[0]
            log_dict = {
                "file_name": self.dp_name,
                "error_type": type(exc_val).__name__,
                "error_msg": str(exc_val),
                "orig_module": frame_summary.filename,
                "line": frame_summary.lineno,
            }
            for key, value in self.kwargs.items():
                if isinstance(value, dict):
                    log_dict["type"] = key
                    log_dict.update(value)
            logger.warning(
                LoggingRecord(f"MappingContextManager error. Will filter {self.filter_level}", log_dict)  # type: ignore
            )

            return True
        if exc_type is None:
            self.context_error = False
        return None


class DefaultMapper:
    """
    A class that wraps a function and places some pre-defined values starting from the second argument  once the
    function is invoked.

    <https://stackoverflow.com/questions/36314/what-is-currying>
    """

    def __init__(self, func: Callable[[DP, S], T], *args: Any, **kwargs: Any) -> None:
        """
        Args:
            func: A mapping function
            args: Default `args` to pass to the function
            kwargs: Default `kwargs` to pass to the function
        """
        self.func = func
        self.argument_args = args
        self.argument_kwargs = kwargs

    def __call__(self, dp: Any) -> Any:
        """
        Call the wrapped function with the given datapoint and default arguments.

        Args:
            dp: Datapoint within a dataflow.

        Returns:
            The return value of the invoked function with default arguments.
        """
        return self.func(dp, *self.argument_args, **self.argument_kwargs)


def curry(func: Callable[..., T]) -> Callable[..., Callable[[DP], T]]:
    """
    Decorator for converting functions that map

    ```python
    dps: Union[JsonDict, Image] -> Union[JsonDict, Image]
    ```

    to `DefaultMapper`s. They will be initialized with all arguments except `dp` and can be called later with only the
    datapoint as argument. This setting is useful when incorporating the function within a dataflow.

    Example:
        ```python
        @curry
        def json_to_image(dp, config_arg_1, config_arg_2, ...) -> Image:
            ...
        df = ...
        df = MapData(df, json_to_image(config_arg_1=val_1, config_arg_2=val_2))
        ```

    Args:
        func: A callable [[`Image`], [Any]] -> [`Image`]

    Returns:
        A `DefaultMapper`.
    """

    @functools.wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> DefaultMapper:
        return DefaultMapper(func, *args, **kwargs)

    return wrap


def maybe_get_fake_score(add_fake_score: bool) -> Optional[float]:
    """
    Returns a fake score, if `add_fake_score` is `True`. Will otherwise return `None`.

    Args:
        add_fake_score: Boolean.

    Returns:
        A uniform random variable in `(0,1)` or `None`.
    """
    if add_fake_score:
        return np.random.uniform(0.0, 1.0, 1)[0]
    return None


class LabelSummarizer:
    """
    A class for generating label statistics. Useful when mapping and generating a `SummaryAnnotation`.

    Example:
        ```python
        summarizer = LabelSummarizer({"1": "label_1", "2": "label_2"})
        for dp in some_dataflow:
            summarizer.dump(dp["label_id"])
        summarizer.print_summary_histogram()
        ```

    Args:
        categories: A dict of categories as given as in `categories.get_categories()`.
    """

    def __init__(self, categories: Mapping[int, ObjectTypes]) -> None:
        """
        Args:
            categories: A dict of categories as given as in `categories.get_categories()`.
        """
        self.categories = categories
        cat_numbers = len(self.categories.keys())
        self.hist_bins = np.arange(1, cat_numbers + 2)
        self.summary = np.zeros(cat_numbers)

    def dump(self, item: Union[Sequence[Union[str, int]], str, int]) -> None:
        """
        Dump a category number.

        Args:
            item: A category number.
        """
        np_item = np.asarray(item, dtype="int8")
        self.summary += np.histogram(np_item, bins=self.hist_bins)[0]

    def get_summary(self) -> dict[int, int]:
        """
        Get a dictionary with category ids and the number dumped.

        Returns:
            A dictionary mapping category ids to counts.
        """
        return dict(list(zip(self.categories.keys(), self.summary.tolist())))

    def print_summary_histogram(self, dd_logic: bool = True) -> None:
        """
        Prints a summary from all dumps.

        Args:
            dd_logic: Follow dd category convention when printing histogram (last background bucket omitted).
        """
        if dd_logic:
            data = list(itertools.chain(*[[self.categories[i].value, v] for i, v in enumerate(self.summary, 1)]))
        else:
            data = list(itertools.chain(*[[self.categories[i + 1].value, v] for i, v in enumerate(self.summary[:-1])]))
        num_columns = min(6, len(data))
        total_img_anns = sum(data[1::2])
        data.extend([None] * ((num_columns - len(data) % num_columns) % num_columns))
        data.extend(["total", total_img_anns])
        data = itertools.zip_longest(*[data[i::num_columns] for i in range(num_columns)])  # type: ignore
        table = tabulate(
            data, headers=["category", "#box"] * (num_columns // 2), tablefmt="pipe", stralign="center", numalign="left"
        )
        logger.info(LoggingRecord(f"Ground-Truth category distribution:\n {colored(table, 'cyan')}"))
