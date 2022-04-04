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
import functools
import itertools
from types import TracebackType
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from tabulate import tabulate
from termcolor import colored

from ..utils.detection_types import BaseExceptionType, MapFunc
from ..utils.logger import log_once, logger

__all__ = ["MappingContextManager", "DefaultMapper", "maybe_get_fake_score", "LabelSummarizer", "cur"]


class MappingContextManager:
    """
    A context for logging and catching some exceptions. Useful in a mapping function. It will remember outside the
    context if an exception has been thrown.
    """

    def __init__(self, dp_name: Optional[str] = "") -> None:
        """
        :param dp_name: A name for the datapoint to be mapped
        """
        self.dp_name = dp_name
        self.context_error = True

    def __enter__(self) -> "MappingContextManager":
        """
        context enter
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[BaseExceptionType],
        exc_val: Optional[BaseExceptionType],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """
        context exit
        """
        if exc_type in (KeyError, ValueError, IndexError, AssertionError) and exc_tb is not None:
            log_once(
                f"""dp: {self.dp_name}, err: {type(exc_val).__name__},
            msg: {str(exc_val)} in: {str(exc_tb.tb_frame)} will be filtered"""
            )
            return True
        if exc_type is None:
            self.context_error = False
        return None


class DefaultMapper:  # pylint: disable=R0903
    """
    A class that wraps a function and places some pre-defined values starting from the second argument  once the
    function is invoked.

    https://stackoverflow.com/questions/36314/what-is-currying
    """

    def __init__(self, func: Callable[[Any], Any], *args: Any, **kwargs: Any) -> None:
        """
        :param func: A mapping function
        :param args: Default args to pass to the function
        :param kwargs: Default kwargs to pass to the function
        """
        self.func = func
        self.argument_args = args
        self.argument_kwargs = kwargs

    def __call__(self, dp: Any) -> Any:
        """
        :param dp: datapoint within a dataflow
        :return: The return value of the invoked function with default arguments.
        """
        return self.func(dp, *self.argument_args, **self.argument_kwargs)


def cur(func: MapFunc) -> DefaultMapper:
    """
    Decorator for converting functions that map

    dps: Union[JsonDict,Image]  -> Union[JsonDict,Image]

    to DefaultMappers. They will be initialized with all arguments except dp and can be called later with only the
    datapoint as argument. This setting is useful when incorporating the function within a dataflow.

    **Example:**

               @cur
               def json_to_image(dp, config_arg_1, config_arg_2,...) -> Image:
               ...

               can be applied like:

               df = ...
               df = MapData(df,json_to_image(config_arg_1=val_1,config_arg_2=val_2))

    :param func: A callable [[:class:`Image`],[Any]] -> [:class:`Image`]
    :return: A DefaultMapper
    """

    @functools.wraps(func)
    def wrap(*args: Any, **kwargs: Any) -> DefaultMapper:
        ret = DefaultMapper(func, *args, **kwargs)
        return ret

    return wrap  # type: ignore


def maybe_get_fake_score(add_fake_score: bool) -> Optional[float]:
    """
    Returns a fake score, if add_fake_score = True. Will otherwise return None

    :param add_fake_score: boolean
    :return: A uniform random variable in (0,1)
    """
    if add_fake_score:
        return np.random.uniform(0.0, 1.0, 1)[0]
    return None


class LabelSummarizer:
    """
    A class for generating label statistics. Useful, when mapping and generating a SummaryAnnotation.
    """

    def __init__(self, categories: Dict[str, str]) -> None:
        """
        :param categories: A dict of categories as given as in categories.get_categories().
        """
        self.categories = categories
        cat_numbers = len(self.categories.keys())
        self.hist_bins = np.arange(1, cat_numbers + 2)
        self.summary = np.zeros(cat_numbers)

    def dump(self, item: Union[List[Union[str, int]], str, int]) -> None:
        """
        Dump a category number

        :param item: A category number.
        """
        item = np.asarray(item, dtype="int8")  # type: ignore
        self.summary += np.histogram(item, bins=self.hist_bins)[0]

    def get_summary(self) -> Dict[str, np.int32]:
        """
        Get a dictionary with category ids and the number dumped
        """
        return dict(list(zip(self.categories.keys(), self.summary.astype(np.int32))))

    def print_summary_histogram(self) -> None:
        """
        Prints a summary from all dumps.
        """
        data = list(itertools.chain(*[[self.categories[str(i + 1)], v] for i, v in enumerate(self.summary[:-1])]))
        num_columns = min(6, len(data))
        total_img_anns = sum(data[1::2])
        data.extend([None] * ((num_columns - len(data) % num_columns) % num_columns))
        data.extend(["total", total_img_anns])
        data = itertools.zip_longest(*[data[i::num_columns] for i in range(num_columns)])  # type: ignore
        table = tabulate(
            data, headers=["category", "#box"] * (num_columns // 2), tablefmt="pipe", stralign="center", numalign="left"
        )
        logger.info("Ground-Truth category distribution:\n %s", colored(table, "cyan"))
