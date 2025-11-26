# -*- coding: utf-8 -*-
# File: stats.py

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
Dataflows for calculating statistical values of the underlying dataset
"""

from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt

from ..utils.logger import LoggingRecord, logger
from ..utils.tqdm import get_tqdm
from .base import DataFlow, ProxyDataFlow


class MeanFromDataFlow(ProxyDataFlow):
    """
    Get the mean of some dataflow. Takes a component from a dataflow and calculates iteratively the mean.

    Example:
        ```python
        df: some dataflow
        MeanFromDataFlow(df).start() # If you want to put MeanFromDataFlow at the end of a dataflow

        or

        df: some dataflow
        df = MeanFromDataFlow(df)

        is also possible. Testing with the progress bar will stop once the requested size has been reached.
        ```


    """

    def __init__(
        self,
        df: DataFlow,
        axis: Optional[Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]] = None,
        key: Optional[str] = None,
        max_datapoints: Optional[int] = None,
    ):
        """
        Args:
            df: the dataflow to test
            axis: The axis along which to calculate the mean. It will always calculate the mean along
                  the dataflow length axis, which is the 0th axis. E.g. for calculating the mean of an image dataset
                  use
                      ```python
                      MeanFromDataFlow(df,key="image",axis=(0,1,2)).start()
                      ```

            key: The datapoint key giving the values to evaluate the mean. If None it tries to use
                 the whole datapoint.
            max_datapoints: Will stop considering datapoints with index>max_datapoints.
        """

        super().__init__(df)
        if isinstance(axis, int):
            axis = (axis,)
        if axis is not None:
            if 0 not in axis:
                raise ValueError("0-th axis required for calculating mean")
        self.axis = axis
        self.key = key
        self.max_datapoints = max_datapoints
        self.mean: npt.NDArray[Any]
        self._reset_called = False

    def reset_state(self) -> None:
        self._reset_called = True
        super().reset_state()

    def __iter__(self) -> Any:
        """
        Will run the mean calculation at the until max_datapoints (if given) has been reached, then it will
        produce data normally.
        """
        self.start()
        yield from self.df

    def start(self) -> npt.NDArray[Any]:
        """
        Start calculating the mean with a progress bar.
        """

        if not self._reset_called:
            self.df.reset_state()
        itr = iter(self.df)

        logger.info(LoggingRecord("Calculating mean"))

        len_df: Optional[int]
        try:
            len_df = len(self.df)
        except NotImplementedError:
            len_df = None
        if len_df is not None and self.max_datapoints is not None:
            len_df = min(len_df, self.max_datapoints)

        with get_tqdm(total=len_df) as status_bar:
            n = None
            for n, dp in enumerate(itr, 1):
                if isinstance(dp, dict):
                    assert isinstance(self.key, str), self.key
                    val = dp[self.key]
                elif isinstance(dp, list):
                    val = dp
                else:
                    assert isinstance(self.key, str), self.key
                    val = getattr(dp, self.key)

                if isinstance(val, (tuple, list)):
                    val = np.asarray(val)

                if n == 1:
                    val_0_ndim = val.ndim
                    inner_axis = None
                    if self.axis is None:
                        self.mean = np.zeros(())
                    else:
                        inner_axis = tuple(np.subtract(self.axis, tuple((1 for _ in self.axis))))[1:]
                        mean_shape = tuple(val.shape[i] for i in range(len(val.shape)) if i not in inner_axis)
                        self.mean = np.zeros(mean_shape)

                if val.ndim == val_0_ndim:
                    x = np.mean(val, axis=self.axis if inner_axis is None else inner_axis)
                    self.mean += (x - self.mean) / n

                status_bar.update()
                if self.max_datapoints is not None:
                    if n == self.max_datapoints:
                        break

        logger.info(LoggingRecord(f"Mean from {n} datapoints along axis {self.axis}: {self.mean}"))

        return self.mean


class StdFromDataFlow(ProxyDataFlow):
    """
    Gets the standard deviation of some dataflow. Takes a component from a dataflow and calculates iteratively
    the standard deviation.

    Example:
        ```python
        df= ...
        StdFromDataFlow(df).start()

        if you want to put  StdFromDataFlow at the end of a dataflow

        df: some dataflow
        df = StdFromDataFlow(df)

        is also possible. The testing with the progress bar will stop once the requested size has been reached.
        ```

    """

    def __init__(
        self,
        df: DataFlow,
        axis: Optional[Union[int, tuple[int], tuple[int, int], tuple[int, int, int]]] = None,
        key: Optional[str] = None,
        max_datapoints: Optional[int] = None,
    ):
        """
        Args:
            df: the dataflow to test
            axis: The axis along which to calculate the mean. It will always calculate the std along
                     the dataflow length axis, which is the 0th axis. E.g. for calculating the mean of an image dataset
                     use
                         ```python
                         StdFromDataFlow(df,key="image",axis=(0,1,2)).start()
                         ```

            key: The datapoint key giving the values to evaluate the std. If None it tries to use
                 the whole datapoint.
            max_datapoints: Will stop considering datapoints with index>max_datapoints.
        """

        super().__init__(df)
        if isinstance(axis, int):
            axis = (axis,)
        if axis is not None:
            if 0 not in axis:
                raise ValueError("0-th axis required for calculating std")
        self.axis = axis
        self.key = key
        self.max_datapoints = max_datapoints
        self.std: npt.NDArray[Any]
        self._reset_called = False

    def reset_state(self) -> None:
        self._reset_called = True
        super().reset_state()

    def __iter__(self) -> Any:
        """
        Will run the mean calculation at the until max_datapoints (if given) has been reached, then it will
        produce data normally.
        """
        self.start()
        yield from self.df

    def start(self) -> npt.NDArray[Any]:
        """
        Start calculating the mean with a progress bar.
        """
        len_df: Optional[int]

        if not self._reset_called:
            self.df.reset_state()
        itr = iter(self.df)

        logger.info(LoggingRecord("Calculating standard deviation"))
        try:
            len_df = len(self.df)
        except NotImplementedError:
            len_df = None
        if len_df is not None and self.max_datapoints is not None:
            len_df = min(len_df, self.max_datapoints)
        n = None
        with get_tqdm(total=len_df) as status_bar:
            for n, dp in enumerate(itr, 1):
                if isinstance(dp, dict):
                    assert isinstance(self.key, str), self.key
                    val = dp[self.key]
                elif isinstance(dp, list):
                    val = dp
                else:
                    assert isinstance(self.key, str), self.key
                    val = getattr(dp, self.key)

                if isinstance(val, (tuple, list)):
                    val = np.asarray(val)

                if n == 1:
                    val_0_ndim = val.ndim
                    inner_axis = None
                    if self.axis is None:
                        self.std = np.zeros(())
                        ex, ex2 = np.zeros(()), np.zeros(())
                    else:
                        inner_axis = tuple(np.subtract(self.axis, tuple((1 for _ in self.axis))))[1:]
                        std_shape = tuple(val.shape[i] for i in range(len(val.shape)) if i not in inner_axis)
                        self.std = np.zeros(std_shape)
                        ex, ex2 = np.zeros(std_shape), np.zeros(std_shape)

                if val.ndim == val_0_ndim:
                    x = np.mean(val, axis=self.axis if inner_axis is None else inner_axis)
                    if n == 1:
                        k = x

                    ex += x - k
                    ex2 += (x - k) * (x - k)

                status_bar.update()
                if self.max_datapoints is not None:
                    if n == self.max_datapoints:
                        break

            var = (ex2 - (ex * ex) / n) / (n - 1)
            self.std = np.sqrt(var)

        logger.info(LoggingRecord(f"Standard deviation from {n} datapoints along axis {self.axis}: {self.std}"))

        return self.std
