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
from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from dataflow.dataflow import DataFlow, ProxyDataFlow  # type: ignore

from ..utils.logger import logger
from ..utils.tqdm import get_tqdm


class MeanFromDataFlow(ProxyDataFlow):  # type: ignore
    """
    Gets the mean of some dataflow. Takes a component from a dataflow and calculates iteratively the mean.

    **Example:**

        .. code-block:: python

            df: some dataflow
            MeanFromDataFlow(df).start() if you want to put MeanFromDataFlow at the end of a dataflow
            df: some dataflow
            df = MeanFromDataFlow(df)

        is also possible. Testing with the progress bar will stop once the requested size has been reached.
    """

    def __init__(
        self,
        df: DataFlow,
        axis: Optional[Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]] = None,
        key: Optional[str] = None,
        max_datapoints: Optional[int] = None,
    ):
        """
        :param df: the dataflow to test
        :param axis: The axis along which to calculate the mean. It will always calculate the mean along
                     the dataflow length axis, which is the 0th axis. E.g. for calculating the mean of an image dataset
                     use

                     .. code-block:: python

                         MeanFromDataFlow(df,key="image",axis=(0,1,2)).start()

        :param key: The datapoint key giving the values to evaluate the mean. If None it tries to use
                    the whole datapoint.
        :param max_datapoints: Will stop considering datapoints with index>max_datapoints.
        """

        super().__init__(df)
        if isinstance(axis, int):
            axis = (axis,)
        if axis is not None:
            assert 0 in axis, "Requires 0-th axis for calculating mean"
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
        yield from self.ds

    def start(self) -> npt.NDArray[Any]:
        """
        Start calculating the mean with a progress bar.
        """

        if not self._reset_called:
            self.ds.reset_state()
        itr = self.ds.__iter__()

        logger.info("____________________ CALCULATING MEAN ____________________")

        len_df: Optional[int]
        try:
            len_df = len(self.ds)
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

        logger.info("Mean from %s datapoints along axis %s: %s", n, self.axis, self.mean)

        return self.mean


class StdFromDataFlow(ProxyDataFlow):  # type: ignore
    """
    Gets the standard deviation of some dataflow. Takes a component from a dataflow and calculates iteratively
    the standard deviation.

    **Example:**

        .. code-block:: python

            df= ...
            StdFromDataFlow(df).start()

        if you want to put  StdFromDataFlow at the end of a dataflow

         .. code-block:: python

            df: some dataflow
            df = StdFromDataFlow(df)

        is also possible. The testing with the progress bar will stop once the requested size has been reached.
    """

    def __init__(
        self,
        df: DataFlow,
        axis: Optional[Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]] = None,
        key: Optional[str] = None,
        max_datapoints: Optional[int] = None,
    ):
        """
        :param df: the dataflow to test
        :param axis: The axis along which to calculate the mean. It will always calculate the std along
                     the dataflow length axis, which is the 0th axis. E.g. for calculating the mean of an image dataset
                     use

                     .. code-block:: python

                         StdFromDataFlow(df,key="image",axis=(0,1,2)).start()

        :param key: The datapoint key giving the values to evaluate the std. If None it tries to use
                    the whole datapoint.
        :param max_datapoints: Will stop considering datapoints with index>max_datapoints.
        """

        super().__init__(df)
        if isinstance(axis, int):
            axis = (axis,)
        if axis is not None:
            assert 0 in axis, "Requires 0-th axis for calculating mean"
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
        yield from self.ds

    def start(self) -> npt.NDArray[Any]:
        """
        Start calculating the mean with a progress bar.
        """
        len_df: Optional[int]

        if not self._reset_called:
            self.ds.reset_state()
        itr = self.ds.__iter__()

        logger.info("____________________ CALCULATING STANDARD DEVIATION ____________________")
        try:
            len_df = len(self.ds)
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

        logger.info("Standard deviation from %s datapoints along axis %s: %s", n, self.axis, self.std)

        return self.std
