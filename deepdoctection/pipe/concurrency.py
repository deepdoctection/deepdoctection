# -*- coding: utf-8 -*-
# File: concurrency.py

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
Module for multithreading tasks
"""

import itertools
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Callable, List, Optional, Union

import tqdm  # type: ignore

from ..dataflow import DataFlow
from ..datapoint.image import Image
from ..utils.tqdm import get_tqdm
from .base import PipelineComponent


class MultiThreadPipelineComponent:
    """
    Running a pipeline component in multiple thread to increase through put. Datapoints will be queued
    and processed once calling the meth:`start`.

    The number of threads is derived from the list of pipeline components. It makes no sense to create the various
    components.

    Think of the pipeline component as an asynchronous process. Because the entire data flow is loaded into memory
    before the process is started, storage capacity must be guaranteed.

    If pre- and post-processing are to be carried out before the task within the wrapped pipeline component, this can
    also be transferred as a function. These tasks are also assigned to the threads.

    Note that the order in the dataflow and when returning lists is generally is no longer retained.
    """

    def __init__(
        self,
        pipeline_components: List[PipelineComponent],
        pre_proc_func: Optional[Callable[[Image], Image]] = None,
        post_proc_func: Optional[Callable[[Image], Image]] = None,
        max_datapoints: Optional[int] = None,
    ) -> None:

        """
        :param pipeline_components: list of identical pipeline component. Number of threads created is determined by
                                    :func:`len`
        :param pre_proc_func: pass a function, that reads and returns an image. Will execute before entering the pipe
                              component
        :param post_proc_func: pass a function, that reads and returns an image. Will execute after entering the pipe
                               component
        :param max_datapoints: max datapoints to process
        """

        self.pipe_components = pipeline_components
        self.input_queue: queue.Queue = queue.Queue()  # type: ignore
        self.pre_proc_func = pre_proc_func
        self.post_proc_func = post_proc_func
        self.max_datapoints = max_datapoints

    def put_task(self, df: Union[DataFlow, List[Image]]) -> None:
        """
        Put a dataflow or a list of datapoints to the queue. Note, that the process will not start before :meth:`start`
        is called. If you do not know how many datapoints will be cached, use max_datapoint to ensure no oom.

        :param df: A list or a dataflow of Image
        """

        self._put_datapoints_to_queue(df)

    def start(self) -> List[Image]:
        """
        Creates a worker for each component and starts processing the data points of the queue. A list of the results
        is returned once all points in the queue have been processed.

        :return: A list of Images
        """
        kwargs = {"thread_name_prefix": "EvalWorker"} if sys.version_info.minor >= 6 else {}
        with ThreadPoolExecutor(max_workers=len(self.pipe_components), **kwargs) as executor, tqdm.tqdm(  # type: ignore
            total=self.input_queue.qsize()
        ) as pbar:
            futures = []
            for component in self.pipe_components:
                futures.append(
                    executor.submit(
                        self._thread_predict_on_queue,
                        self.input_queue,
                        component,
                        pbar,
                        self.pre_proc_func,
                        self.post_proc_func,
                    )
                )
            all_results = list(itertools.chain(*[fut.result() for fut in futures]))
        return all_results

    @staticmethod
    def _thread_predict_on_queue(
        input_queue: queue.Queue,  # type: ignore
        component: PipelineComponent,
        tqdm_bar: Optional[tqdm.tqdm] = None,
        pre_proc_func: Optional[Callable[[Image], Image]] = None,
        post_proc_func: Optional[Callable[[Image], Image]] = None,
    ) -> List[Image]:

        outputs = []

        with ExitStack() as stack:
            if tqdm_bar is None:
                tqdm_bar = stack.enter_context(get_tqdm(total=input_queue.qsize()))

            while not input_queue.empty():
                inp = input_queue.get_nowait()
                if pre_proc_func is not None:
                    inp = pre_proc_func(inp)
                out = component.pass_datapoint(inp)
                if post_proc_func is not None:
                    out = post_proc_func(out)
                outputs.append(out)
                tqdm_bar.update(1)
        return outputs

    def _put_datapoints_to_queue(self, df: Union[DataFlow, List[Image]]) -> None:
        if isinstance(df, DataFlow):
            df.reset_state()
        for idx, dp in enumerate(df):
            if self.max_datapoints is not None:
                if self.max_datapoints >= idx:
                    break
            self.input_queue.put(dp)
