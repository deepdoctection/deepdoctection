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
from __future__ import annotations

import itertools
import queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from typing import Callable, Optional, Sequence, Union

import tqdm

from ..dataflow import DataFlow, MapData
from ..datapoint.image import Image, MetaAnnotation
from ..utils.context import timed_operation
from ..utils.tqdm import get_tqdm
from ..utils.types import QueueType, TqdmType
from .base import PipelineComponent
from .common import ImageParsingService, PageParsingService
from .registry import pipeline_component_registry


@pipeline_component_registry.register("ImageCroppingService")
class MultiThreadPipelineComponent(PipelineComponent):
    """
    This module provides functionality for running pipeline components in multiple threads to increase throughput.
    Datapoints are queued and processed once calling the `start` method.

    Note:
        The number of threads is derived from the list of `pipeline_components`. It makes no sense to create various
        components.

        Think of the pipeline component as an asynchronous process. Because the entire data flow is loaded into memory
        before the process is started, storage capacity must be guaranteed.

        If pre- and post-processing are to be carried out before the task within the wrapped pipeline component, this
        can also be transferred as a function. These tasks are also assigned to the threads.

        The order in the dataflow and when returning lists is generally no longer retained.

    Example:
        ```python
        some_component = SubImageLayoutService(some_predictor, some_category)
        some_component_clone = some_component.clone()

        multi_thread_comp = MultiThreadPipelineComponent(
            pipeline_components=[some_component, some_component_clone],
            pre_proc_func=maybe_load_image,
            post_proc_func=maybe_remove_image
        )

        multi_thread_comp.put_task(some_dataflow)
        output_list = multi_thread_comp.start()
        ```

    Info:
        You cannot run `MultiThreadPipelineComponent` in `DoctectionPipe` as this requires batching datapoints and
        neither can you run `MultiThreadPipelineComponent` in combination with a humble `PipelineComponent` unless you
        take care of batching/unbatching between each component by yourself. The easiest way to build a pipeline with
        `MultiThreadPipelineComponent` can be accomplished as follows:

    Example:
        ```python
        # define the pipeline component
        some_component = SubImageLayoutService(some_predictor, some_category)
        some_component_clone = some_component.clone()

        # creating two threads, one for each component
        multi_thread_comp = MultiThreadPipelineComponent(
            pipeline_components=[some_component, some_component_clone],
            pre_proc_func=maybe_load_image,
            post_proc_func=maybe_remove_image
        )

        # currying `to_image`, so that you can call it in `MapData`.
        @curry
        def _to_image(dp, dpi):
            return to_image(dp, dpi)

        # set-up the dataflow/stream, e.g.
        df = SerializerPdfDoc.load(path, max_datapoints=max_datapoints)
        df = MapData(df, to_image(dpi=300))
        df = BatchData(df, batch_size=32, remainder=True)
        df = multi_thread_comp.predict_dataflow(df)
        df = FlattenData(df)
        df = MapData(df, lambda x: x[0])

        df.reset_state()

        for dp in df:
            ...
        ```
    """

    def __init__(
        self,
        pipeline_components: Sequence[Union[PipelineComponent, ImageParsingService]],
        pre_proc_func: Optional[Callable[[Image], Image]] = None,
        post_proc_func: Optional[Callable[[Image], Image]] = None,
        max_datapoints: Optional[int] = None,
    ) -> None:
        """
        Args:
            pipeline_components: List of identical `PipelineComponent`. Number of threads created is determined by
                                 `len`.
            pre_proc_func: Function that reads and returns an image. Will execute before entering the pipe component.
            post_proc_func: Function that reads and returns an image. Will execute after entering the pipe component.
            max_datapoints: Maximum datapoints to process.
        """

        self.pipe_components = pipeline_components
        self.input_queue: QueueType = queue.Queue()
        self.pre_proc_func = pre_proc_func
        self.post_proc_func = post_proc_func
        self.max_datapoints = max_datapoints
        self.timer_on = False
        super().__init__(f"multi_thread_{self.pipe_components[0].name}")

    def put_task(self, df: Union[DataFlow, list[Image]]) -> None:
        """
        Put a `DataFlow` or a list of datapoints to the queue.

        Note:
            The process will not start before `start` is called. If you do not know how many datapoints will be cached,
            use `max_datapoints` to ensure no OOM.

        Args:
            df: A list or a `DataFlow` of `Image`.
        """

        self._put_datapoints_to_queue(df)

    def start(self) -> list[Image]:
        """
        Creates a worker for each component and starts processing the datapoints of the queue.

        Example:
            ```python
            output_list = multi_thread_comp.start()
            ```

        Returns:
            A list of `Image` objects.
        """
        with ThreadPoolExecutor(
            max_workers=len(self.pipe_components), thread_name_prefix="EvalWorker"
        ) as executor, tqdm.tqdm(total=self.input_queue.qsize()) as pbar:
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
        input_queue: QueueType,
        component: Union[PipelineComponent, PageParsingService, ImageParsingService],
        tqdm_bar: Optional[TqdmType] = None,
        pre_proc_func: Optional[Callable[[Image], Image]] = None,
        post_proc_func: Optional[Callable[[Image], Image]] = None,
    ) -> list[Image]:
        outputs = []

        with ExitStack() as stack:
            if tqdm_bar is None:
                tqdm_bar = stack.enter_context(get_tqdm(total=input_queue.qsize()))

            while not input_queue.empty():
                inp = input_queue.get_nowait()
                if pre_proc_func is not None:
                    inp = pre_proc_func(inp)
                out = component.pass_datapoint(inp)
                if post_proc_func is not None and out is not None:
                    out = post_proc_func(out)
                if out is not None:
                    outputs.append(out)
                tqdm_bar.update(1)
        return outputs

    def _put_datapoints_to_queue(self, df: Union[DataFlow, list[Image]]) -> None:
        if isinstance(df, DataFlow):
            df.reset_state()
        for idx, dp in enumerate(df):
            if self.max_datapoints is not None:
                if self.max_datapoints >= idx:
                    break
            self.input_queue.put(dp)

    def pass_datapoints(self, dpts: list[Image]) -> list[Image]:
        """
        Put the list of datapoints into a thread-safe queue and start a separate thread for each pipeline component.

        The order of appearance of the output might not be the same as the input.

        Args:
            dpts: List of `Image` datapoints.

        Returns:
            List of processed `Image` datapoints.
        """
        for dp in dpts:
            self.input_queue.put(dp)
        if self.timer_on:
            with timed_operation(self.pipe_components[0].name):
                dpts = self.start()
        else:
            dpts = self.start()
        return dpts

    def predict_dataflow(self, df: DataFlow) -> DataFlow:
        """
        Map a datapoint via `pass_datapoints` within a dataflow pipeline.

        Args:
            df: An input `DataFlow`.

        Returns:
            An output `DataFlow`.
        """
        return MapData(df, self.pass_datapoints)

    def serve(self, dp: Image) -> None:
        raise NotImplementedError("MultiThreadPipelineComponent does not follow the PipelineComponent implementation")

    def clone(self) -> MultiThreadPipelineComponent:
        raise NotImplementedError("MultiThreadPipelineComponent does not allow cloning")

    def get_meta_annotation(self) -> MetaAnnotation:
        return self.pipe_components[0].get_meta_annotation()

    def clear_predictor(self) -> None:
        for pipe in self.pipe_components:
            pipe.clear_predictor()
