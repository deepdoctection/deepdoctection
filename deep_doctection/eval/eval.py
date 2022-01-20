# -*- coding: utf-8 -*-
# File: eval.py

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
Module for :class:`Evaluator`
"""

__all__ = ["Evaluator"]

from typing import Optional, Union, List, Dict, Any, Type
from copy import copy
from ..datasets.base import DatasetBase
from .base import MetricBase
from ..pipe.base import PredictorPipelineComponent, PipelineComponent
from ..pipe.concurrency import MultiThreadPipelineComponent
from ..mapper.cats import remove_cats
from ..utils.logger import logger
from ..utils.file_utils import tensorpack_available
from ..dataflow import MapData, DataFromList  # type: ignore
from ..mapper.misc import maybe_load_image, maybe_remove_image

if tensorpack_available():
    from ..extern.tpdetect import TPFrcnnDetector


class Evaluator:  # pylint: disable=R0903
    """
    The API for evaluating pipeline components on a given dataset. For a given model, a given dataset and a given
    metric, this class will stream the dataset, call the predictor and will evaluate the predictions against the ground
    truth with respect to the given metric.

    After initializing the evaluator the process itself will start after calling the meth:`run`.

    The following takes place under the hood:

    Setup of the dataflow according to the build- and split inputs. The meth:`datasets.DataFlowBaseBuilder.build` will
    be invoked twice as one dataflow must be kept with its ground truth while the other must go through an annotation
    erasing process and after that passing the predictor. Predicted and gt datapoints will be converted into the
    required metric input format and dumped into lists. Both lists will be passed to :meth:`MetricBase.get_distance`.
    """

    def __init__(
        self,
        dataset: DatasetBase,
        predictor_pipe_component: PredictorPipelineComponent,
        metric: Type[MetricBase],
        num_threads: int = 2,
    ) -> None:
        """
        Evaluating a pipeline component on a dataset with a given metric.

        :param dataset: dataset
        :param predictor_pipe_component: A pipeline component with predictor and annotation factory.
        :param metric: metric
        """

        logger.info(
            "Building multi threading pipeline component to increase prediction throughput. Using %i threads",
            num_threads,
        )
        pipeline_components: List[PipelineComponent] = []

        # try to copy as little as possible
        # TODO: remove copying procedure outside class
        assert isinstance(predictor_pipe_component.predictor, TPFrcnnDetector)
        tmp_tp_predictor = predictor_pipe_component.predictor.tp_predictor
        tmp_predictor = predictor_pipe_component.predictor
        tmp_dp_manager = predictor_pipe_component.dp_manager
        predictor_pipe_component.dp_manager = None  # type: ignore
        predictor_pipe_component.predictor = None  # type: ignore
        tmp_predictor.tp_predictor = None
        for _ in range(num_threads - 1):
            copy_pipe_component = copy(predictor_pipe_component)
            copy_pipe_component.predictor = copy(tmp_predictor)
            copy_pipe_component.predictor.tp_predictor = copy(tmp_tp_predictor)
            copy_pipe_component.dp_manager = copy(tmp_dp_manager)
            pipeline_components.append(copy_pipe_component)

        predictor_pipe_component.dp_manager = tmp_dp_manager
        predictor_pipe_component.predictor = tmp_predictor
        predictor_pipe_component.predictor.tp_predictor = tmp_tp_predictor
        pipeline_components.append(predictor_pipe_component)
        self.dataset = dataset

        self.pipe_component = MultiThreadPipelineComponent(
            pipeline_components=pipeline_components,
            pre_proc_func=maybe_load_image,
            post_proc_func=maybe_remove_image,
        )
        self.metric = metric
        self._sanity_checks()

    def run(
        self,
        category_names: Optional[Union[str, List[str]]] = None,
        sub_categories: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
        output_as_dict: bool = False,
        **kwargs: str
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Start evaluation process and return the results.

        :param category_names: Single or list of categories on which evaluation should run. Pass nothing to do
                               something else.
        :param sub_categories: Dict of categories/sub-categories or categories/ list of sub categories.
        :param output_as_dict: Return result in a list or dict.
        :param kwargs: Pass the necessary arguments in order to build the dataflow, e.g. "split", "build_mode",
                       "max_datapoints" etc.

        :return: dict with metric results.
        """

        assert self.dataset.dataflow.categories is not None, "dataset requires dataflow.categories to be not None"
        df_gt = self.dataset.dataflow.build(**kwargs)
        df_pr = self.dataset.dataflow.build(**kwargs)

        remove = remove_cats(category_names, sub_categories)  # type: ignore
        df_pr = MapData(df_pr, remove)
        self.pipe_component.put_task(df_pr)

        logger.info("Predicting objects...")
        df_pr_list = self.pipe_component.start()

        df_pr = DataFromList(df_pr_list)

        logger.info("Starting evaluation...")
        output = self.metric.get_distance(df_gt, df_pr, self.dataset.dataflow.categories, output_as_dict)

        return output

    def _sanity_checks(self) -> None:
        assert self.dataset.dataflow.categories is not None
