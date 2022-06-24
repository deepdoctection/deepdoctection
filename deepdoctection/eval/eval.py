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

from copy import deepcopy
from typing import Any, Dict, List, Type, Union, Optional

from ..dataflow import DataFromList, MapData, DataFlow, CacheData
from ..datasets.base import DatasetBase
from ..mapper.cats import remove_cats, filter_cat
from ..mapper.misc import maybe_load_image, maybe_remove_image
from ..pipe.base import PredictorPipelineComponent
from ..pipe.doctectionpipe import DoctectionPipe
from ..pipe.concurrency import MultiThreadPipelineComponent
from ..utils.logger import logger
from .base import MetricBase


class Evaluator:  # pylint: disable=R0903
    """
    The API for evaluating pipeline components or pipelines on a given dataset. For a given model, a given dataset and
    a given metric, this class will stream the dataset, call the predictor(s) and will evaluate the predictions against
    the ground truth with respect to the given metric.

    After initializing the evaluator the process itself will start after calling the meth:`run`.

    The following takes place under the hood:

    Setup of the dataflow according to the build- and split inputs. The meth:`datasets.DataFlowBaseBuilder.build` will
    be invoked twice as one dataflow must be kept with its ground truth while the other must go through an annotation
    erasing process and after that passing the predictor. Predicted and gt datapoints will be converted into the
    required metric input format and dumped into lists. Both lists will be passed to :meth:`MetricBase.get_distance`.

    **Example:**

        You can evaluate the predictor on a subset of categories by filtering the ground truth dataset. When using
        the coco metric all predicted objects that are not in the set of filtered objects will be not taken into
        account.

        .. code-block:: python

            publaynet = get_dataset("publaynet")
            publaynet.dataflow.categories.filter_categories(categories=["TEXT","TITLE"])
            coco_metric = metric_registry.get("coco")
            profile = ModelCatalog.get_profile("layout/d2_model_0829999_layout_inf_only.pt")
            path_weights = ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout_inf_only.pt")
            path_config_yaml= ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout_inf_only.pt")

            layout_detector = D2FrcnnDetector(path_config_yaml, path_weights, profile.categories)
            layout_service = ImageLayoutService(layout_detector)
            evaluator = Evaluator(publaynet, layout_service, coco_metric)

            output = evaluator.run(max_datapoints=10)

    """

    def __init__(
        self,
        dataset: DatasetBase,
        component_or_pipeline: Union[PredictorPipelineComponent, DoctectionPipe],
        metric: Type[MetricBase],
        num_threads: int = 2,
    ) -> None:
        """
        Evaluating a pipeline component on a dataset with a given metric.

        :param dataset: dataset
        :param component_or_pipeline: A pipeline component with predictor and annotation factory.
        :param metric: metric
        """


        self.dataset = dataset
        self.pipe_component: Optional[MultiThreadPipelineComponent] = None
        self.pipe: Optional[DoctectionPipe] = None

        # when passing a component, we will process prediction on num_threads
        if isinstance(component_or_pipeline, PredictorPipelineComponent):
            logger.info(
                "Building multi threading pipeline component to increase prediction throughput. Using %i threads",
                num_threads,
            )
            pipeline_components: List[PredictorPipelineComponent] = []

            for _ in range(num_threads - 1):
                copy_pipe_component = component_or_pipeline.clone()
                pipeline_components.append(copy_pipe_component)

            pipeline_components.append(component_or_pipeline)

            self.pipe_component = MultiThreadPipelineComponent(
                pipeline_components=pipeline_components,
                pre_proc_func=maybe_load_image,
                post_proc_func=maybe_remove_image,
            )
        else:
            self.pipe = component_or_pipeline
            for component in self.pipe.pipe_component_list:
                component.timer_on = False

        self.metric = metric()
        self._sanity_checks()

    def run(self, output_as_dict: bool = False, **kwargs: Union[str,int]) -> \
            Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Start evaluation process and return the results.

        :param output_as_dict: Return result in a list or dict.
        :param kwargs: Pass the necessary arguments in order to build the dataflow, e.g. "split", "build_mode",
                       "max_datapoints" etc.

        :return: dict with metric results.
        """

        assert self.dataset.dataflow.categories is not None, "dataset requires dataflow.categories to be not None"
        df_gt = self.dataset.dataflow.build(**kwargs)
        df_pr = self.dataset.dataflow.build(**kwargs)

        df_pr = MapData(df_pr, deepcopy)
        df_pr = self._clean_up_predict_dataflow_annotations(df_pr)
        df_pr = self._run_pipe_or_component(df_pr)

        logger.info("Starting evaluation...")
        result = self.metric.get_distance(df_gt, df_pr, self.dataset.dataflow.categories)

        if output_as_dict:
            return self.metric.result_list_to_dict(result)

        return result

    def _sanity_checks(self) -> None:
        assert self.dataset.dataflow.categories is not None

    def _run_pipe_or_component(self, df_pr: DataFlow) -> DataFlow:
        if self.pipe_component:
            self.pipe_component.put_task(df_pr)
            logger.info("Predicting objects...")
            df_pr_list = self.pipe_component.start()
            return DataFromList(df_pr_list)
        df_pr = MapData(df_pr,maybe_load_image)
        df_pr = self.pipe.analyze(dataset_dataflow=df_pr, output="image")
        df_pr = MapData(df_pr,maybe_remove_image)
        df_list = CacheData(df_pr).get_cache()
        return DataFromList(df_list)

    def _clean_up_predict_dataflow_annotations(self, df_pr: DataFlow) -> DataFlow:
        # will use the first pipe component of the multi thread component
        pipe_or_component = self.pipe_component.pipe_components[0] if self.pipe_component is not None else self.pipe
        meta_anns = pipe_or_component.get_meta_annotation()
        possible_cats_in_datapoint = self.dataset.dataflow.categories.get_categories(as_dict=False, filtered=True)
        anns_to_keep = {ann for ann in possible_cats_in_datapoint if ann not in meta_anns["image_annotations"]}
        sub_cats_to_remove = meta_anns["sub_categories"]
        df_pr = MapData(df_pr,filter_cat(anns_to_keep , possible_cats_in_datapoint))
        df_pr = MapData(df_pr,remove_cats(sub_categories=sub_cats_to_remove))
        return df_pr



