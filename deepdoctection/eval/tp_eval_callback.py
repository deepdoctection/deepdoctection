# -*- coding: utf-8 -*-
# File: tp_eval_callback.py

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
Module for EvalCallback in Tensorpack
"""

from itertools import count
from typing import Dict, List, Optional, Type, Union

from ..datasets import DatasetBase
from ..extern.tpdetect import TPFrcnnDetector
from ..pipe.base import PredictorPipelineComponent
from ..utils.file_utils import tensorpack_available
from ..utils.logger import logger
from ..utils.metacfg import AttrDict
from .base import MetricBase
from .eval import Evaluator

# pylint: disable=import-error
if tensorpack_available():
    from tensorpack.callbacks import Callback
    from tensorpack.predict import OnlinePredictor
    from tensorpack.utils.gpu import get_num_gpu
# pylint: enable=import-error


# The following class is modified from
# https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/eval.py

__all__ = ["EvalCallback"]


class EvalCallback(Callback):  # type: ignore  # pylint: disable=R0903
    """
    A callback that runs evaluation once a while. It supports evaluation on any pipeline component.
    """

    _chief_only = False

    def __init__(
        self,
        dataset: DatasetBase,
        category_names: Optional[Union[str, List[str]]],
        sub_categories: Optional[Union[Dict[str, str], Dict[str, List[str]]]],
        metric: Type[MetricBase],
        pipeline_component: PredictorPipelineComponent,
        in_names: str,
        out_names: str,
        **build_eval_kwargs: str
    ) -> None:
        """
        :param dataset: dataset
        :param category_names: String or list of category names
        :param sub_categories: Dict of categories/sub-categories or categories/list of sub-categories. See also
                               :class:`eval.Evaluator`
        :param metric: metric
        :param pipeline_component: Pipeline component with a detector.
        :param in_names: Specify tensor input names.
                         E.g. :meth:`extern.tp.tpfrcnn.GeneralizedRCNN.get_inference_tensor_names`
        :param out_names: Specify tensor output names.
        :param build_eval_kwargs: Pass the necessary arguments in order to build the dataflow, e.g. "split",
                                  "build_mode", "max_datapoints" etc.
        """
        self.dataset_name = dataset.dataset_info.name
        self.build_eval_kwargs = build_eval_kwargs
        self.in_names, self.out_names = in_names, out_names
        assert hasattr(pipeline_component, "predictor"), "pipeline component must have a predictor"
        self.num_gpu = get_num_gpu()
        self.category_names = category_names
        self.sub_categories = sub_categories
        assert isinstance(pipeline_component.predictor, TPFrcnnDetector)
        self.cfg = pipeline_component.predictor.model.cfg
        if _use_replicated(self.cfg):
            self.evaluator = Evaluator(dataset, pipeline_component, metric, num_threads=self.num_gpu * 2)
        else:
            raise NotImplementedError("Can only evaluate in replicated training mode.")

    def _setup_graph(self) -> None:
        if _use_replicated(self.cfg):
            for idx, comp in enumerate(self.evaluator.pipe_component.pipe_components):
                comp.predictor.tp_predictor = self._build_predictor(idx % self.num_gpu)  # type: ignore

    def _build_predictor(self, idx: int) -> OnlinePredictor:
        return self.trainer.get_predictor(self.in_names, self.out_names, device=idx)

    def _before_train(self) -> None:
        eval_period = self.cfg.TRAIN.EVAL_PERIOD
        self.epochs_to_eval = set()
        for k in count(1):
            if k * eval_period > self.trainer.max_epoch:
                break
            self.epochs_to_eval.add(k * eval_period)
        self.epochs_to_eval.add(self.trainer.max_epoch)
        logger.info("[EvalCallback] Will evaluate every %i epochs", eval_period)

    def _eval(self) -> None:
        scores = self.evaluator.run(self.category_names, self.sub_categories, True, **self.build_eval_kwargs)
        assert isinstance(scores, dict)
        for k, val in scores.items():
            self.trainer.monitors.put_scalar(self.dataset_name + "-" + k, val)

    def _trigger_epoch(self) -> None:
        if self.epoch_num in self.epochs_to_eval:
            logger.info("Running evaluation ...")
            self._eval()


def _use_replicated(config: AttrDict) -> bool:
    if not hasattr(config, "TRAINER"):
        return False
    if config.TRAINER == "replicated":
        return True
    return False
