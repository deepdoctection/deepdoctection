# -*- coding: utf-8 -*-
# File: d2_frcnn_train.py

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
Module for training Detectron2 GeneralizedRCNN
"""


import copy
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from detectron2.config import CfgNode, get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.transforms import RandomFlip, ResizeShortestEdge
from detectron2.engine import DefaultTrainer, EvalHook
from torch.utils.data import DataLoader, IterableDataset

from ..datasets.adapter import DatasetAdapter
from ..datasets.base import DatasetBase
from ..eval.base import MetricBase
from ..eval.eval import Evaluator
from ..eval.registry import metric_registry
from ..extern.d2detect import D2FrcnnDetector
from ..extern.pt.ptutils import get_num_gpu
from ..mapper.d2struct import image_to_d2_frcnn_training
from ..pipe.base import PredictorPipelineComponent
from ..pipe.registry import pipeline_component_registry
from ..utils.logger import logger
from ..utils.utils import string_to_dict


def _set_config(path_config_yaml: str, conf_list: List[str]) -> CfgNode:
    cfg = get_cfg()
    cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.01
    cfg.merge_from_file(path_config_yaml)
    cfg.merge_from_list(conf_list)
    cfg.freeze()
    return cfg


class D2Trainer(DefaultTrainer):
    """
    Detectron2 DefaultTrainer with some custom method for handling datasets and running evaluation. The setting is made
    to train standard models in detectron2.
    """

    def __init__(self, cfg: CfgNode, torch_dataset: IterableDataset[Any], mapper: DatasetMapper) -> None:
        self.dataset = torch_dataset
        self.mapper = mapper
        self.evaluator: Optional[Evaluator] = None
        super().__init__(cfg)

    def build_train_loader(self, cfg: CfgNode) -> DataLoader[Any]:  # pylint: disable=W0221
        """
        Overwritten method from DefaultTrainer.

        :param cfg: Configuration
        :return: The data loader for a given dataset adapter, mapper.
        """
        return build_detection_train_loader(
            dataset=self.dataset, mapper=self.mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH
        )

    def eval_with_dd_evaluator(self, **build_eval_kwargs: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Running the Evaluator. This method will be called from the EvalHook

        :param build_eval_kwargs: dataflow eval config kwargs of the underlying dataset
        :return: A dict of evaluation results
        """
        assert self.evaluator is not None
        assert self.evaluator.pipe_component is not None
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.predictor.d2_predictor = copy.deepcopy(self.model).eval()  # type: ignore # pylint: disable=E1101
        scores = self.evaluator.run(True, **build_eval_kwargs)
        return scores

    def setup_evaluator(
        self,
        dataset_val: DatasetBase,
        pipeline_component: PredictorPipelineComponent,
        metric: Union[Type[MetricBase], MetricBase],
    ) -> None:
        """
        Setup of evaluator before starting training. During training, predictors will be replaced by current
        checkpoints.

        :param dataset_val: dataset on which to run evaluation
        :param pipeline_component: pipeline component to plug into the evaluator
        :param metric: A metric class
        """
        self.evaluator = Evaluator(dataset_val, pipeline_component, metric, num_threads=get_num_gpu() * 2)
        assert self.evaluator.pipe_component
        for comp in self.evaluator.pipe_component.pipe_components:
            assert isinstance(comp, PredictorPipelineComponent)
            assert isinstance(comp.predictor, D2FrcnnDetector)
            comp.predictor.d2_predictor = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):  # type: ignore
        raise NotImplementedError


def train_d2_faster_rcnn(
    path_config_yaml: str,
    dataset_train: Union[str, DatasetBase],
    path_weights: str,
    config_overwrite: Optional[List[str]] = None,
    log_dir: str = "train_log/frcnn",
    build_train_config: Optional[Sequence[str]] = None,
    dataset_val: Optional[DatasetBase] = None,
    build_val_config: Optional[Sequence[str]] = None,
    metric_name: Optional[str] = None,
    metric: Optional[Union[Type[MetricBase], MetricBase]] = None,
    pipeline_component_name: Optional[str] = None,
) -> None:
    """
    Adaptation of https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py for training Detectron2
    standard models

    Train Detectron2 from Scratch or fine-tune a model using this API. Compared to Tensorpack this framework trains much
    faster, e.g. https://detectron2.readthedocs.io/en/latest/notes/benchmarks.html .

    This training script is devoted to the case where one cluster with one GPU is available. To run on several machines
    with more than one GPU use :func:`detectron2.engine.launch` .

    .. code-block:: python

        if __name__ == "__main__":

                launch(train_d2_faster_rcnn,
                       num_gpus,
                       num_machines,
                       machine_rank,
                       dist_url,
                       args=(path_config_yaml,
                             path_weights,
                             config_overwrite,
                             log_dir,
                             build_train_config,
                             dataset_val,
                             build_val_config,
                             metric_name,
                             metric,
                             pipeline_component_name),)


    :param path_config_yaml: path to a D2 config file. Check
                             https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
                             for various settings.
    :param dataset_train: the dataset to use for training.
    :param path_weights: path to a checkpoint, if you want to continue training or fine-tune. Will train from scratch if
                         nothing is passed.
    :param config_overwrite: Pass a list of arguments if some configs from the .yaml file should be replaced. Use the
                             list convention, e.g. ['TRAIN.STEPS_PER_EPOCH=500', 'OUTPUT.RESULT_SCORE_THRESH=0.4']
    :param log_dir: Path to log dir. Will default to `train_log/frcnn`
    :param build_train_config: dataflow build setting. Again, use list convention setting, e.g. ['max_datapoints=1000']
    :param dataset_val: the dataset to use for validation.
    :param build_val_config: same as `build_train_config` but for validation
    :param metric_name: A metric name to choose for validation. Will use the default setting. If you want a custom
                        metric setting pass a metric explicitly.
    :param metric: A metric to choose for validation.
    :param pipeline_component_name: A pipeline component name to use for validation.
    """

    assert get_num_gpu() > 0, "Has to train with GPU!"

    build_train_dict: Dict[str, str] = {}
    if build_train_config is not None:
        build_train_dict = string_to_dict(",".join(build_train_config))
    if "split" not in build_train_dict:
        build_train_dict["split"] = "train"

    build_val_dict: Dict[str, str] = {}
    if build_val_config is not None:
        build_val_dict = string_to_dict(",".join(build_val_config))
    if "split" not in build_val_dict:
        build_val_dict["split"] = "val"

    if config_overwrite is None:
        config_overwrite = []
    conf_list = ["MODEL.WEIGHTS", path_weights, "OUTPUT_DIR", log_dir]
    for conf in config_overwrite:
        key, val = conf.split("=", maxsplit=1)
        conf_list.extend([key, val])
    cfg = _set_config(path_config_yaml, conf_list)

    if metric_name is not None:
        metric = metric_registry.get(metric_name)

    dataset = DatasetAdapter(dataset_train, True, image_to_d2_frcnn_training(False), **build_train_dict)
    augment_list = [ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN), RandomFlip()]
    mapper = DatasetMapper(is_train=True, augmentations=augment_list, image_format="BGR")

    logger.info("Config: \n %s", str(cfg), cfg)
    trainer = D2Trainer(cfg, dataset, mapper)
    trainer.resume_or_load()

    if (
        cfg.TEST.EVAL_PERIOD > 0
        and dataset_val is not None
        and (metric_name is not None or metric is not None)
        and pipeline_component_name is not None
    ):
        categories = dataset_val.dataflow.categories.get_categories(filtered=True)
        detector = D2FrcnnDetector(path_config_yaml, path_weights, categories, config_overwrite, cfg.MODEL.DEVICE)
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        pipeline_component = pipeline_component_cls(detector)
        assert isinstance(pipeline_component, PredictorPipelineComponent)

        if metric_name is not None:
            metric = metric_registry.get(metric_name)
        assert metric is not None

        trainer.setup_evaluator(dataset_val, pipeline_component, metric)
        trainer.register_hooks(
            [
                EvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    lambda: trainer.eval_with_dd_evaluator(**build_val_dict),  # pylint: disable=W0108
                )
            ]
        )
    return trainer.train()
