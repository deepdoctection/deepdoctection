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
Training Detectron2 `GeneralizedRCNN`
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Type, Union

from lazy_imports import try_import

from ..datasets.adapter import DatasetAdapter
from ..datasets.base import DatasetBase
from ..datasets.registry import get_dataset
from ..eval.base import MetricBase
from ..eval.eval import Evaluator
from ..eval.registry import metric_registry
from ..extern.d2detect import D2FrcnnDetector
from ..mapper.d2struct import image_to_d2_frcnn_training
from ..pipe.base import PipelineComponent
from ..pipe.registry import pipeline_component_registry
from ..utils.error import DependencyError
from ..utils.file_utils import get_wandb_requirement, wandb_available
from ..utils.logger import LoggingRecord, logger
from ..utils.types import PathLikeOrStr
from ..utils.utils import string_to_dict

with try_import() as d2_import_guard:
    from detectron2.config import CfgNode, get_cfg
    from detectron2.data import DatasetMapper, build_detection_train_loader
    from detectron2.data.transforms import RandomFlip, ResizeShortestEdge
    from detectron2.engine import DefaultTrainer, HookBase, default_writers, hooks
    from detectron2.utils import comm
    from detectron2.utils.events import EventWriter, get_event_storage
    from fvcore.nn.precise_bn import get_bn_modules  # type: ignore

with try_import() as pt_import_guard:
    from torch import cuda
    from torch.utils.data import DataLoader, IterableDataset

with try_import() as wb_import_guard:
    import wandb


def _set_config(
    path_config_yaml: PathLikeOrStr,
    conf_list: list[str],
    dataset_train: DatasetBase,
    dataset_val: Optional[DatasetBase],
    metric_name: Optional[str],
    metric: Optional[Union[Type[MetricBase], MetricBase]],
    pipeline_component_name: Optional[str],
) -> CfgNode:
    cfg = get_cfg()
    cfg.defrost()
    cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.01
    cfg.DATASETS.TRAIN = (dataset_train.dataset_info.name,)
    cfg.WANDB = CfgNode()
    cfg.WANDB.USE_WANDB = False
    cfg.WANDB.PROJECT = None
    cfg.WANDB.REPO = "deepdoctection"
    cfg.merge_from_file(path_config_yaml.as_posix() if isinstance(path_config_yaml, Path) else path_config_yaml)
    cfg.merge_from_list(conf_list)

    cfg.TEST.DO_EVAL = (
        cfg.TEST.EVAL_PERIOD > 0
        and dataset_val is not None
        and (metric_name is not None or metric is not None)
        and pipeline_component_name is not None
    )
    if cfg.TEST.DO_EVAL:
        cfg.DATASETS.TEST = (dataset_val.dataset_info.name,)  # type: ignore
    cfg.freeze()
    return cfg


def _update_for_eval(config_overwrite: list[str]) -> list[str]:
    ret = [item for item in config_overwrite if not "WANDB" in item]
    return ret


class WandbWriter(EventWriter):
    """
    Write all scalars to a wandb tool.
    """

    def __init__(
        self,
        project: str,
        repo: str,
        config: Optional[Union[dict[str, Any], CfgNode]] = None,
        window_size: int = 20,
        **kwargs: Any,
    ):
        """
        Args:
            project: W&B Project name.
            repo: Repository name.
            config: The project level configuration object.
            window_size: The scalars will be median-smoothed by this window size.
            **kwargs: Other arguments passed to `wandb.init(...)`.
        """
        if config is None:
            config = {}
        self._window_size = window_size
        self._run = wandb.init(project=project, config=config, **kwargs) if not wandb.run else wandb.run
        self._run._label(repo=repo)

    def write(self) -> None:
        storage = get_event_storage()

        log_dict = {}
        for key, (val, _) in storage.latest_with_smoothing_hint(self._window_size).items():
            log_dict[key] = val

        self._run.log(log_dict)

    def close(self) -> None:
        self._run.finish()


class D2Trainer(DefaultTrainer):
    """
    Detectron2 `DefaultTrainer` with some custom method for handling datasets and running evaluation.

    Info:
        The setting is made to train standard models in Detectron2.
    """

    def __init__(self, cfg: CfgNode, torch_dataset: IterableDataset[Any], mapper: DatasetMapper) -> None:
        self.dataset = torch_dataset
        self.mapper = mapper
        self.evaluator: Optional[Evaluator] = None
        self.build_val_dict: Mapping[str, str] = {}
        super().__init__(cfg)

    def build_hooks(self) -> list[HookBase]:
        """
        Builds the list of hooks for training.

        Note:
            This ensures that the `EvalHook` is being called before the writer and all metrics are being written to
            JSON, Tensorboard etc.

        Returns:
            List of `HookBase` objects.


        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            (
                hooks.PreciseBN(
                    # Run at the same freq as (but before) evaluation.
                    cfg.TEST.EVAL_PERIOD,
                    self.model,  # pylint: disable=E1101
                    # Build a new data loader to not affect training
                    self.build_train_loader(cfg),
                    cfg.TEST.PRECISE_BN.NUM_ITER,
                )
                if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)  # pylint: disable=E1101
                else None
            ),
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        if self.cfg.TEST.DO_EVAL:
            ret.append(
                hooks.EvalHook(
                    cfg.TEST.EVAL_PERIOD,
                    lambda: self.eval_with_dd_evaluator(**self.build_val_dict),  # pylint: disable=W0108
                )
            )

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))

        return ret

    def build_writers(self) -> list[EventWriter]:
        """
        Build a list of writers to be using `default_writers()`.

        Note:
            If you'd like a different list of writers, you can overwrite it in your trainer.

        Returns:
            A list of `EventWriter` objects.
        """
        writers_list = default_writers(self.cfg.OUTPUT_DIR, self.max_iter)
        if self.cfg.WANDB.USE_WANDB:
            _, _wandb_available, err_msg = get_wandb_requirement()
            if not _wandb_available:
                raise DependencyError(err_msg)
            if self.cfg.WANDB.PROJECT is None:
                raise ValueError("When using W&B, you must specify a project, i.e. WANDB.PROJECT")
            writers_list.append(WandbWriter(self.cfg.WANDB.PROJECT, self.cfg.WANDB.REPO, self.cfg))
        return writers_list

    def build_train_loader(self, cfg: CfgNode) -> DataLoader[Any]:  # pylint: disable=W0221
        """
        Builds the data loader for training.

        Args:
            cfg: Configuration.

        Returns:
            The data loader for a given dataset adapter and mapper.
        """
        return build_detection_train_loader(
            dataset=self.dataset, mapper=self.mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH
        )

    def eval_with_dd_evaluator(self, **build_eval_kwargs: str) -> Union[list[dict[str, Any]], dict[str, Any]]:
        """
        Runs the evaluator. This method will be called from the `EvalHook`.

        Args:
            **build_eval_kwargs: Dataflow eval config kwargs of the underlying dataset.

        Returns:
            A dict or list of dicts with evaluation results.
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
        pipeline_component: PipelineComponent,
        metric: Union[Type[MetricBase], MetricBase],
        build_val_dict: Optional[Mapping[str, str]] = None,
    ) -> None:
        """
        Setup of evaluator before starting training.

        Note:
            During training, predictors will be replaced by current checkpoints.

        Args:
            dataset_val: Dataset on which to run evaluation.
            pipeline_component: Pipeline component to plug into the evaluator.
            metric: A metric class or instance.
            build_val_dict: Evaluation dataflow build config.
        """
        if wandb_available():
            run = wandb.run if wandb.run is not None else None
        else:
            run = None
        self.evaluator = Evaluator(
            dataset_val,
            pipeline_component,
            metric,
            num_threads=cuda.device_count() * 2,
            run=run,
        )
        if build_val_dict:
            self.build_val_dict = build_val_dict
        assert self.evaluator.pipe_component
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.clear_predictor()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):  # type: ignore
        raise NotImplementedError()


def train_d2_faster_rcnn(
    path_config_yaml: PathLikeOrStr,
    dataset_train: Union[str, DatasetBase],
    path_weights: PathLikeOrStr,
    config_overwrite: Optional[list[str]] = None,
    log_dir: PathLikeOrStr = "train_log/frcnn",
    build_train_config: Optional[Sequence[str]] = None,
    dataset_val: Optional[DatasetBase] = None,
    build_val_config: Optional[Sequence[str]] = None,
    metric_name: Optional[str] = None,
    metric: Optional[Union[Type[MetricBase], MetricBase]] = None,
    pipeline_component_name: Optional[str] = None,
) -> None:
    """
    Adaptation of https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py for training Detectron2
    standard models.

    Trains Detectron2 from scratch or fine-tunes a model using this API.

    Info:
        This training script is devoted to the case where one cluster with one GPU is available. To run on several
        machines with more than one GPU use `detectron2.engine.launch`.

    Example:
        ```python
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
        ```

    Args:
        path_config_yaml: Path to a Detectron2 config file.
        dataset_train: The dataset to use for training.
        path_weights: Path to a checkpoint, if you want to continue training or fine-tune. Will train from scratch if
                      an empty string is passed.
        config_overwrite: List of arguments if some configs from the .yaml file should be replaced.
        log_dir: Path to log dir. Will default to `train_log/frcnn`.
        build_train_config: Dataflow build setting.
        dataset_val: The dataset to use for validation.
        build_val_config: Same as `build_train_config` but for validation.
        metric_name: A metric name to choose for validation.
        metric: A metric to choose for validation.
        pipeline_component_name: A pipeline component name to use for validation.
    """

    assert cuda.device_count() > 0, "Has to train with GPU!"

    build_train_dict: dict[str, str] = {}
    if build_train_config is not None:
        build_train_dict = string_to_dict(",".join(build_train_config))
    if "split" not in build_train_dict:
        build_train_dict["split"] = "train"

    build_val_dict: dict[str, str] = {}
    if build_val_config is not None:
        build_val_dict = string_to_dict(",".join(build_val_config))
    if "split" not in build_val_dict:
        build_val_dict["split"] = "val"

    if config_overwrite is None:
        config_overwrite = []
    conf_list = [
        "MODEL.WEIGHTS",
        os.fspath(path_weights),
        "OUTPUT_DIR",
        os.fspath(log_dir),
    ]
    for conf in config_overwrite:
        key, val = conf.split("=", maxsplit=1)
        conf_list.extend([key, val])

    if isinstance(dataset_train, str):
        dataset_train = get_dataset(dataset_train)

    cfg = _set_config(
        path_config_yaml, conf_list, dataset_train, dataset_val, metric_name, metric, pipeline_component_name
    )

    if metric_name is not None:
        metric = metric_registry.get(metric_name)

    dataset = DatasetAdapter(
        dataset_train, True, image_to_d2_frcnn_training(False), True, number_repetitions=-1, **build_train_dict
    )
    augment_list = [ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN), RandomFlip()]
    mapper = DatasetMapper(is_train=True, augmentations=augment_list, image_format="BGR")

    logger.info(LoggingRecord(f"Config: \n {str(cfg)}", dict(cfg)))

    trainer = D2Trainer(cfg, dataset, mapper)
    trainer.resume_or_load()

    if cfg.TEST.DO_EVAL:
        categories = dataset_val.dataflow.categories.get_categories(filtered=True)  # type: ignore
        config_overwrite = _update_for_eval(config_overwrite)
        detector = D2FrcnnDetector(path_config_yaml, path_weights, categories, config_overwrite, cfg.MODEL.DEVICE)
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        pipeline_component = pipeline_component_cls(detector)

        if metric_name is not None:
            metric = metric_registry.get(metric_name)
        assert metric is not None

        trainer.setup_evaluator(dataset_val, pipeline_component, metric, build_val_dict)  # type: ignore
    return trainer.train()
