# -*- coding: utf-8 -*-
# File: tp_frcnn_train.py

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
Training Tensorpack's `GeneralizedRCNN`
"""

import os
from typing import Optional, Sequence, Type, Union

from lazy_imports import try_import

from ..dataflow.base import DataFlow
from ..dataflow.common import MapData
from ..dataflow.parallel_map import MultiProcessMapData
from ..dataflow.serialize import DataFromList
from ..datasets.base import DatasetBase
from ..eval.base import MetricBase
from ..eval.registry import metric_registry
from ..eval.tp_eval_callback import EvalCallback
from ..extern.tp.tfutils import disable_tfv2
from ..extern.tp.tpfrcnn.common import CustomResize
from ..extern.tp.tpfrcnn.config.config import model_frcnn_config, train_frcnn_config
from ..extern.tp.tpfrcnn.modeling.generalized_rcnn import ResNetFPNModel
from ..extern.tp.tpfrcnn.preproc import anchors_and_labels, augment
from ..extern.tpdetect import TPFrcnnDetector
from ..mapper.maputils import LabelSummarizer
from ..mapper.tpstruct import image_to_tp_frcnn_training
from ..pipe.registry import pipeline_component_registry
from ..utils.file_utils import set_mp_spawn
from ..utils.fs import get_load_image_func
from ..utils.logger import log_once
from ..utils.metacfg import AttrDict, set_config_by_yaml
from ..utils.tqdm import get_tqdm
from ..utils.types import JsonDict, PathLikeOrStr
from ..utils.utils import string_to_dict

with try_import() as tp_import_guard:
    # todo: check how dataflow import is directly possible without having an AssertionError
    # pylint: disable=import-error
    from tensorpack.callbacks import (
        EstimatedTimeLeft,
        GPUMemoryTracker,
        GPUUtilizationTracker,
        HostMemoryTracker,
        ModelSaver,
        PeriodicCallback,
        ScheduledHyperParamSetter,
        SessionRunTimeout,
        ThroughputTracker,
    )
    from tensorpack.dataflow import ProxyDataFlow, imgaug
    from tensorpack.input_source import QueueInput
    from tensorpack.tfutils import SmartInit
    from tensorpack.train import SyncMultiGPUTrainerReplicated, TrainConfig, launch_train_with_config
    from tensorpack.utils import logger

__all__ = ["train_faster_rcnn"]


class LoadAugmentAddAnchors:
    """
    A helper class for default mapping `load_augment_add_anchors`.

    Args:
        config: An `AttrDict` configuration for TP FRCNN.
    """

    def __init__(self, config: AttrDict) -> None:
        self.cfg = config

    def __call__(self, dp: JsonDict) -> Optional[JsonDict]:
        return load_augment_add_anchors(dp, self.cfg)


def load_augment_add_anchors(dp: JsonDict, config: AttrDict) -> Optional[JsonDict]:
    """
    Transforming an image before entering the graph. This function bundles all the necessary steps to feed
    the network for training.

    Args:
        dp: A dict with `file_name`, `gt_boxes`, `gt_labels` and optional `image`.
        config: An `AttrDict` with a TP frcnn config.

    Returns:
        A dict with all necessary keys for feeding the graph.

    Note:
        If `image` is not in `dp`, it will be loaded from `file_name`.
    """
    cfg = config
    if "image" not in dp:
        loader = get_load_image_func(dp["file_name"])
        dp["image"] = loader(dp["file_name"])

    augment_list = [
        CustomResize(cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE, cfg.PREPROC.MAX_SIZE),
        imgaug.Flip(horiz=True),
    ]
    dp = augment(dp, augment_list, False)
    dp_with_anchors = anchors_and_labels(
        dp,
        cfg.FPN.ANCHOR_STRIDES,
        cfg.RPN.ANCHOR_SIZES,
        cfg.RPN.ANCHOR_RATIOS,
        cfg.PREPROC.MAX_SIZE,
        cfg.FRCNN.BATCH_PER_IM,
        cfg.RPN.FG_RATIO,
        cfg.RPN.POSITIVE_ANCHOR_THRESH,
        cfg.RPN.NEGATIVE_ANCHOR_THRESH,
        cfg.RPN.CROWD_OVERLAP_THRESH,
    )
    if dp_with_anchors is not None:
        dp_with_anchors.pop("file_name")
    return dp_with_anchors


def get_train_dataflow(
    dataset: DatasetBase, config: AttrDict, use_multi_proc_for_train: bool, **build_train_kwargs: str
) -> DataFlow:
    """
    Return a dataflow for training TP FRCNN. The returned dataflow depends on the dataset and the configuration of
    the model, as the augmentation is part of the data preparation.

    Args:
        dataset: A dataset for object detection.
        config: An `AttrDict` with a TP FRCNN config.
        use_multi_proc_for_train: If set to `True` will use multi processes for augmenting.
        build_train_kwargs: Build configuration of the dataflow.

    Returns:
        A dataflow.

    Note:
        If `use_multi_proc_for_train` is `True`, multi-processing will be used for augmentation.
    """

    set_mp_spawn()
    cfg = config
    df = dataset.dataflow.build(**build_train_kwargs)
    df = MapData(df, image_to_tp_frcnn_training(add_mask=False))  # pylint: disable=E1120

    logger.info("Loading dataset into memory")

    max_datapoints: Optional[int] = int(build_train_kwargs.get("max_datapoints", 0))
    if not max_datapoints:
        max_datapoints = None

    datapoints = []
    summarizer = LabelSummarizer(cfg.DATA.CLASS_DICT)
    df.reset_state()

    with get_tqdm(total=max_datapoints) as status_bar:
        for dp in df:
            if "image" in dp:
                log_once(
                    "Datapoint have images as np arrays stored and they will be loaded into memory. "
                    "To avoid OOM set 'load_image'=False in dataflow build config. This will load "
                    "images when needed and reduce memory costs!!!",
                    "warn",
                )
            summarizer.dump(dp["gt_labels"])
            datapoints.append(dp)
            status_bar.update()
    summarizer.print_summary_histogram(dd_logic=False)
    num_datapoints = len(datapoints)
    logger.info("Total #images for training: %i", num_datapoints)
    df = DataFromList(datapoints, shuffle=True)
    buffer_size = min(num_datapoints - 1, 200)

    load_augment_anchors = LoadAugmentAddAnchors(cfg)  # can't use dec: curry as pickling will fail in mp
    if use_multi_proc_for_train:
        num_cpu = os.cpu_count()
        if num_cpu is None:
            num_cpu = 0
        df = MultiProcessMapData(
            df,
            num_proc=1 if buffer_size < 3 else num_cpu // 2,
            map_func=load_augment_anchors,
            buffer_size=buffer_size,
        )
    else:
        df = MapData(df, load_augment_anchors)
    return ProxyDataFlow(df)


def train_faster_rcnn(
    path_config_yaml: PathLikeOrStr,
    dataset_train: DatasetBase,
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
    Easy adaptation of the training script for Tensorpack Faster-RCNN.

    Train Faster-RCNN from Scratch or fine-tune a model using Tensorpack's training API. Observe the training with
    Tensorpack callbacks and evaluate the training progress with a validation data set after certain training intervals.

    Info:
        Tensorpack provides a training API under TF1. Training runs under a TF2 installation if TF2 behavior is
        deactivated.

    Args:
        path_config_yaml: Path to TP config file. Check the `deepdoctection.extern.tp.tpfrcnn.config.config` for various
                          settings.
        dataset_train: The dataset to use for training.
        path_weights: Path to a checkpoint, if you want to continue training or fine-tune. Will train from scratch if
                      nothing is passed.
        config_overwrite: Pass a list of arguments if some configs from the .yaml file should be replaced. Use the list
                          convention, e.g. `[`TRAIN.STEPS_PER_EPOCH=500`, `OUTPUT.RESULT_SCORE_THRESH=0.4`]`.
        log_dir: Path to log dir. Will default to `TRAIN.LOG_DIR`.
        build_train_config: Dataflow build setting. Use list convention setting, e.g. `[`max_datapoints=1000`]`.
        dataset_val: The dataset to use for validation.
        build_val_config: Same as `build_train_config` but for validation.
        metric_name: A metric name to choose for validation. Will use the default setting. If you want a custom metric
                     setting pass a metric explicitly.
        metric: A metric to choose for validation.
        pipeline_component_name: A pipeline component to use for validation.

    Example:
        ```python
        train_faster_rcnn(
            path_config_yaml="config.yaml",
            dataset_train=my_train_dataset,
            path_weights="weights.ckpt"
        )
        ```
    """

    assert disable_tfv2()  # TP works only in Graph mode

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

    config_overwrite = [] if config_overwrite is None else config_overwrite

    log_dir = "TRAIN.LOG_DIR=" + os.fspath(log_dir)
    config_overwrite.append(log_dir)

    config = set_config_by_yaml(path_config_yaml)
    config.freeze(False)
    if config_overwrite:
        config.update_args(config_overwrite)
    config.freeze(True)

    categories = dataset_train.dataflow.categories.get_categories(filtered=True)
    model_frcnn_config(config, categories, False)
    model = ResNetFPNModel(config=config)

    warmup_schedule, lr_schedule, step_number = train_frcnn_config(config)

    train_dataflow = get_train_dataflow(dataset_train, config, True, **build_train_dict)
    # This is what's commonly referred to as "epochs"

    try:
        size = len(train_dataflow)
        total_passes = config.TRAIN.LR_SCHEDULE[-1] * 8 / size
        logger.info("Total passes of the training set is: %i", total_passes)

    except NotImplementedError:
        logger.info("Cannot evaluate size of dataflow and total passes")

    # Create callbacks ...

    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1, checkpoint_dir=config.TRAIN.LOG_DIR),
            every_k_epochs=config.TRAIN.CHECKPOINT_PERIOD,
        ),
        # linear warmup
        ScheduledHyperParamSetter("learning_rate", warmup_schedule, interp="linear", step_based=True),
        ScheduledHyperParamSetter("learning_rate", lr_schedule),
        GPUMemoryTracker(),
        HostMemoryTracker(),
        ThroughputTracker(samples_per_step=config.TRAIN.NUM_GPUS),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000),
        GPUUtilizationTracker(),
    ]

    if (
        config.TRAIN.EVAL_PERIOD > 0
        and dataset_val is not None
        and (metric_name is not None or metric is not None)
        and pipeline_component_name is not None
    ):
        if metric_name is not None:
            metric = metric_registry.get(metric_name)
        categories = dataset_val.dataflow.categories.get_categories(filtered=True)
        detector = TPFrcnnDetector(
            path_config_yaml,
            path_weights,
            categories,
            config_overwrite,
            True,
        )  # only a wrapper for the predictor itself. Will be replaced in Callback
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        pipeline_component = pipeline_component_cls(detector)
        category_names = list(categories.values())
        callbacks.extend(
            [
                EvalCallback(
                    dataset_val,
                    category_names,
                    dataset_val.dataflow.categories.cat_to_sub_cat,
                    metric,  # type: ignore
                    pipeline_component,
                    *model.get_inference_tensor_names(),  # type: ignore
                    cfg=detector.model.cfg,
                    **build_val_dict
                )
            ]
        )

    session_init = SmartInit(path_weights, ignore_mismatch=True)

    factor = 8.0 / config.TRAIN.NUM_GPUS

    train_cfg = TrainConfig(
        model=model,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=step_number,
        max_epoch=config.TRAIN.LR_SCHEDULE[-1] * factor // step_number,
        session_init=session_init,
        starting_epoch=config.TRAIN.STARTING_EPOCH,
    )

    trainer = SyncMultiGPUTrainerReplicated(config.TRAIN.NUM_GPUS, average=False)
    launch_train_with_config(train_cfg, trainer)
