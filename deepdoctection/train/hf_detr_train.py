# -*- coding: utf-8 -*-
# File: hf_detr_train.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Module for training Hugging Face Detr implementation. Note, that this scripts only trans Tabletransformer like Detr
models that are a slightly different from the plain Detr model that are provided by the transformer library.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Optional, Sequence, Type, Union

from lazy_imports import try_import

from ..datasets.adapter import DatasetAdapter
from ..datasets.base import DatasetBase
from ..datasets.registry import get_dataset
from ..eval.base import MetricBase
from ..eval.eval import Evaluator
from ..eval.registry import metric_registry
from ..extern.hfdetr import HFDetrDerivedDetector
from ..mapper.hfstruct import DetrDataCollator, image_to_hf_detr_training
from ..pipe.base import PipelineComponent
from ..pipe.registry import pipeline_component_registry
from ..utils.logger import LoggingRecord, logger
from ..utils.types import PathLikeOrStr
from ..utils.utils import string_to_dict

with try_import() as pt_import_guard:
    from torch import nn
    from torch.utils.data import Dataset

with try_import() as hf_import_guard:
    from transformers import (
        AutoFeatureExtractor,
        IntervalStrategy,
        PretrainedConfig,
        PreTrainedModel,
        TableTransformerForObjectDetection,
        Trainer,
        TrainingArguments,
    )


class DetrDerivedTrainer(Trainer):
    """
    Huggingface Trainer for training Transformer models with a custom evaluate method in order
    to use dd Evaluator. Train setting is not defined in the trainer itself but in config setting as
    defined in `TrainingArguments`. Please check the Transformer documentation

    <https://huggingface.co/docs/transformers/main_classes/trainer>

    for custom training setting.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: TrainingArguments,
        data_collator: DetrDataCollator,
        train_dataset: Dataset[Any],
    ):
        self.evaluator: Optional[Evaluator] = None
        self.build_eval_kwargs: Optional[dict[str, Any]] = None
        super().__init__(model, args, data_collator, train_dataset)

    def setup_evaluator(
        self,
        dataset_val: DatasetBase,
        pipeline_component: PipelineComponent,
        metric: Union[Type[MetricBase], MetricBase],
        **build_eval_kwargs: Union[str, int],
    ) -> None:
        """
        Setup of evaluator before starting training. During training, predictors will be replaced by current
        checkpoints.

        :param dataset_val: dataset on which to run evaluation
        :param pipeline_component: pipeline component to plug into the evaluator
        :param metric: A metric class
        :param build_eval_kwargs:
        """

        self.evaluator = Evaluator(dataset_val, pipeline_component, metric, num_threads=1)
        assert self.evaluator.pipe_component
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.clear_predictor()
        self.build_eval_kwargs = build_eval_kwargs

    def evaluate(
        self,
        eval_dataset: Optional[Dataset[Any]] = None,  # pylint: disable=W0613
        ignore_keys: Optional[list[str]] = None,  # pylint: disable=W0613
        metric_key_prefix: str = "eval",  # pylint: disable=W0613
    ) -> dict[str, float]:
        """
        Overwritten method from `Trainer`. Arguments will not be used.
        """
        assert self.evaluator is not None
        assert self.evaluator.pipe_component is not None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.predictor.hf_detr_predictor = copy.deepcopy(self.model).eval()  # type: ignore
        if isinstance(self.build_eval_kwargs, dict):
            scores = self.evaluator.run(True, **self.build_eval_kwargs)
        else:
            scores = self.evaluator.run(True)

        self.log(scores)

        return scores


def train_hf_detr(
    path_config_json: PathLikeOrStr,
    dataset_train: Union[str, DatasetBase],
    path_weights: PathLikeOrStr,
    path_feature_extractor_config_json: str,
    config_overwrite: Optional[list[str]] = None,
    log_dir: PathLikeOrStr = "train_log/detr",
    build_train_config: Optional[Sequence[str]] = None,
    dataset_val: Optional[DatasetBase] = None,
    build_val_config: Optional[Sequence[str]] = None,
    metric_name: Optional[str] = None,
    metric: Optional[Union[Type[MetricBase], MetricBase]] = None,
    pipeline_component_name: Optional[str] = None,
) -> None:
    """
    Train Tabletransformer from scratch or fine-tune using an adaptation of the transformer trainer.
    Allowing experiments by using different config settings.

    :param path_config_json: path to a Tabletransformer config file
    :param dataset_train: dataset to use for training
    :param path_weights: path to a checkpoint, if you want to resume training or fine-tune. Will train from scratch if
                         an empty string is passed
    :param path_feature_extractor_config_json: path to a feature extractor config file. In many situations you can use
                                               the standard config file:

                                                   ModelCatalog.
                                                   get_full_path_preprocessor_configs
                                                   ("microsoft/table-transformer-detection/pytorch_model.bin")

    :param config_overwrite: Pass a list of arguments if some configs from the .json file are supposed to be replaced.
                             Use the list convention, e.g. ['per_device_train_batch_size=4']
    :param log_dir: Will default to 'train_log/detr'
    :param build_train_config: dataflow build setting. Again, use list convention setting, e.g. ['max_datapoints=1000']
    :param dataset_val: the dataset to use for validation
    :param build_val_config: same as `build_train_config` but for dataflow validation
    :param metric_name: A metric name to choose for validation. Will use the default setting. If you want a custom
                        metric setting, pass a metric explicitly.
    :param metric: A metric to choose for validation
    :param pipeline_component_name: A pipeline component name to use for validation
    """

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

    if isinstance(dataset_train, str):
        dataset_train = get_dataset(dataset_train)

    categories = dataset_train.dataflow.categories.get_categories(as_dict=False, filtered=True)
    categories_dict_name_as_key = dataset_train.dataflow.categories.get_categories(name_as_key=True, filtered=True)

    dataset = DatasetAdapter(
        dataset_train,
        True,
        image_to_hf_detr_training(category_names=categories),
        True,
        number_repetitions=-1,
        **build_train_dict,
    )

    number_samples = len(dataset)
    conf_dict = {
        "output_dir": os.fspath(log_dir),
        "remove_unused_columns": False,
        "per_device_train_batch_size": 2,
        "max_steps": number_samples,
        "evaluation_strategy": (
            "steps"
            if (dataset_val is not None and metric is not None and pipeline_component_name is not None)
            else "no"
        ),
        "eval_steps": 5000,
    }

    for conf in config_overwrite:
        key, val = conf.split("=", maxsplit=1)
        try:
            val = int(val)  # type: ignore
        except ValueError:
            try:
                val = float(val)  # type: ignore
            except ValueError:
                pass
        conf_dict[key] = val

    # Will inform about dataloader warnings if max_steps exceeds length of dataset
    if conf_dict["max_steps"] > number_samples:  # type: ignore
        logger.warning(
            LoggingRecord(
                f"After {number_samples} dataloader will log warning at every iteration " f"about unexpected samples"
            )
        )

    arguments = TrainingArguments(**conf_dict)
    logger.info(LoggingRecord(f"Config: \n {arguments.to_dict()}", arguments.to_dict()))

    id2label = {int(k) - 1: v for v, k in categories_dict_name_as_key.items()}
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path=path_config_json,
        num_labels=len(id2label),
    )

    if path_weights != "":
        model = TableTransformerForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )
    else:
        model = TableTransformerForObjectDetection(config)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        pretrained_model_name_or_path=path_feature_extractor_config_json
    )
    data_collator = DetrDataCollator(feature_extractor)
    trainer = DetrDerivedTrainer(model, arguments, data_collator, dataset)

    if arguments.evaluation_strategy in (IntervalStrategy.STEPS,):
        categories = dataset_val.dataflow.categories.get_categories(filtered=True)  # type: ignore
        detector = HFDetrDerivedDetector(
            path_config_json, path_weights, path_feature_extractor_config_json, categories  # type: ignore
        )
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        pipeline_component = pipeline_component_cls(detector)

        if metric_name is not None:
            metric = metric_registry.get(metric_name)
        assert metric is not None

        trainer.setup_evaluator(dataset_val, pipeline_component, metric)  # type: ignore

    trainer.train()
