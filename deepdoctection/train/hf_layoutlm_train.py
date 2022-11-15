# -*- coding: utf-8 -*-
# File: hf_layoutlm_train.py

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
Module for training Huggingface implementation of LayoutLm
"""
import copy
import json
import pprint
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

from torch.nn import Module
from torch.utils.data import Dataset
from transformers import (
    IntervalStrategy,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
    LayoutLMTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.trainer import Trainer, TrainingArguments

from ..datasets.adapter import DatasetAdapter
from ..datasets.base import DatasetBase
from ..datasets.registry import get_dataset
from ..eval.base import MetricBase
from ..eval.eval import Evaluator
from ..extern.hflayoutlm import HFLayoutLmSequenceClassifier, HFLayoutLmTokenClassifier
from ..mapper.laylmstruct import LayoutLMDataCollator, image_to_layoutlm_features, image_to_raw_layoutlm_features
from ..pipe.base import LanguageModelPipelineComponent
from ..pipe.registry import pipeline_component_registry
from ..utils.logger import logger
from ..utils.settings import DatasetType, LayoutType, ObjectTypes, WordType
from ..utils.utils import string_to_dict

_ARCHITECTURES_TO_MODEL_CLASS = {
    "LayoutLMForTokenClassification": LayoutLMForTokenClassification,
    "LayoutLMForSequenceClassification": LayoutLMForSequenceClassification,
}
_ARCHITECTURES_TO_TOKENIZER = {
    "LayoutLMForTokenClassification": LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased"),
    "LayoutLMForSequenceClassification": LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased"),
}
_MODEL_TYPE_AND_TASK_TO_MODEL_CLASS: Mapping[Tuple[str, ObjectTypes], Any] = {
    ("layoutlm", DatasetType.sequence_classification): LayoutLMForSequenceClassification,
    ("layoutlm", DatasetType.token_classification): LayoutLMForTokenClassification,
}
_MODEL_TYPE_TO_TOKENIZER = {"layoutlm": LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")}
_DS_TYPE_TO_DD_LM_CLASS: Mapping[ObjectTypes, Any] = {
    DatasetType.token_classification: HFLayoutLmTokenClassifier,
    DatasetType.sequence_classification: HFLayoutLmSequenceClassifier,
}


class LayoutLMTrainer(Trainer):
    """
    Huggingface Trainer for training Transformer models with a custom evaluate method in order
    to use dd Evaluator. Train setting is not defined in the trainer itself but in config setting as
    defined in `TrainingArguments`. Please check the Transformer documentation

    https://huggingface.co/docs/transformers/main_classes/trainer

    for custom training setting.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, Module],
        args: TrainingArguments,
        data_collator: LayoutLMDataCollator,
        train_dataset: Dataset[Any],
    ):
        self.evaluator: Optional[Evaluator] = None
        self.build_eval_kwargs: Optional[Dict[str, Any]] = None
        super().__init__(model, args, data_collator, train_dataset)

    def setup_evaluator(
        self,
        dataset_val: DatasetBase,
        pipeline_component: LanguageModelPipelineComponent,
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

        self.evaluator = Evaluator(dataset_val, pipeline_component, metric, num_threads=2)
        assert self.evaluator.pipe_component
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.language_model.model = None  # type: ignore
        self.build_eval_kwargs = build_eval_kwargs

    def evaluate(
        self,
        eval_dataset: Optional[Dataset[Any]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Overwritten method from :class:`Trainer`. Arguments will not be used.
        """
        assert self.evaluator is not None
        assert self.evaluator.pipe_component is not None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.language_model.model = copy.deepcopy(self.model).eval()  # type: ignore
        if isinstance(self.build_eval_kwargs, dict):
            scores = self.evaluator.run(True, **self.build_eval_kwargs)
        else:
            scores = self.evaluator.run(True)

        self.log(scores)

        return scores


def _get_model_class_and_tokenizer(path_config_json: str, dataset_type: ObjectTypes) -> Tuple[Any, Any]:
    with open(path_config_json, "r", encoding="UTF-8") as file:
        config_json = json.load(file)

    model_type = config_json.get("model_type")

    if architectures := config_json.get("architectures"):
        model_cls = _ARCHITECTURES_TO_MODEL_CLASS.get(architectures[0])
        tokenizer_fast = _ARCHITECTURES_TO_TOKENIZER.get(architectures[0])
    elif model_type:
        model_cls = _MODEL_TYPE_AND_TASK_TO_MODEL_CLASS.get((model_type, dataset_type))
        tokenizer_fast = _MODEL_TYPE_TO_TOKENIZER[model_type]
    else:
        raise KeyError("model_type and architectures not available in configs")

    if not model_cls:
        raise ValueError("model not eligible to run with this framework")

    return model_cls, tokenizer_fast


def train_hf_layoutlm(
    path_config_json: str,
    dataset_train: Union[str, DatasetBase],
    path_weights: str,
    config_overwrite: Optional[List[str]] = None,
    log_dir: str = "train_log/layoutlm",
    build_train_config: Optional[Sequence[str]] = None,
    dataset_val: Optional[DatasetBase] = None,
    build_val_config: Optional[Sequence[str]] = None,
    metric: Optional[Union[Type[MetricBase], MetricBase]] = None,
    pipeline_component_name: Optional[str] = None,
) -> None:
    """
    Script for fine-tuning LayoutLM models either for sequence classification (e.g. classifying documents) or token
    classification using HF Trainer and custom evaluation. The theoretical foundation can be taken from

    https://arxiv.org/abs/1912.13318

    This is not the pre-training script.

    In order to remain within the framework of this library, the basic LayoutLM model must be downloaded from the HF-hub
    in a first step for fine-tuning. Two models are available for this, which are registered in the ModelCatalog:

    "microsoft/layoutlm-base-uncased/pytorch_model.bin"

     and

     "microsoft/layoutlm-large-uncased/pytorch_model.bin"


    .. code-block:: python

        ModelDownloadManager.maybe_download_weights_and_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")

    The corresponding cased models are currently not available, but this is only to keep the model selection small.

    If the config file and weights have been downloaded, the model can be trained for the desired task.

    How does the model selection work?

    The base model is selected by the transferred config file and the weights. Depending on the dataset type
    ("SEQUENCE_CLASSIFICATION" or "TOKEN_CLASSIFICATION"), the complete model is then put together by placing a suitable
    top layer on the base model.

    :param path_config_json: Absolute path to HF config file, e.g.
                             ModelCatalog.get_full_path_configs("microsoft/layoutlm-base-uncased/pytorch_model.bin")
    :param dataset_train: Dataset to use for training. Only datasets of type "SEQUENCE_CLASSIFICATION" or
                          "TOKEN_CLASSIFICATION" are supported.
    :param path_weights: path to a checkpoint for further fine-tuning
    :param config_overwrite: Pass a list of arguments if some configs from `TrainingArguments` should be replaced. Check
                             https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
                             for the full training default setting.
    :param log_dir: Path to log dir. Will default to `train_log/layoutlm`
    :param build_train_config: dataflow build setting. Again, use list convention setting, e.g. ['max_datapoints=1000']
    :param dataset_val: Dataset to use for validation. Dataset type must be the same as type of `dataset_train`
    :param build_val_config: same as `build_train_config` but for validation
    :param metric: A metric to choose for validation.
    :param pipeline_component_name: A pipeline component name to use for validation (e.g. LMSequenceClassifierService or
                                    LMTokenClassifierService.
    """

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

    if isinstance(dataset_train, str):
        dataset_train = get_dataset(dataset_train)

    # We wrap our dataset into a torch dataset
    dataset_type = dataset_train.dataset_info.type
    if dataset_type == DatasetType.sequence_classification:
        categories_dict_name_as_key = dataset_train.dataflow.categories.get_categories(as_dict=True, name_as_key=True)
    elif dataset_type == DatasetType.token_classification:
        categories_dict_name_as_key = dataset_train.dataflow.categories.get_sub_categories(
            categories=LayoutType.word,
            sub_categories={LayoutType.word: [WordType.token_tag]},
            keys=False,
            values_as_dict=True,
            name_as_key=True,
        )[LayoutType.word][WordType.token_tag]
    else:
        raise ValueError("Dataset type not supported for training")

    dataset = DatasetAdapter(
        dataset_train,
        True,
        image_to_raw_layoutlm_features(categories_dict_name_as_key, dataset_type),
        **build_train_dict,
    )

    number_samples = len(dataset)
    # A setup of necessary configuration. Everything else will be equal to the default setting of the transformer
    # library.
    # Need to set remove_unused_columns to False, as the DataCollator for column removal will remove some raw features
    # that are necessary for the tokenizer.
    conf_dict = {
        "output_dir": log_dir,
        "remove_unused_columns": False,
        "per_device_train_batch_size": 8,
        "max_steps": number_samples,
        "evaluation_strategy": "steps"
        if (dataset_val is not None and metric is not None and pipeline_component_name is not None)
        else "no",
        "eval_steps": 100,
    }

    if isinstance(dataset_train, str):
        dataset_train = get_dataset(dataset_train)

    # We allow to overwrite the default setting by the user.
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
            "After %s dataloader will log warning at every iteration about unexpected samples", number_samples
        )

    arguments = TrainingArguments(**conf_dict)
    logger.info("Config: \n %s", str(arguments.to_dict()), arguments.to_dict())

    model_cls, tokenizer_fast = _get_model_class_and_tokenizer(path_config_json, dataset_type)

    id2label = {int(k) - 1: v for v, k in categories_dict_name_as_key.items()}

    logger.info("Will setup a head with the following classes\n %s", pprint.pformat(id2label, width=100, compact=True))

    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json, id2label=id2label)
    model = model_cls.from_pretrained(pretrained_model_name_or_path=path_weights, config=config)
    data_collator = LayoutLMDataCollator(tokenizer_fast, return_tensors="pt")
    trainer = LayoutLMTrainer(model, arguments, data_collator, dataset)

    if arguments.evaluation_strategy in (IntervalStrategy.STEPS,):
        dd_model_cls = _DS_TYPE_TO_DD_LM_CLASS[dataset_type]
        if dataset_type == DatasetType.sequence_classification:
            categories = dataset_val.dataflow.categories.get_categories(filtered=True)  # type: ignore
        else:
            categories = dataset_val.dataflow.categories.get_sub_categories(  # type: ignore
                categories=LayoutType.word, sub_categories={LayoutType.word: [WordType.token_tag]}, keys=False
            )[LayoutType.word][WordType.token_tag]
        dd_model = dd_model_cls(
            path_config_json=path_config_json,
            path_weights=path_weights,
            categories=categories,
            device="cuda",
        )
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        if dataset_type == DatasetType.sequence_classification:
            pipeline_component = pipeline_component_cls(tokenizer_fast, dd_model, image_to_layoutlm_features)
        else:
            pipeline_component = pipeline_component_cls(tokenizer_fast, dd_model, image_to_layoutlm_features, True)
        assert isinstance(pipeline_component, LanguageModelPipelineComponent)

        trainer.setup_evaluator(dataset_val, pipeline_component, metric, **build_val_dict)  # type: ignore

    trainer.train()
