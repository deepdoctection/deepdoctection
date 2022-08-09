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

import json
import copy
from typing import Optional, List, Dict, Union, Type, Any, Sequence

from torch.utils.data import Dataset
from torch.nn import Module

from transformers.trainer import Trainer, TrainingArguments
from transformers import PreTrainedModel, PretrainedConfig, LayoutLMForSequenceClassification, \
    LayoutLMForTokenClassification, LayoutLMTokenizerFast, IntervalStrategy
from ..eval.eval import Evaluator
from ..eval.base import MetricBase
from ..eval.registry import metric_registry
from ..datasets.base import DatasetBase
from ..datasets.adapter import DatasetAdapter
from ..pipe.base import PredictorPipelineComponent, LanguageModelPipelineComponent
from ..pipe.registry import pipeline_component_registry
from ..extern.pt.ptutils import get_num_gpu
from ..extern.hflayoutlm import HFLayoutLmSequenceClassifier, HFLayoutLmTokenClassifier
from ..mapper.laylmstruct import DataCollator, image_to_raw_layoutlm_features, LayoutLMDataCollator, image_to_layoutlm_features
from ..utils.utils import string_to_dict
from ..utils.settings import names


_ARCHITECTURES_TO_MODEL_CLASS = {"LayoutLMForTokenClassification": LayoutLMForTokenClassification,
                                 "LayoutLMForSequenceClassification": LayoutLMForSequenceClassification}
_MODEL_TYPE_AND_TASK_TO_MODEL_CLASS = {("layoutlm",names.DS.TYPE.SEQ): LayoutLMForSequenceClassification,
                                       ("layoutlm",names.DS.TYPE.TOK): LayoutLMForTokenClassification}
_MODEL_TYPE_TO_TOKENIZER = {"layoutlm": LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")}
_DS_TYPE_TO_DD_LM_CLASS = {names.DS.TYPE.TOK: HFLayoutLmTokenClassifier,
                           names.DS.TYPE.SEQ: HFLayoutLmSequenceClassifier}


class LayoutLMTrainer(Trainer):

    def __init__(self, model: Union[PreTrainedModel, Module],
                       args: TrainingArguments, data_collator: DataCollator, train_dataset):
        self.evaluator: Optional[Evaluator] = None
        self.build_eval_kwargs: Optional[Dict[str, Any]] = None
        super().__init__(model,args,data_collator, train_dataset)

    def setup_evaluator(self, dataset_val: DatasetBase, pipeline_component: LanguageModelPipelineComponent, metric: Type[MetricBase],
    **build_eval_kwargs) -> None:

        self.evaluator = Evaluator(dataset_val, pipeline_component, metric, num_threads=get_num_gpu() * 2)
        assert self.evaluator.pipe_component
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.language_model.model = None
        self.build_eval_kwargs = build_eval_kwargs

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        assert self.evaluator is not None
        assert self.evaluator.pipe_component is not None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.language_model.model = copy.deepcopy(self.model).eval()
        scores = self.evaluator.run(True, **self.build_eval_kwargs)

        self.log(scores)

        return scores


def _get_model_class_and_tokenizer(path_config_json: str, dataset_type: str):
    with open(path_config_json, "r", encoding="UTF-8") as file:
        config_json = json.load(file)

    model_type = config_json["model_type"]

    if architectures := config_json.get("architectures"):
        model_cls = _ARCHITECTURES_TO_MODEL_CLASS.get(architectures[0])
    elif model_type:
        model_cls = _MODEL_TYPE_AND_TASK_TO_MODEL_CLASS.get((model_type,dataset_type))
    else:
        raise KeyError("model_type and architectures not available in configs")

    if not model_cls:
        raise ValueError("model not eligible to run with this framework")

    tokenizer_fast = _MODEL_TYPE_TO_TOKENIZER[model_type]

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
    metric: Optional[Type[MetricBase]] = None,
    pipeline_component_name: Optional[str] = None,
) -> None:
    """
    :param path_config_json:
    :param dataset_train:
    :param path_weights:
    :param config_overwrite:
    :param log_dir:
    :param build_train_config:
    :param dataset_val:
    :param build_val_config:
    :param metric:
    :param pipeline_component_name:
    :return:
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
    # Need to set remove_unused_columns to False, as the DataCollator for column removal will remove some raw features
    # that are necessary for the tokenizer. We also define some default settings.
    conf_dict = {"output_dir": log_dir,
                 "remove_unused_columns": False,
                 "per_device_train_batch_size": 8,
                 "max_steps":130,
                 "evaluation_strategy": "steps" if (dataset_val is not None and metric is not None and pipeline_component_name is not None) else "no",
                 "eval_steps": 100 }

    for conf in config_overwrite:
        key, val = conf.split("=", maxsplit=1)
        conf_dict[key] = val

    arguments = TrainingArguments(**conf_dict)
    dataset_type = dataset_train.dataset_info.type

    model_cls, tokenizer_fast = _get_model_class_and_tokenizer(path_config_json,dataset_type)

    id_str_2label = dataset_train.dataflow.categories.get_categories(as_dict=True)
    id2label = {int(k)-1:v for k,v in id_str_2label.items()}
    dataset = DatasetAdapter(dataset_train,
                             True,
                             image_to_raw_layoutlm_features(dataset_train.dataflow.categories.get_categories(as_dict=True,
                             name_as_key=True),
                             dataset_type), **build_train_dict)
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config_json, id2label=id2label)
    model = model_cls.from_pretrained(pretrained_model_name_or_path=path_weights, config=config)
    data_collator = LayoutLMDataCollator(tokenizer_fast, return_tensors="pt")
    trainer = LayoutLMTrainer(model,arguments,data_collator, dataset)

    if arguments.evaluation_strategy in (IntervalStrategy.STEPS,):
        dd_model_cls = _DS_TYPE_TO_DD_LM_CLASS[dataset_type]
        categories = dataset_val.dataflow.categories.get_categories(filtered=True)
        dd_model = dd_model_cls(path_config_json=path_config_json, path_weights=path_weights, categories=categories,device="cuda")
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        pipeline_component = pipeline_component_cls(tokenizer_fast,dd_model,image_to_layoutlm_features)
        assert isinstance(pipeline_component, LanguageModelPipelineComponent)

        trainer.setup_evaluator(dataset_val,pipeline_component,metric,**build_val_dict)

    trainer.train()

    #tokenizer_fast = LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased")


