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
import os
import pprint
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

from torch.nn import Module
from torch.utils.data import Dataset
from transformers import (
    IntervalStrategy,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
    LayoutLMTokenizerFast,
    LayoutLMv2Config,
    LayoutLMv2ForSequenceClassification,
    LayoutLMv2ForTokenClassification,
    LayoutLMv3Config,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    PretrainedConfig,
    PreTrainedModel,
    RobertaTokenizerFast,
    XLMRobertaTokenizerFast,
)
from transformers.trainer import Trainer, TrainingArguments

from ..datasets.adapter import DatasetAdapter
from ..datasets.base import DatasetBase
from ..datasets.registry import get_dataset
from ..eval.accmetric import ClassificationMetric
from ..eval.eval import Evaluator
from ..extern.hflayoutlm import (
    HFLayoutLmSequenceClassifier,
    HFLayoutLmTokenClassifier,
    HFLayoutLmv2SequenceClassifier,
    HFLayoutLmv2TokenClassifier,
    HFLayoutLmv3SequenceClassifier,
    HFLayoutLmv3TokenClassifier,
)
from ..mapper.laylmstruct import LayoutLMDataCollator, image_to_raw_layoutlm_features
from ..pipe.base import LanguageModelPipelineComponent
from ..pipe.lm import get_tokenizer_from_architecture
from ..pipe.registry import pipeline_component_registry
from ..utils.env_info import get_device
from ..utils.error import DependencyError
from ..utils.file_utils import wandb_available
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import DatasetType, LayoutType, ObjectTypes, WordType
from ..utils.utils import string_to_dict

if wandb_available():
    import wandb

_ARCHITECTURES_TO_MODEL_CLASS = {
    "LayoutLMForTokenClassification": (LayoutLMForTokenClassification, HFLayoutLmTokenClassifier, PretrainedConfig),
    "LayoutLMForSequenceClassification": (
        LayoutLMForSequenceClassification,
        HFLayoutLmSequenceClassifier,
        PretrainedConfig,
    ),
    "LayoutLMv2ForTokenClassification": (
        LayoutLMv2ForTokenClassification,
        HFLayoutLmv2TokenClassifier,
        LayoutLMv2Config,
    ),
    "LayoutLMv2ForSequenceClassification": (
        LayoutLMv2ForSequenceClassification,
        HFLayoutLmv2SequenceClassifier,
        LayoutLMv2Config,
    ),
}


_MODEL_TYPE_AND_TASK_TO_MODEL_CLASS: Mapping[Tuple[str, ObjectTypes], Any] = {
    ("layoutlm", DatasetType.sequence_classification): (
        LayoutLMForSequenceClassification,
        HFLayoutLmSequenceClassifier,
        PretrainedConfig,
    ),
    ("layoutlm", DatasetType.token_classification): (
        LayoutLMForTokenClassification,
        HFLayoutLmTokenClassifier,
        PretrainedConfig,
    ),
    ("layoutlmv2", DatasetType.sequence_classification): (
        LayoutLMv2ForSequenceClassification,
        HFLayoutLmv2SequenceClassifier,
        LayoutLMv2Config,
    ),
    ("layoutlmv2", DatasetType.token_classification): (
        LayoutLMv2ForTokenClassification,
        HFLayoutLmv2TokenClassifier,
        LayoutLMv2Config,
    ),
    ("layoutlmv3", DatasetType.sequence_classification): (
        LayoutLMv3ForSequenceClassification,
        HFLayoutLmv3SequenceClassifier,
        LayoutLMv3Config,
    ),
    ("layoutlmv3", DatasetType.token_classification): (
        LayoutLMv3ForTokenClassification,
        HFLayoutLmv3TokenClassifier,
        LayoutLMv3Config,
    ),
}
_MODEL_TYPE_TO_TOKENIZER = {
    ("layoutlm", False): LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased"),
    ("layoutlmv2", False): LayoutLMTokenizerFast.from_pretrained("microsoft/layoutlm-base-uncased"),
    ("layoutlmv2", True): XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", add_prefix_space=True),
    ("layoutlmv3", False): RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True),
}


class LayoutLMTrainer(Trainer):
    """
    Huggingface Trainer for training Transformer models with a custom evaluate method in order
    to use dd Evaluator. Train setting is not defined in the trainer itself but in config setting as
    defined in `TrainingArguments`. Please check the Transformer documentation

    <https://huggingface.co/docs/transformers/main_classes/trainer>

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
        metric: Union[Type[ClassificationMetric], ClassificationMetric],
        run: Optional["wandb.sdk.wandb_run.Run"] = None,
        **build_eval_kwargs: Union[str, int],
    ) -> None:
        """
        Setup of evaluator before starting training. During training, predictors will be replaced by current
        checkpoints.

        :param dataset_val: dataset on which to run evaluation
        :param pipeline_component: pipeline component to plug into the evaluator
        :param metric: A metric class
        :param run: WandB run
        :param build_eval_kwargs:
        """

        self.evaluator = Evaluator(dataset_val, pipeline_component, metric, num_threads=1, run=run)
        assert self.evaluator.pipe_component
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.language_model.model = None  # type: ignore
        self.build_eval_kwargs = build_eval_kwargs

    def evaluate(
        self,
        eval_dataset: Optional[Dataset[Any]] = None,  # pylint: disable=W0613
        ignore_keys: Optional[List[str]] = None,  # pylint: disable=W0613
        metric_key_prefix: str = "eval",  # pylint: disable=W0613
    ) -> Dict[str, float]:
        """
        Overwritten method from `Trainer`. Arguments will not be used.
        """
        if self.evaluator is None:
            raise ValueError("Evaluator not set up. Please use `setup_evaluator` before running evaluation")
        if self.evaluator.pipe_component is None:
            raise ValueError("Pipeline component not set up. Please use `setup_evaluator` before running evaluation")

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


def _get_model_class_and_tokenizer(
    path_config_json: str, dataset_type: ObjectTypes, use_xlm_tokenizer: bool
) -> Tuple[Any, Any, Any, Any]:
    with open(path_config_json, "r", encoding="UTF-8") as file:
        config_json = json.load(file)

    model_type = config_json.get("model_type")

    if architectures := config_json.get("architectures"):
        model_cls, model_wrapper_cls, config_cls = _ARCHITECTURES_TO_MODEL_CLASS[architectures[0]]
        tokenizer_fast = get_tokenizer_from_architecture(architectures[0], use_xlm_tokenizer)
    elif model_type:
        model_cls, model_wrapper_cls, config_cls = _MODEL_TYPE_AND_TASK_TO_MODEL_CLASS[(model_type, dataset_type)]
        tokenizer_fast = _MODEL_TYPE_TO_TOKENIZER[(model_type, use_xlm_tokenizer)]
    else:
        raise KeyError("model_type and architectures not available in configs")

    if not model_cls:
        raise UserWarning("model not eligible to run with this framework")

    return config_cls, model_cls, model_wrapper_cls, tokenizer_fast


def train_hf_layoutlm(
    path_config_json: str,
    dataset_train: Union[str, DatasetBase],
    path_weights: str,
    config_overwrite: Optional[List[str]] = None,
    log_dir: str = "train_log/layoutlm",
    build_train_config: Optional[Sequence[str]] = None,
    dataset_val: Optional[DatasetBase] = None,
    build_val_config: Optional[Sequence[str]] = None,
    metric: Optional[Union[Type[ClassificationMetric], ClassificationMetric]] = None,
    pipeline_component_name: Optional[str] = None,
    use_xlm_tokenizer: bool = False,
    use_token_tag: bool = True,
    segment_positions: Optional[Union[LayoutType, Sequence[LayoutType]]] = None,
) -> None:
    """
    Script for fine-tuning LayoutLM models either for sequence classification (e.g. classifying documents) or token
    classification using HF Trainer and custom evaluation. It currently supports LayoutLM, LayoutLMv2, LayoutLMv3 and
    LayoutXLM. Training similar but different models like LILT <https://arxiv.org/abs/2202.13669> can be done by
    changing a few lines of code regarding the selection of the tokenizer.

    The theoretical foundation can be taken from

    <https://arxiv.org/abs/1912.13318>

    This is not the pre-training script.

    In order to remain within the framework of this library, the base and uncased LayoutLM model must be downloaded
    from the HF-hub in a first step for fine-tuning.  Models are available for this, which are registered in the
    ModelCatalog. It is possible to choose one of the following options:

        "microsoft/layoutlm-base-uncased/pytorch_model.bin"
        "microsoft/layoutlmv2-base-uncased/pytorch_model.bin"
        "microsoft/layoutxlm-base/pytorch_model.bin"
        "microsoft/layoutlmv3-base/pytorch_model.bin"

     and

         "microsoft/layoutlm-large-uncased/pytorch_model.bin"

    (You can also choose the large versions of LayoutLMv2 and LayoutXLM but you need to organize the download yourself.)

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
    :param use_xlm_tokenizer: This is only necessary if you pass weights of layoutxlm. The config cannot distinguish
                              between Layoutlmv2 and Layoutxlm, so you need to pass this info explicitly.
    :param use_token_tag: Will only be used for dataset_type="token_classification". If use_token_tag=True, will use
                          labels from sub category `WordType.token_tag` (with `B,I,O` suffix), otherwise
                          `WordType.token_class`.
    :param segment_positions: Using bounding boxes of segment instead of words improves model accuracy significantly.
                              Choose a single or a sequence of layout segments to use their bounding boxes. Note, that
                              the layout segments need to have a child-relationship with words. If a word does not
                              appear as child, it will use the word bounding box.
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
        if use_token_tag:
            categories_dict_name_as_key = dataset_train.dataflow.categories.get_sub_categories(
                categories=LayoutType.word,
                sub_categories={LayoutType.word: [WordType.token_tag]},
                keys=False,
                values_as_dict=True,
                name_as_key=True,
            )[LayoutType.word][WordType.token_tag]
        else:
            categories_dict_name_as_key = dataset_train.dataflow.categories.get_sub_categories(
                categories=LayoutType.word,
                sub_categories={LayoutType.word: [WordType.token_class]},
                keys=False,
                values_as_dict=True,
                name_as_key=True,
            )[LayoutType.word][WordType.token_class]
    else:
        raise UserWarning("Dataset type not supported for training")

    config_cls, model_cls, model_wrapper_cls, tokenizer_fast = _get_model_class_and_tokenizer(
        path_config_json, dataset_type, use_xlm_tokenizer
    )
    image_to_raw_layoutlm_kwargs = {"dataset_type": dataset_type, "use_token_tag": use_token_tag}
    if segment_positions:
        image_to_raw_layoutlm_kwargs["segment_positions"] = segment_positions  # type: ignore
    image_to_raw_layoutlm_kwargs.update(model_wrapper_cls.default_kwargs_for_input_mapping())
    dataset = DatasetAdapter(
        dataset_train,
        True,
        image_to_raw_layoutlm_features(**image_to_raw_layoutlm_kwargs),
        use_token_tag,
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
        "evaluation_strategy": (
            "steps"
            if (dataset_val is not None and metric is not None and pipeline_component_name is not None)
            else "no"
        ),
        "eval_steps": 100,
        "use_wandb": False,
        "wandb_project": None,
        "wandb_repo": "deepdoctection",
        "sliding_window_stride": 0,
        "max_batch_size": 0,
    }

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

    max_batch_size = conf_dict.pop("max_batch_size")
    sliding_window_stride = conf_dict.pop("sliding_window_stride")
    if sliding_window_stride and not max_batch_size:
        logger.warning(
            LoggingRecord(
                "sliding_window_stride is not 0 and max_batch_size is 0. This can result in CUDA out of "
                "memory because the batch size can be higher than per_device_train_batch_size. Set "
                "max_batch_size to a positive number if you encounter this type of problem.",
            )
        )

    use_wandb = conf_dict.pop("use_wandb")
    wandb_project = conf_dict.pop("wandb_project")
    wandb_repo = conf_dict.pop("wandb_repo")

    # Initialize Wandb, if necessary
    run = None
    if use_wandb:
        if not wandb_available():
            raise DependencyError("WandB must be installed separately")
        run = wandb.init(project=wandb_project, config=conf_dict)  # type: ignore
        run._label(repo=wandb_repo)  # type: ignore # pylint: disable=W0212
    else:
        os.environ["WANDB_DISABLED"] = "True"

    # Will inform about dataloader warnings if max_steps exceeds length of dataset
    if conf_dict["max_steps"] > number_samples:  # type: ignore
        logger.warning(
            LoggingRecord(
                f"After {number_samples} dataloader will log warning at every iteration about unexpected " f"samples"
            )
        )

    arguments = TrainingArguments(**conf_dict)  # pylint: disable=E1123
    logger.info(LoggingRecord(f"Config: \n {arguments.to_dict()}", arguments.to_dict()))

    id2label = {int(k) - 1: v for v, k in categories_dict_name_as_key.items()}

    logger.info(
        LoggingRecord(
            f"Will setup a head with the following classes\n " f"{pprint.pformat(id2label, width=100, compact=True)}"
        )
    )

    config = config_cls.from_pretrained(pretrained_model_name_or_path=path_config_json, id2label=id2label)
    model = model_cls.from_pretrained(pretrained_model_name_or_path=path_weights, config=config)
    data_collator = LayoutLMDataCollator(
        tokenizer_fast,
        return_tensors="pt",
        sliding_window_stride=sliding_window_stride,  # type: ignore
        max_batch_size=max_batch_size,  # type: ignore
    )
    trainer = LayoutLMTrainer(model, arguments, data_collator, dataset)

    if arguments.evaluation_strategy in (IntervalStrategy.STEPS,):
        assert metric is not None  # silence mypy
        if dataset_type == DatasetType.sequence_classification:
            categories = dataset_val.dataflow.categories.get_categories(filtered=True)  # type: ignore
        else:
            if use_token_tag:
                categories = dataset_val.dataflow.categories.get_sub_categories(  # type: ignore
                    categories=LayoutType.word, sub_categories={LayoutType.word: [WordType.token_tag]}, keys=False
                )[LayoutType.word][WordType.token_tag]
                metric.set_categories(category_names=LayoutType.word, sub_category_names={"word": ["token_tag"]})
            else:
                categories = dataset_val.dataflow.categories.get_sub_categories(  # type: ignore
                    categories=LayoutType.word, sub_categories={LayoutType.word: [WordType.token_class]}, keys=False
                )[LayoutType.word][WordType.token_class]
                metric.set_categories(category_names=LayoutType.word, sub_category_names={"word": ["token_class"]})
        dd_model = model_wrapper_cls(
            path_config_json=path_config_json,
            path_weights=path_weights,
            categories=categories,
            device=get_device(),
        )
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        if dataset_type == DatasetType.sequence_classification:
            pipeline_component = pipeline_component_cls(tokenizer_fast, dd_model)
        else:
            pipeline_component = pipeline_component_cls(
                tokenizer_fast,
                dd_model,
                use_other_as_default_category=True,
                sliding_window_stride=sliding_window_stride,
            )
        assert isinstance(pipeline_component, LanguageModelPipelineComponent)

        trainer.setup_evaluator(dataset_val, pipeline_component, metric, run, **build_val_dict)  # type: ignore

    trainer.train()
