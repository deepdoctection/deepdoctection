# -*- coding: utf-8 -*-
# File: hfdetr.py

# Copyright 2022 Dr. Janis Meyer. All rights reserved.
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
HF Detr and DeformableDetr models.
"""

from __future__ import annotations

import os
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, Optional, Sequence, Union

from lazy_imports import try_import

from ..utils.file_utils import get_pytorch_requirement, get_transformers_requirement
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..utils.types import PathLikeOrStr, PixelValues, Requirement
from .base import DetectionResult, ModelCategories, ObjectDetector
from .pt.ptutils import get_torch_device

with try_import() as pt_import_guard:
    import torch  # pylint: disable=W0611
    from torchvision.ops import boxes as box_ops  # type: ignore

with try_import() as tr_import_guard:
    from transformers import (  # pylint: disable=W0611
        DeformableDetrForObjectDetection,
        DeformableDetrImageProcessorFast,
        DetrImageProcessorFast,
        PretrainedConfig,
        TableTransformerForObjectDetection,
    )

    if TYPE_CHECKING:
        EligibleDetrModel = Union[TableTransformerForObjectDetection, DeformableDetrForObjectDetection]
        DetrImageProcessor = Union[DetrImageProcessorFast, DeformableDetrImageProcessorFast]


def _detr_post_processing(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, nms_thresh: float
) -> torch.Tensor:
    return box_ops.batched_nms(boxes.float(), scores, labels, nms_thresh)


def detr_predict_image(
    np_img: PixelValues,
    predictor: EligibleDetrModel,
    feature_extractor: DetrImageProcessor,
    device: torch.device,
    threshold: float,
    nms_threshold: float,
) -> list[DetectionResult]:
    """
    Calling predictor. Before, tensors must be transferred to the device where the model is loaded.

    Args:
        np_img: Image as `np.array`.
        predictor: `TableTransformerForObjectDetection` instance.
        feature_extractor: Feature extractor instance.
        device: Device where the model is loaded.
        threshold: Will filter all predictions with confidence score less threshold.
        nms_threshold: Threshold to perform NMS on prediction outputs.
                       Note:
                           NMS does not belong to canonical Detr inference processing.

    Returns:
        List of `DetectionResult` after running prediction.
    """
    target_sizes = [np_img.shape[:2]]
    inputs = feature_extractor(images=np_img, return_tensors="pt")
    inputs.data["pixel_values"] = inputs.data["pixel_values"].to(device)
    inputs.data["pixel_mask"] = inputs.data["pixel_mask"].to(device)
    outputs = predictor(**inputs)
    outputs["encoder_last_hidden_state"] = outputs["encoder_last_hidden_state"].to("cpu")
    outputs["last_hidden_state"] = outputs["last_hidden_state"].to("cpu")
    outputs["logits"] = outputs["logits"].to("cpu")
    outputs["pred_boxes"] = outputs["pred_boxes"].to("cpu")
    results = feature_extractor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[
        0
    ]
    keep = _detr_post_processing(results["boxes"], results["scores"], results["labels"], nms_threshold)
    keep_boxes = results["boxes"][keep]
    keep_scores = results["scores"][keep]
    keep_labels = results["labels"][keep]
    return [
        DetectionResult(box=box.tolist(), score=score.item(), class_id=class_id.item())
        for box, score, class_id in zip(keep_boxes, keep_scores, keep_labels)
    ]


class HFDetrDerivedDetectorMixin(ObjectDetector, ABC):
    """Base class for Detr object detector. This class only implements the basic wrapper functions"""

    def __init__(self, categories: Mapping[int, TypeOrStr], filter_categories: Optional[Sequence[TypeOrStr]] = None):
        """
        Args:
            categories: A dict with key (indices) and values (category names).
            filter_categories: The model might return objects that are not supposed to be predicted and that should
                               be filtered. Pass a list of category names that must not be returned.
        """
        self.categories = ModelCategories(init_categories=categories)
        if filter_categories:
            self.categories.filter_categories = tuple(get_type(cat) for cat in filter_categories)

    def _map_category_names(self, detection_results: list[DetectionResult]) -> list[DetectionResult]:
        """
        Populating category names to `DetectionResult`. Will also filter categories.

        Args:
            detection_results: List of `DetectionResult`s.

        Returns:
            List of `DetectionResult`s with `class_name`.
        """
        filtered_detection_result: list[DetectionResult] = []
        shifted_categories = self.categories.shift_category_ids(shift_by=-1)
        for result in detection_results:
            result.class_name = shifted_categories.get(
                result.class_id if result.class_id is not None else -1, DefaultType.DEFAULT_TYPE
            )
            if result.class_name != DefaultType.DEFAULT_TYPE:
                if result.class_id is not None:
                    result.class_id += 1
                    filtered_detection_result.append(result)

        return filtered_detection_result

    @staticmethod
    def get_name(path_weights: PathLikeOrStr) -> str:
        """
        Returns the name of the model.

        Args:
            path_weights: Path to the model weights.

        Returns:
            Model name string.
        """
        return "Transformers_Tatr_" + "_".join(Path(path_weights).parts[-2:])

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """
        Returns:
            Tuple of `category_name`s.
        """
        return self.categories.get_categories(as_dict=False)


class HFDetrDerivedDetector(HFDetrDerivedDetectorMixin):
    """
    Model wrapper for `TableTransformerForObjectDetection` that again is based on
    <https://github.com/microsoft/table-transformer>.

    The wrapper can be used to load pre-trained models for table detection and table structure recognition. Running Detr
    models trained from scratch on custom datasets is possible as well.

    Note:
        This wrapper will load `TableTransformerForObjectDetection` that is slightly different compared to
        `DetrForObjectDetection` that can be found in the transformer library as well.

    Example:
        ```python
        config_path = ModelCatalog.
        get_full_path_configs("microsoft/table-transformer-structure-recognition/pytorch_model.bin")
        weights_path = ModelDownloadManager.
        get_full_path_weights("microsoft/table-transformer-structure-recognition/pytorch_model.bin")
        feature_extractor_config_path = ModelDownloadManager.
        get_full_path_preprocessor_configs("microsoft/table-transformer-structure-recognition/pytorch_model.bin")
        categories = ModelCatalog.
        get_profile("microsoft/table-transformer-structure-recognition/pytorch_model.bin").categories

        detr_predictor = HFDetrDerivedDetector(config_path,weights_path,feature_extractor_config_path,categories)

        detection_result = detr_predictor.predict(bgr_image_np_array)
        ```
    """

    def __init__(
        self,
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        path_feature_extractor_config_json: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        Set up the predictor.

        Args:
            path_config_json: The path to the json config.
            path_weights: The path to the model checkpoint.
            path_feature_extractor_config_json: The path to the feature extractor config.
            categories: A dict with key (indices) and values (category names).
            device: "cpu" or "cuda". If not specified will auto select depending on what is available.
            filter_categories: The model might return objects that are not supposed to be predicted and that should
                              be filtered. Pass a list of category names that must not be returned.
        """
        super().__init__(categories, filter_categories)

        self.path_config = Path(path_config_json)
        self.path_weights = Path(path_weights)
        self.path_feature_extractor_config = Path(path_feature_extractor_config_json)

        self.name = self.get_name(self.path_weights)
        self.model_id = self.get_model_id()

        self.config = self.get_config(path_config_json)

        self.hf_detr_predictor = self.get_model(self.path_weights, self.config)
        self.feature_extractor = self.get_pre_processor(self.path_feature_extractor_config, self.config)

        self.device = get_torch_device(device)
        self.hf_detr_predictor.to(self.device)

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Predicts objects in an image.

        Args:
            np_img: Image as `np.array`.

        Returns:
            List of `DetectionResult`.
        """
        results = detr_predict_image(
            np_img,
            self.hf_detr_predictor,
            self.feature_extractor,
            self.device,
            self.config.threshold,
            self.config.nms_threshold,
        )
        return self._map_category_names(results)

    @staticmethod
    def get_model(path_weights: PathLikeOrStr, config: PretrainedConfig) -> EligibleDetrModel:
        """
        Builds the Detr model.

        Args:
            path_weights: The path to the model checkpoint.
            config: `PretrainedConfig` instance.

        Returns:
            `TableTransformerForObjectDetection` instance.

        Raises:
            ValueError: If model architecture is not eligible.
        """
        if "TableTransformerForObjectDetection" in config.architectures:
            return TableTransformerForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=os.fspath(path_weights), config=config
            )
        if "DeformableDetrForObjectDetection" in config.architectures:
            return DeformableDetrForObjectDetection.from_pretrained(
                pretrained_model_name_or_path=os.fspath(path_weights), config=config
            )
        raise ValueError(
            f"Model architecture {config.architectures} not eligible. Please use either "
            "TableTransformerForObjectDetection or DeformableDetrForObjectDetection."
        )

    @staticmethod
    def get_pre_processor(path_feature_extractor_config: PathLikeOrStr, config: PretrainedConfig) -> DetrImageProcessor:
        """
        Builds the feature extractor.

        Args:
            path_feature_extractor_config: Path to feature extractor config.
            config: Model configuration.

        Returns:
            `DetrImageProcessor` instance.

        Raises:
            ValueError: If model architecture is not eligible.
        """
        if "TableTransformerForObjectDetection" in config.architectures:
            return DetrImageProcessorFast.from_pretrained(
                pretrained_model_name_or_path=os.fspath(path_feature_extractor_config),
            )
        if "DeformableDetrForObjectDetection" in config.architectures:
            return DeformableDetrImageProcessorFast.from_pretrained(
                pretrained_model_name_or_path=os.fspath(path_feature_extractor_config),
            )
        raise ValueError(
            f"Model architecture {config.architectures} not eligible. Please use either "
            "TableTransformerForObjectDetection or DeformableDetrForObjectDetection."
        )

    @staticmethod
    def get_config(path_config: PathLikeOrStr) -> PretrainedConfig:
        """
        Builds the config.

        Args:
            path_config: The path to the config.

        Returns:
            `PretrainedConfig` instance.
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=os.fspath(path_config))
        config.use_timm_backbone = True
        config.threshold = 0.1
        config.nms_threshold = 0.05
        return config

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> HFDetrDerivedDetector:
        return self.__class__(
            self.path_config,
            self.path_weights,
            self.path_feature_extractor_config,
            self.categories.get_categories(),
            self.device,
            self.categories.filter_categories,
        )

    @staticmethod
    def get_wrapped_model(
        path_config_json: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
    ) -> TableTransformerForObjectDetection:
        """
        Get the wrapped model.

        Args:
            path_config_json: The path to the json config.
            path_weights: The path to the model checkpoint.
            device: "cpu" or "cuda". If not specified will auto select depending on what is available.

        Returns:
            `TableTransformerForObjectDetection` instance.
        """
        config = HFDetrDerivedDetector.get_config(path_config_json)
        hf_detr_predictor = HFDetrDerivedDetector.get_model(path_weights, config)
        device = get_torch_device(device)
        hf_detr_predictor.to(device)
        return hf_detr_predictor

    def clear_model(self) -> None:
        self.hf_detr_predictor = None
