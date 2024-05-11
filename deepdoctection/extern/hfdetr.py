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
HF Detr model for object detection.
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import List, Literal, Mapping, Optional, Sequence, Union

from lazy_imports import try_import

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import get_pytorch_requirement, get_transformers_requirement
from ..utils.settings import TypeOrStr, get_type
from .base import DetectionResult, ObjectDetector
from .pt.ptutils import get_torch_device

with try_import() as pt_import_guard:
    import torch  # pylint: disable=W0611
    from torchvision.ops import boxes as box_ops  # type: ignore

with try_import() as tr_import_guard:
    from transformers import (  # pylint: disable=W0611
        AutoFeatureExtractor,
        DetrFeatureExtractor,
        PretrainedConfig,
        TableTransformerForObjectDetection,
    )


def _detr_post_processing(
    boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, nms_thresh: float
) -> torch.Tensor:
    return box_ops.batched_nms(boxes.float(), scores, labels, nms_thresh)


def detr_predict_image(
    np_img: ImageType,
    predictor: TableTransformerForObjectDetection,
    feature_extractor: DetrFeatureExtractor,
    device: torch.device,
    threshold: float,
    nms_threshold: float,
) -> List[DetectionResult]:
    """
    Calling predictor. Before doing that, tensors must be transferred to the device where the model is loaded. After
    running prediction it will present prediction in DetectionResult format-

    :param np_img: image as numpy array
    :param predictor: TableTransformerForObjectDetection
    :param feature_extractor: feature extractor
    :param device: device where the model is loaded
    :param threshold: Will filter all predictions with confidence score less threshold
    :param nms_threshold: Threshold to perform NMS on prediction outputs. (Note, that NMS does not belong to canonical
                          Detr inference processing)

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

    def __init__(self, categories: Mapping[str, TypeOrStr], filter_categories: Optional[Sequence[TypeOrStr]] = None):
        """

        :param categories: A dict with key (indices) and values (category names).
        :param filter_categories: The model might return objects that are not supposed to be predicted and that should
                                  be filtered. Pass a list of category names that must not be returned
        """
        self.categories = {idx: get_type(cat) for idx, cat in categories.items()}
        if filter_categories:
            filter_categories = [get_type(cat) for cat in filter_categories]
        self.filter_categories = filter_categories

    def _map_category_names(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        """
        Populating category names to detection results. Will also filter categories

        :param detection_results: list of detection results
        :return: List of detection results with attribute class_name populated
        """
        filtered_detection_result: List[DetectionResult] = []
        for result in detection_results:
            result.class_name = self.categories[str(result.class_id + 1)]  # type: ignore
            if isinstance(result.class_id, int):
                result.class_id += 1
            if self.filter_categories:
                if result.class_name not in self.filter_categories:
                    filtered_detection_result.append(result)
            else:
                filtered_detection_result.append(result)

        return filtered_detection_result

    @staticmethod
    def get_name(path_weights: str) -> str:
        """Returns the name of the model"""
        return "Transformers_Tatr_" + "_".join(Path(path_weights).parts[-2:])


class HFDetrDerivedDetector(HFDetrDerivedDetectorMixin):
    """
    Model wrapper for TableTransformerForObjectDetection that again is based on

    https://github.com/microsoft/table-transformer .

    The wrapper can be used to load pre-trained models for table detection and table structure recognition. Running Detr
    models trained from scratch on custom datasets is possible as well. Note, that this wrapper will load
    `TableTransformerForObjectDetection` that is slightly different compared to `DetrForObjectDetection` that can be
    found in the transformer library as well.

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
    """

    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        path_feature_extractor_config_json: str,
        categories: Mapping[str, TypeOrStr],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        Set up the predictor.
        :param path_config_json: The path to the json config.
        :param path_weights: The path to the model checkpoint.
        :param path_feature_extractor_config_json: The path to the feature extractor config.
        :param categories: A dict with key (indices) and values (category names).
        :param device: "cpu" or "cuda". If not specified will auto select depending on what is available
        :param filter_categories: The model might return objects that are not supposed to be predicted and that should
                                  be filtered. Pass a list of category names that must not be returned
        """
        super().__init__(categories, filter_categories)

        self.path_config = path_config_json
        self.path_weights = path_weights
        self.path_feature_extractor_config = path_feature_extractor_config_json

        self.name = self.get_name(self.path_weights)
        self.model_id = self.get_model_id()

        self.config = self.get_config(path_config_json)

        self.hf_detr_predictor = self.get_model(self.path_weights, self.config)
        self.feature_extractor = self.get_pre_processor(self.path_feature_extractor_config)

        self.device = get_torch_device(device)
        self.hf_detr_predictor.to(self.device)

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
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
    def get_model(path_weights: str, config: PretrainedConfig) -> TableTransformerForObjectDetection:
        """
        Builds the Detr model

        :param path_weights: The path to the model checkpoint.
        :param config: `PretrainedConfig`
        :return: TableTransformerForObjectDetection instance
        """
        return TableTransformerForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=config
        )

    @staticmethod
    def get_pre_processor(path_feature_extractor_config: str) -> DetrFeatureExtractor:
        """
        Builds the feature extractor

        :return: DetrFeatureExtractor
        """
        return AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=path_feature_extractor_config)

    @staticmethod
    def get_config(path_config: str) -> PretrainedConfig:
        """
        Builds the config

        :param path_config: The path to the json config.
        :return: PretrainedConfig instance
        """
        config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=path_config)
        config.use_timm_backbone = True
        config.threshold = 0.1
        config.nms_threshold = 0.05
        return config

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> HFDetrDerivedDetector:
        return self.__class__(
            self.path_config, self.path_weights, self.path_feature_extractor_config, self.categories, self.device
        )

    @staticmethod
    def get_wrapped_model(
        path_config_json: str, path_weights: str, device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None
    ) -> TableTransformerForObjectDetection:
        """
        Get the wrapped model

        :param path_config_json: The path to the json config.
        :param path_weights: The path to the model checkpoint.
        :param device: "cpu" or "cuda". If not specified will auto select depending on what is available
        :return: TableTransformerForObjectDetection instance
        """
        config = HFDetrDerivedDetector.get_config(path_config_json)
        hf_detr_predictor = HFDetrDerivedDetector.get_model(path_weights, config)
        device = get_torch_device()
        return hf_detr_predictor.to(device)
