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

from typing import Any, List, Literal, Mapping, Optional, Sequence

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import (
    get_pytorch_requirement,
    get_transformers_requirement,
    pytorch_available,
    transformers_available,
)
from ..utils.settings import TypeOrStr, get_type
from .base import DetectionResult, ObjectDetector
from .pt.ptutils import set_torch_auto_device

if pytorch_available():
    import torch
    from torchvision.ops import boxes as box_ops

if transformers_available():
    from transformers import (
        AutoFeatureExtractor,
        DetrFeatureExtractor,
        PretrainedConfig,
        TableTransformerForObjectDetection,
    )


def _detr_post_processing(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, nms_thresh: float):
    return box_ops.batched_nms(boxes.float(), scores, labels, nms_thresh)


def detr_predict_image(
    np_img: ImageType,
    predictor: TableTransformerForObjectDetection,
    feature_extractor: Any,
    device: Literal["cpu", "cuda"],
    threshold: float,
    nms_threshold: float,
) -> List[DetectionResult]:
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


class HFDetrDerivedDetector(ObjectDetector):
    def __init__(
        self,
        path_config_json: str,
        path_weights: str,
        path_feature_extractor_config_json: str,
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):

        self.name = "Detr"
        self.categories = {idx: get_type(cat) for idx, cat in categories.items()}
        self.path_config = path_config_json
        self.path_weights = path_weights
        self.path_feature_extractor_config = path_feature_extractor_config_json
        self.config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=self.path_config)
        self.config.threshold = 0.1
        self.config.nms_threshold = 0.05
        self.hf_detr_predictor = self.set_model(path_weights)
        self.feature_extractor = self.set_pre_processor()

        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.hf_detr_predictor.to(self.device)
        if filter_categories:
            filter_categories = [get_type(cat) for cat in filter_categories]
        self.filter_categories = filter_categories

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

    def set_model(self, path_weights: str) -> TableTransformerForObjectDetection:
        """
        Builds the Detr model

        :param path_weights: weights
        :return: TableTransformerForObjectDetection instance
        """
        return TableTransformerForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=path_weights, config=self.config
        )

    def set_pre_processor(self) -> DetrFeatureExtractor:
        """
        Builds the feature extractor

        :return: DetrFeatureExtractor
        """
        return AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=self.path_feature_extractor_config)

    def _map_category_names(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        """
        Populating category names to detection results. Will also filter categories

        :param detection_results: list of detection results
        :return: List of detection results with attribute class_name populated
        """
        filtered_detection_result: List[DetectionResult] = []
        for result in detection_results:
            result.class_name = self.categories[str(result.class_id + 1)]
            if isinstance(result.class_id, int):
                result.class_id += 1
            if self.filter_categories:
                if result.class_name not in self.filter_categories:
                    filtered_detection_result.append(result)
            else:
                filtered_detection_result.append(result)

        return filtered_detection_result

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> "HFDetrDerivedDetector":
        return self.__class__(
            self.path_config, self.path_weights, self.path_feature_extractor_config, self.categories, self.device
        )
