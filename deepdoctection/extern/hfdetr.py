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

from typing import List, Literal, Mapping, Optional


from .base import DetectionResult, ObjectDetector
from .pt.ptutils import set_torch_auto_device

from ..utils.detection_types import Requirement, ImageType
from ..utils.file_utils import (
    get_pytorch_requirement,
    get_transformers_requirement,
    transformers_available,
)
from ..utils.settings import get_type, TypeOrStr, ObjectTypes

if transformers_available():
    from transformers import AutoFeatureExtractor, TableTransformerForObjectDetection, PretrainedConfig


def detr_predict_image(np_img: ImageType,
                       predictor: TableTransformerForObjectDetection,
                       feature_extractor,
                       device,
                       threshold) -> List[DetectionResult]:
    target_sizes = [np_img.shape[:2]]
    inputs = feature_extractor(images=np_img, return_tensors="pt")
    inputs.data["pixel_values"]= inputs.data["pixel_values"].to(device)
    inputs.data["pixel_mask"] = inputs.data["pixel_mask"].to(device)
    outputs = predictor(**inputs)
    outputs["encoder_last_hidden_state"] = outputs["encoder_last_hidden_state"].to("cpu")
    outputs["last_hidden_state"] = outputs["last_hidden_state"].to("cpu")
    outputs["logits"] = outputs["logits"].to("cpu")
    outputs["pred_boxes"] = outputs["pred_boxes"].to("cpu")
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    return [DetectionResult(box=box.tolist(), score=score.item(), class_id=class_id.item()) for box, score, class_id
               in zip(results["boxes"], results["scores"], results["labels"])]


class HFDetrDerivedDetector(ObjectDetector):

    def __init__(self, path_config_json: str,
                       path_weights: str,
                       path_feature_extractor_config_json: str,
                       categories: Mapping[str, TypeOrStr],
                       device: Optional[Literal["cpu", "cuda"]] = None):

        self.name = "Detr"
        self.categories = categories
        self.path_config = path_config_json
        self.path_weights = path_weights
        self.path_feature_extractor_config = path_feature_extractor_config_json
        self.config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=self.path_config)
        self.config.threshold = 0.1
        self.model = TableTransformerForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=path_weights, config= self.config
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path=
                                                                      self.path_feature_extractor_config)

        if device is not None:
            self.device = device
        else:
            self.device = set_torch_auto_device()
        self.model.to(self.device)

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        results = detr_predict_image(np_img, self.model, self.feature_extractor, self.device, self.config.threshold)
        return self._map_category_names(results)

    def _map_category_names(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        """
        Populating category names to detection results

        :param detection_results: list of detection results
        :return: List of detection results with attribute class_name populated
        """
        for result in detection_results:
            result.class_name = self.categories[str(result.class_id + 1)]
            if isinstance(result.class_id, int):
                result.class_id += 1
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_transformers_requirement()]

    def clone(self) -> "HFDetrDerivedDetector":
        return self.__class__(self.path_config,
                              self.path_weights,
                              self.path_feature_extractor_config,
                              self.categories,
                              self.device)
