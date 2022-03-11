# -*- coding: utf-8 -*-
# File: d2detect.py

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
D2 Faster Frcnn model as predictor for deepdoctection pipeline
"""

from copy import copy
from typing import Dict, List, Optional

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import (
    detectron2_available,
    get_detectron2_requirement,
    get_pytorch_requirement,
    pytorch_available,
)
from .base import DetectionResult, ObjectDetector, PredictorBase
from .d2.d2 import d2_predict_image

if pytorch_available():
    import torch.cuda  # type: ignore

if detectron2_available():
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import CfgNode, get_cfg  # pylint: disable=W0611
    from detectron2.modeling import GeneralizedRCNN, build_model  # pylint: disable=W0611


class D2FrcnnDetector(ObjectDetector):
    """
    D2 Faster-RCNN implementation with all the available backbones, normalizations throughout the model
    as well as FPN, optional Cascade-RCNN and many more.

    Currently, masks are not included in the data model.

    There are no adjustment to the original implementation of Detectron2. Only one post-processing step is followed by
    the standard D2 output that takes into account of the situation that detected objects are disjoint. For more infos
    on this topic, see https://github.com/facebookresearch/detectron2/issues/978 .
    """

    def __init__(
        self,
        path_yaml: str,
        path_weights: str,
        categories: Dict[str, str],
        config_overwrite: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        """
        Set up the predictor.

        The configuration of the model uses the full stack of build model tools of D2. For more information
        please check https://detectron2.readthedocs.io/en/latest/tutorials/models.html#build-models-from-yacs-config .

        :param path_yaml: The path to the yaml config. If the model is built using several config files, always use
                          the highest level .yaml file.
        :param path_weights: The path to the model checkpoint.
        :param categories: A dict with key (indices) and values (category names). Index 0 must be reserved for a
                           dummy 'BG' category. Note, that this convention is different from the builtin D2 framework,
                           where models in the model zoo are trained with 'BG' class having the highest index.
        :param config_overwrite:  Overwrite some hyper parameters defined by the yaml file with some new values. E.g.
                                 ["OUTPUT.FRCNN_NMS_THRESH=0.3","OUTPUT.RESULT_SCORE_THRESH=0.6"].
        """

        self._categories_d2 = self._map_to_d2_categories(copy(categories))
        if config_overwrite is None:
            config_overwrite = []
        self.path_weights = path_weights
        d2_conf_list = ["MODEL.WEIGHTS", path_weights]
        for conf in config_overwrite:
            key, val = conf.split("=", maxsplit=1)
            d2_conf_list.extend([key, val])

        self.path_yaml = path_yaml
        self.categories = copy(categories)
        self.config_overwrite = config_overwrite
        self.device = device
        self.cfg = self._set_config(path_yaml, d2_conf_list, device)
        self.d2_predictor = D2FrcnnDetector.set_model(self.cfg)
        self._instantiate_d2_predictor()

    @staticmethod
    def _set_config(path_yaml: str, d2_conf_list: List[str], device: str) -> "CfgNode":
        cfg = get_cfg()
        # additional attribute with default value, so that the true value can be loaded from the configs
        cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.1
        cfg.merge_from_file(path_yaml)
        cfg.merge_from_list(d2_conf_list)
        if not torch.cuda.is_available() or device == "cpu":
            cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()
        return cfg

    @staticmethod
    def set_model(config: "CfgNode") -> "GeneralizedRCNN":
        """
        Build the D2 model. It uses the available builtin tools of D2

        :param config: Model config
        :return: The GeneralizedRCNN model
        """
        return build_model(config.clone()).eval()

    def _instantiate_d2_predictor(self) -> None:
        checkpointer = DetectionCheckpointer(self.d2_predictor)
        checkpointer.load(self.cfg.MODEL.WEIGHTS)

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Prediction per image.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        detection_results = d2_predict_image(
            np_img,
            self.d2_predictor,
            self.cfg.INPUT.MIN_SIZE_TEST,
            self.cfg.INPUT.MAX_SIZE_TEST,
            self.cfg.NMS_THRESH_CLASS_AGNOSTIC,
        )
        return self._map_category_names(detection_results)

    def _map_category_names(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        """
        Populating category names to detection results

        :param detection_results: list of detection results
        :return: List of detection results with attribute class_name populated
        """
        for result in detection_results:
            result.class_name = self._categories_d2[str(result.class_id)]
            result.class_id = result.class_id + 1
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_pytorch_requirement(), get_detectron2_requirement()]

    @classmethod
    def _map_to_d2_categories(cls, categories: Dict[str, str]) -> Dict[str, str]:
        return {str(int(k) - 1): v for k, v in categories.items()}

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_yaml, self.path_weights, self.categories, self.config_overwrite, self.device)
