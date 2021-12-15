# -*- coding: utf-8 -*-
# File: tpdetect.py

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
TP Faster RCNN model as predictor for Deep-Doctection pipeline
"""

from copy import copy
from typing import Dict, Optional, List, Union

from ..utils.metacfg import set_config_by_yaml
from ..utils.detection_types import ImageType, Requirement
from .base import ObjectDetector, DetectionResult
from .tp.tfutils import tensorpack_available

if tensorpack_available():
    from .tp.tpcompat import TensorpackPredictor
    from .tp.tpfrcnn.config.config import model_frcnn_config
    from .tp.tpfrcnn.modeling.generalized_rcnn import ResNetFPNModel
    from .tp.tpfrcnn.predict import tp_predict_image
    from .tp.tfutils import get_tensorpack_requirement


class TPFrcnnDetector(TensorpackPredictor, ObjectDetector):
    """
    Tensorpack Faster-RCNN implementation with FPN and Cascade-RCNN. The backbones Resnet-50, Resnet-101 and their
    Resnext counterparts are also available. Normalization options (group normalization, synchronized batch
    normalization) are also available.

    Current masks are not included in the data model. However, Mask-RCNN is implemented in this version.

    There are hardly any adjustments to the original implementation of Tensorpack. As post-processing, another round
    of NMS can be carried out for the output, which operates on a class-agnostic basis. For a discussion, see
    https://github.com/facebookresearch/detectron2/issues/978 .
    """

    def __init__(
        self,
        path_yaml: str,
        path_weights: str,
        categories: Dict[str, str],
        config_overwrite: Optional[List[str]] = None,
        ignore_mismatch: bool = False,
    ):
        """
        Set up the predictor

        The configuration of the model is stored in a yaml-file, which needs to be passed through. For more details,
        please check

        https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/config.py .

        Mask-Mode can be used as well here.

        :param path_yaml: The path to the yaml config
        :param path_weights: The path to the model checkpoint
        :param categories: A dict with key (indices) and values (category names). Index 0 must be reserved for a
                           dummy 'BG' category.
        :param config_overwrite: Overwrite some hyperparameters defined by the yaml file with some new values. E.g.
                                 ["OUTPUT.FRCNN_NMS_THRESH=0.3","OUTPUT.RESULT_SCORE_THRESH=0.6"]
        :param ignore_mismatch: When True will ignore mismatches between checkpoint weights and models. This is needed
                                if a pre-trained model is to be fine-tuned on a custom dataset.
        """

        self.categories = copy(categories)
        model = TPFrcnnDetector.set_model(path_yaml, categories, config_overwrite)
        super().__init__(model, path_weights, ignore_mismatch)
        assert self._number_gpus > 0, "Model currently only support inference with GPU"

    @staticmethod
    def set_model(
        path_yaml: str, categories: Dict[str, str], config_overwrite: Union[List[str], None]
    ) -> ResNetFPNModel:
        """
        Calls all necessary methods to build TP ResNetFPNModel

        :param path_yaml: path to the model config
        :param categories: A dict of categories with indices as keys
        :param config_overwrite: A list with special config attributes reset (consult meth: __init__)
        :return: The FPNResnet model.
        """

        if config_overwrite is None:
            config_overwrite = []

        hyper_param_config = set_config_by_yaml(path_yaml)

        if len(config_overwrite):
            hyper_param_config.update_args(config_overwrite)

        model_frcnn_config(config=hyper_param_config, categories=categories)
        return ResNetFPNModel(config=hyper_param_config)

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Prediction per image.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        detection_results = tp_predict_image(
            np_img,
            self.tp_predictor,
            self._model.cfg.PREPROC.SHORT_EDGE_SIZE,
            self._model.cfg.PREPROC.MAX_SIZE,
            self._model.cfg.MRCNN.ACCURATE_PASTE,
        )
        return self._map_category_names(detection_results)

    def _map_category_names(self, detection_results: List[DetectionResult]) -> List[DetectionResult]:
        """
        Populating category names to detection results

        :param detection_results: list of detection results
        :return: List of detection results with attribute class_name populated
        """
        for result in detection_results:
            result.class_name = self._model.cfg.DATA.CLASS_DICT[str(result.class_id)]
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_tensorpack_requirement()]
