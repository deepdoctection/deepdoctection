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
TP Faster-RCNN model
"""
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from ..utils.file_utils import get_tensorflow_requirement, get_tensorpack_requirement
from ..utils.metacfg import set_config_by_yaml
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..utils.types import PathLikeOrStr, PixelValues, Requirement
from .base import DetectionResult, ModelCategories, ObjectDetector
from .tp.tpcompat import TensorpackPredictor
from .tp.tpfrcnn.config.config import model_frcnn_config
from .tp.tpfrcnn.modeling.generalized_rcnn import ResNetFPNModel
from .tp.tpfrcnn.predict import tp_predict_image


class TPFrcnnDetectorMixin(ObjectDetector, ABC):
    """Base class for TP FRCNN detector. This class only implements the basic wrapper functions"""

    def __init__(self, categories: Mapping[int, TypeOrStr], filter_categories: Optional[Sequence[TypeOrStr]] = None):
        categories = {k: get_type(v) for k, v in categories.items()}
        categories.update({0: get_type("background")})
        self.categories = ModelCategories(categories)
        if filter_categories:
            self.categories.filter_categories = tuple(get_type(cat) for cat in filter_categories)

    def _map_category_names(self, detection_results: list[DetectionResult]) -> list[DetectionResult]:
        """
        Populating category names to detection results

        :param detection_results: list of detection results
        :return: List of detection results with attribute class_name populated
        """
        filtered_detection_result: list[DetectionResult] = []
        for result in detection_results:
            result.class_name = self.categories.categories.get(
                result.class_id if result.class_id else -1, DefaultType.DEFAULT_TYPE
            )
            if result.class_name != DefaultType.DEFAULT_TYPE:
                filtered_detection_result.append(result)
        return filtered_detection_result

    @staticmethod
    def get_name(path_weights: PathLikeOrStr, architecture: str) -> str:
        """Returns the name of the model"""
        return f"Tensorpack_{architecture}" + "_".join(Path(path_weights).parts[-2:])

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)


class TPFrcnnDetector(TensorpackPredictor, TPFrcnnDetectorMixin):
    """
    Tensorpack Faster-RCNN implementation with FPN and optional Cascade-RCNN. The backbones Resnet-50, Resnet-101 and
    their Resnext counterparts are also available. Normalization options (group normalization, synchronized batch
    normalization) for backbone in FPN can be chosen as well.

    Currently, masks are not included in the data model. However, Mask-RCNN is implemented in this version.

    There are hardly any adjustments to the original implementation of Tensorpack. As post-processing, another round
    of NMS can be carried out for the output, which operates on a class-agnostic basis. For a discussion, see
    <https://github.com/facebookresearch/detectron2/issues/978> .

        config_path = ModelCatalog.get_full_path_configs("dd/tp/conf_frcnn_rows.yaml")
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs("item/model-162000.data-00000-of-00001")
        categories = ModelCatalog.get_profile("item/model-162000.data-00000-of-00001").categories

        tp_predictor = TPFrcnnDetector("tp_frcnn", config_path,weights_path,categories)  # first argument is only a name
        detection_results = tp_predictor.predict(bgr_image_np_array)

    """

    def __init__(
        self,
        path_yaml: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        config_overwrite: Optional[list[str]] = None,
        ignore_mismatch: bool = False,
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        Set up the predictor.

        The configuration of the model is stored in a yaml-file, which needs to be passed through. For more details,
        please check

        <https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/config.py> .

        Mask-Mode could be used as well here provided the data structure is established.

        :param path_yaml: The path to the yaml config
        :param path_weights: The path to the model checkpoint
        :param categories: A dict with key (indices) and values (category names). Index 0 must be reserved for a
                           dummy 'BG' category.
        :param config_overwrite: Overwrite some hyperparameters defined by the yaml file with some new values. E.g.
                                 ["OUTPUT.FRCNN_NMS_THRESH=0.3","OUTPUT.RESULT_SCORE_THRESH=0.6"]
        :param ignore_mismatch: When True will ignore mismatches between checkpoint weights and models. This is needed
                                if a pre-trained model is to be fine-tuned on a custom dataset.
        :param filter_categories: The model might return objects that are not supposed to be predicted and that should
                                  be filtered. Pass a list of category names that must not be returned
        """
        self.path_yaml = Path(path_yaml)
        self.config_overwrite = config_overwrite

        model = TPFrcnnDetector.get_wrapped_model(path_yaml, categories, config_overwrite)
        TensorpackPredictor.__init__(self, model, path_weights, ignore_mismatch)
        TPFrcnnDetectorMixin.__init__(self, categories, filter_categories)

        self.name = self.get_name(path_weights, self._model.cfg.TAG)
        self.model_id = self.get_model_id()

    @staticmethod
    def get_wrapped_model(
        path_yaml: PathLikeOrStr, categories: Mapping[int, TypeOrStr], config_overwrite: Union[list[str], None]
    ) -> ResNetFPNModel:
        """
        Calls all necessary methods to build TP ResNetFPNModel

        :param path_yaml: path to the model config
        :param categories: A dict of categories with indices as keys
        :param config_overwrite: A list with special config attributes reset (consult  __init__)
        :return: The FPNResnet model.
        """

        if config_overwrite is None:
            config_overwrite = []

        hyper_param_config = set_config_by_yaml(path_yaml)

        hyper_param_config.freeze(freezed=False)
        if config_overwrite:
            hyper_param_config.update_args(config_overwrite)
        hyper_param_config.freeze()

        model_frcnn_config(config=hyper_param_config, categories=categories, print_summary=False)
        return ResNetFPNModel(config=hyper_param_config)

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
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

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_tensorflow_requirement(), get_tensorpack_requirement()]

    def clone(self) -> TPFrcnnDetector:
        return self.__class__(
            path_yaml=self.path_yaml,
            path_weights=self.path_weights,
            categories=dict(self.categories.get_categories()),
            config_overwrite=self.config_overwrite,
            ignore_mismatch=self.ignore_mismatch,
            filter_categories=self.categories.filter_categories,
        )

    def clear_model(self) -> None:
        self.tp_predictor = None
