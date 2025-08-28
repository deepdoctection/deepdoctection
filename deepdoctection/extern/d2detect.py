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
D2 `GeneralizedRCNN` models in PyTorch or Torchscript
"""

from __future__ import annotations

import io
import os
from abc import ABC
from copy import copy
from pathlib import Path
from typing import Literal, Mapping, Optional, Sequence, Union

import numpy as np
from lazy_imports import try_import

from ..utils.file_utils import get_detectron2_requirement, get_pytorch_requirement
from ..utils.metacfg import AttrDict, set_config_by_yaml
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..utils.transform import InferenceResize, ResizeTransform
from ..utils.types import PathLikeOrStr, PixelValues, Requirement
from .base import DetectionResult, ModelCategories, ObjectDetector
from .pt.nms import batched_nms
from .pt.ptutils import get_torch_device

with try_import() as pt_import_guard:
    import torch
    import torch.cuda
    from torch import nn  # pylint: disable=W0611

with try_import() as d2_import_guard:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import CfgNode, get_cfg  # pylint: disable=W0611
    from detectron2.modeling import GeneralizedRCNN, build_model  # pylint: disable=W0611
    from detectron2.structures import Instances  # pylint: disable=W0611


def _d2_post_processing(predictions: dict[str, Instances], nms_thresh_class_agnostic: float) -> dict[str, Instances]:
    """
    D2 postprocessing steps. Apply a class agnostic NMS.

    Args:
        predictions: Prediction outputs from the model.
        nms_thresh_class_agnostic: Nms being performed over all class predictions

    Returns:
        filtered Instances
    """
    instances = predictions["instances"]
    class_masks = torch.ones(instances.pred_classes.shape, dtype=torch.uint8)
    keep = batched_nms(instances.pred_boxes.tensor, instances.scores, class_masks, nms_thresh_class_agnostic)
    fg_instances_keep = instances[keep]
    return {"instances": fg_instances_keep}


def d2_predict_image(
    np_img: PixelValues,
    predictor: nn.Module,
    resizer: InferenceResize,
    nms_thresh_class_agnostic: float,
) -> list[DetectionResult]:
    """
    Run detection on one image. It will also handle the preprocessing internally which is using a custom resizing
    within some bounds.

    Args:
        np_img: ndarray
        predictor: torch nn module implemented in Detectron2
        resizer: instance for resizing the input image
        nms_thresh_class_agnostic: class agnostic NMS threshold

    Returns:
       list of `DetectionResult`s
    """
    height, width = np_img.shape[:2]
    resized_img = resizer.get_transform(np_img).apply_image(np_img)
    image = torch.as_tensor(resized_img.astype(np.float32).transpose(2, 0, 1))

    with torch.no_grad():
        inputs = {"image": image, "height": height, "width": width}
        predictions = predictor([inputs])[0]
        predictions = _d2_post_processing(predictions, nms_thresh_class_agnostic)
    instances = predictions["instances"]
    results = [
        DetectionResult(
            box=instances[k].pred_boxes.tensor.tolist()[0],
            score=instances[k].scores.tolist()[0],
            class_id=instances[k].pred_classes.tolist()[0],
        )
        for k in range(len(instances))
    ]
    return results


def d2_jit_predict_image(
    np_img: PixelValues, d2_predictor: nn.Module, resizer: InferenceResize, nms_thresh_class_agnostic: float
) -> list[DetectionResult]:
    """
    Run detection on an image using Torchscript. It will also handle the preprocessing internally which
    is using a custom resizing within some bounds. Moreover, and different from the setting where D2 is used
    it will also handle the resizing of the bounding box coords to the original image size.

    Args:
        np_img: ndarray
        d2_predictor: torchscript nn module
        resizer: instance for resizing the input image
        nms_thresh_class_agnostic: class agnostic nms threshold

    Returns:
        list of `DetectionResult`s
    """
    height, width = np_img.shape[:2]
    resized_img = resizer.get_transform(np_img).apply_image(np_img)
    new_height, new_width = resized_img.shape[:2]
    image = torch.as_tensor(resized_img.astype("float32").transpose(2, 0, 1))
    with torch.no_grad():
        boxes, classes, scores, _ = d2_predictor(image)
        class_masks = torch.ones(classes.shape, dtype=torch.uint8)
        keep = batched_nms(boxes, scores, class_masks, nms_thresh_class_agnostic).cpu()

        # The exported model does not contain the final resize step, so we need to add it manually here
        inverse_resizer = ResizeTransform(new_height, new_width, height, width, "VIZ")
        np_boxes = np.reshape(boxes.cpu().numpy(), (-1, 2))
        np_boxes = inverse_resizer.apply_coords(np_boxes)
        np_boxes = np.reshape(np_boxes, (-1, 4))

        np_boxes, classes, scores = np_boxes[keep], classes[keep], scores[keep]
        # If only one sample is left, it will squeeze np_boxes, so we need to expand it here
        if np_boxes.ndim == 1:
            np_boxes = np.expand_dims(np_boxes, axis=0)
    detect_result_list = []
    for box, label, score in zip(np_boxes, classes, scores):
        detect_result_list.append(DetectionResult(box=box.tolist(), class_id=label.item(), score=score.item()))
    return detect_result_list


class D2FrcnnDetectorMixin(ObjectDetector, ABC):
    """
    Base class for D2 Faster-RCNN implementation. This class only implements the basic wrapper functions
    """

    def __init__(
        self,
        categories: Mapping[int, TypeOrStr],
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        Args:
            categories: A dict with key (indices) and values (category names). Index 0 must be reserved for a
                        dummy 'BG' category.
                        Note:
                             This convention is different from the builtin D2 framework, where models in the model
                             zoo are trained with `BG` class having the highest index.
            filter_categories: The model might return objects that are not supposed to be predicted and that should
                               be filtered. Pass a list of category names that must not be returned
        """

        self.categories = ModelCategories(init_categories=categories)
        if filter_categories:
            self.categories.filter_categories = tuple(get_type(cat) for cat in filter_categories)

    def _map_category_names(self, detection_results: list[DetectionResult]) -> list[DetectionResult]:
        """
        Populating category names to `DetectionResult`s

        Args:
            detection_results: list of `DetectionResult`s. Will also filter categories
        Returns:
            List of `DetectionResult`s with attribute `class_name` populated
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

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)

    @staticmethod
    def get_inference_resizer(min_size_test: int, max_size_test: int) -> InferenceResize:
        """Returns the resizer for the inference

        Args:
            min_size_test: minimum size of the resized image
            max_size_test: maximum size of the resized image
        """
        return InferenceResize(min_size_test, max_size_test)

    @staticmethod
    def get_name(path_weights: PathLikeOrStr, architecture: str) -> str:
        """Returns the name of the model"""
        return f"detectron2_{architecture}" + "_".join(Path(path_weights).parts[-2:])


class D2FrcnnDetector(D2FrcnnDetectorMixin):
    """
    D2 Faster-RCNN implementation with all the available backbones, normalizations throughout the model
    as well as FPN, optional Cascade-RCNN and many more.

    Currently, masks are not included in the data model.

    Note:
        There are no adjustment to the original implementation of Detectron2. Only one post-processing step is followed
        by the standard D2 output that takes into account of the situation that detected objects are disjoint. For more
        infos on this topic, see <https://github.com/facebookresearch/detectron2/issues/978>.

    Example:
        ```python
            config_path = ModelCatalog.get_full_path_configs("dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml")
            weights_path = ModelDownloadManager.maybe_download_weights_and_configs("item/d2_model-800000-layout.pkl")
            categories = ModelCatalog.get_profile("item/d2_model-800000-layout.pkl").categories

            d2_predictor = D2FrcnnDetector(config_path,weights_path,categories,device="cpu")

            detection_results = d2_predictor.predict(bgr_image_np_array)
        ```
    """

    def __init__(
        self,
        path_yaml: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        config_overwrite: Optional[list[str]] = None,
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        Set up the predictor.

        The configuration of the model uses the full stack of build model tools of D2. For more information
        please check <https://detectron2.readthedocs.io/en/latest/tutorials/models.html#build-models-from-yacs-config>.

        Args:
            path_yaml: The path to the yaml config. If the model is built using several config files, always use
                       the highest level .yaml file.
            path_weights: The path to the model checkpoint.
            categories: A dict with key (indices) and values (category names). Index 0 must be reserved for a
                        dummy `BG` category. Note, that this convention is different from the builtin D2 framework,
                        where models in the model zoo are trained with `BG` class having the highest index.
            config_overwrite:  Overwrite some hyperparameters defined by the yaml file with some new values. E.g.
                               `["OUTPUT.FRCNN_NMS_THRESH=0.3","OUTPUT.RESULT_SCORE_THRESH=0.6"]`.
            device: "cpu" or "cuda". If not specified will auto select depending on what is available
            filter_categories: The model might return objects that are not supposed to be predicted and that should
                               be filtered. Pass a list of category names that must not be returned
        """
        super().__init__(categories, filter_categories)

        self.path_weights = Path(path_weights)
        self.path_yaml = Path(path_yaml)

        config_overwrite = config_overwrite if config_overwrite else []
        self.config_overwrite = config_overwrite
        self.device = get_torch_device(device)

        d2_conf_list = self._get_d2_config_list(path_weights, config_overwrite)
        self.cfg = self._set_config(path_yaml, d2_conf_list, self.device)

        self.name = self.get_name(path_weights, self.cfg.MODEL.META_ARCHITECTURE)
        self.model_id = self.get_model_id()

        self.d2_predictor = self._set_model(self.cfg)
        self._instantiate_d2_predictor(self.d2_predictor, path_weights)
        self.resizer = self.get_inference_resizer(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)

    @staticmethod
    def _set_config(path_yaml: PathLikeOrStr, d2_conf_list: list[str], device: torch.device) -> CfgNode:
        cfg = get_cfg()
        # additional attribute with default value, so that the true value can be loaded from the configs
        cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.1
        cfg.merge_from_file(os.fspath(path_yaml))
        cfg.merge_from_list(d2_conf_list)
        cfg.MODEL.DEVICE = str(device)
        cfg.freeze()
        return cfg

    @staticmethod
    def _set_model(config: CfgNode) -> GeneralizedRCNN:
        """
        Build the model. It uses the available built-in tools of D2

        Args:
            config: Model config

        Returns:
            `GeneralizedRCNN` model
        """
        return build_model(config.clone()).eval()

    @staticmethod
    def _instantiate_d2_predictor(wrapped_model: GeneralizedRCNN, path_weights: PathLikeOrStr) -> None:
        checkpointer = DetectionCheckpointer(wrapped_model)
        checkpointer.load(os.fspath(path_weights))

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Prediction per image.

        Args:
            np_img: image as `np.array`

        Returns:
            A list of `DetectionResult`s
        """
        detection_results = d2_predict_image(
            np_img,
            self.d2_predictor,
            self.resizer,
            self.cfg.NMS_THRESH_CLASS_AGNOSTIC,
        )
        return self._map_category_names(detection_results)

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement(), get_detectron2_requirement()]

    def clone(self) -> D2FrcnnDetector:
        return self.__class__(
            self.path_yaml,
            self.path_weights,
            self.categories.get_categories(),
            self.config_overwrite,
            self.device,
            self.categories.filter_categories,
        )

    @staticmethod
    def get_wrapped_model(
        path_yaml: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        config_overwrite: list[str],
        device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
    ) -> GeneralizedRCNN:
        """
        Get the wrapped model. Useful, if one does not want to build the wrapper but only needs the instantiated model.

        Example:
            ```python
            path_yaml = ModelCatalog.get_full_path_configs("dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml")
            weights_path = ModelDownloadManager.maybe_download_weights_and_configs("item/d2_model-800000-layout.pkl")
            model = D2FrcnnDetector.get_wrapped_model(path_yaml,weights_path,["OUTPUT.FRCNN_NMS_THRESH=0.3",
                                                                              "OUTPUT.RESULT_SCORE_THRESH=0.6"],
                                                                              "cpu")
            detect_result_list = d2_predict_image(np_img,model,InferenceResize(800,1333),0.3)
            ```

        Args:
            path_yaml: The path to the `yaml` config. If the model is built using several config files, always use
                       the highest level `.yaml` file.
            path_weights: The path to the model checkpoint.
            config_overwrite: Overwrite some hyperparameters defined by the yaml file with some new values. E.g.
                              `["OUTPUT.FRCNN_NMS_THRESH=0.3","OUTPUT.RESULT_SCORE_THRESH=0.6"]`.
            device: "cpu" or "cuda". If not specified will auto select depending on what is available

        Returns:
            `GeneralizedRCNN` model
        """

        device = get_torch_device(device)
        d2_conf_list = D2FrcnnDetector._get_d2_config_list(path_weights, config_overwrite)
        cfg = D2FrcnnDetector._set_config(path_yaml, d2_conf_list, device)
        model = D2FrcnnDetector._set_model(cfg)
        D2FrcnnDetector._instantiate_d2_predictor(model, path_weights)
        return model

    @staticmethod
    def _get_d2_config_list(path_weights: PathLikeOrStr, config_overwrite: list[str]) -> list[str]:
        d2_conf_list = ["MODEL.WEIGHTS", os.fspath(path_weights)]
        config_overwrite = config_overwrite if config_overwrite else []
        for conf in config_overwrite:
            key, val = conf.split("=", maxsplit=1)
            d2_conf_list.extend([key, val])
        return d2_conf_list

    def clear_model(self) -> None:
        self.d2_predictor = None


class D2FrcnnTracingDetector(D2FrcnnDetectorMixin):
    """
    D2 Faster-RCNN exported torchscript model. Using this predictor has the advantage that Detectron2 does not have to
    be installed. The associated config setting only contains parameters that are involved in pre-and post-processing.
    Depending on running the model with CUDA or on a CPU, it will need the appropriate exported model.

    Note:
        There are no adjustment to the original implementation of Detectron2. Only one post-processing step is followed
        by the standard D2 output that takes into account of the situation that detected objects are disjoint. For more
        infos on this topic, see <https://github.com/facebookresearch/detectron2/issues/978>.

    Example:
        ```python
        config_path = ModelCatalog.get_full_path_configs("dd/d2/item/CASCADE_RCNN_R_50_FPN_GN.yaml")
        weights_path = ModelDownloadManager.maybe_download_weights_and_configs("item/d2_model-800000-layout.pkl")
        categories = ModelCatalog.get_profile("item/d2_model-800000-layout.pkl").categories

        d2_predictor = D2FrcnnDetector(config_path,weights_path,categories)

        detection_results = d2_predictor.predict(bgr_image_np_array)
        ```
    """

    def __init__(
        self,
        path_yaml: PathLikeOrStr,
        path_weights: PathLikeOrStr,
        categories: Mapping[int, TypeOrStr],
        config_overwrite: Optional[list[str]] = None,
        filter_categories: Optional[Sequence[TypeOrStr]] = None,
    ):
        """
        Set up the Torchscript predictor.

        Args:
            path_yaml: The path to the `yaml` config. If the model is built using several config files, always use
                       the highest level `.yaml` file.
            path_weights: The path to the model checkpoint.
            categories: A dict with key (indices) and values (category names). Index 0 must be reserved for a
                        dummy `BG` category. Note, that this convention is different from the builtin D2 framework,
                        where models in the model zoo are trained with `BG` class having the highest index.
            config_overwrite:  Overwrite some hyperparameters defined by the yaml file with some new values. E.g.
                               `["OUTPUT.FRCNN_NMS_THRESH=0.3","OUTPUT.RESULT_SCORE_THRESH=0.6"]`.
            filter_categories: The model might return objects that are not supposed to be predicted and that should
                               be filtered. Pass a list of category names that must not be returned
        """

        super().__init__(categories, filter_categories)

        self.path_weights = Path(path_weights)
        self.path_yaml = Path(path_yaml)

        self.config_overwrite = copy(config_overwrite)
        self.cfg = self._set_config(self.path_yaml, self.path_weights, self.config_overwrite)

        self.name = self.get_name(path_weights, self.cfg.MODEL.META_ARCHITECTURE)
        self.model_id = self.get_model_id()

        self.resizer = self.get_inference_resizer(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST)
        self.d2_predictor = self.get_wrapped_model(self.path_weights)

    @staticmethod
    def _set_config(
        path_yaml: PathLikeOrStr, path_weights: PathLikeOrStr, config_overwrite: Optional[list[str]]
    ) -> AttrDict:
        cfg = set_config_by_yaml(path_yaml)
        config_overwrite = config_overwrite if config_overwrite else []
        config_overwrite.extend([f"MODEL.WEIGHTS={os.fspath(path_weights)}"])
        cfg.freeze(False)
        cfg.update_args(config_overwrite)
        cfg.freeze()
        return cfg

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Prediction per image.

        Args:
            np_img: image as `np.array`

        Returns:
            A list of `DetectionResult`s
        """
        detection_results = d2_jit_predict_image(
            np_img,
            self.d2_predictor,
            self.resizer,
            self.cfg.NMS_THRESH_CLASS_AGNOSTIC,
        )
        return self._map_category_names(detection_results)

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_pytorch_requirement()]

    def clone(self) -> D2FrcnnTracingDetector:
        return self.__class__(
            self.path_yaml,
            self.path_weights,
            self.categories.get_categories(),
            self.config_overwrite,
            self.categories.filter_categories,
        )

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)

    @staticmethod
    def get_wrapped_model(path_weights: PathLikeOrStr) -> torch.jit.ScriptModule:
        """
        Get the wrapped model. Useful, if one do not want to build the wrapper but only needs the instantiated model.

        Args:
            path_weights: The path to the model checkpoint. The model must be exported as Torchscript.

        Returns:
            `torch.jit.ScriptModule` model
        """
        with open(path_weights, "rb") as file:
            buffer = io.BytesIO(file.read())
        # Load all tensors to the original device
        return torch.jit.load(buffer)

    def clear_model(self) -> None:
        self.d2_predictor = None  # type: ignore
