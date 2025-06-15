# -*- coding: utf-8 -*-
# File: tpcompat.py

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
Compatibility classes and methods related to Tensorpack package.

Info:
    This module provides compatibility classes and methods related to the Tensorpack package.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Union

from lazy_imports import try_import

from ...utils.metacfg import AttrDict
from ...utils.settings import ObjectTypes
from ...utils.types import PathLikeOrStr, PixelValues

with try_import() as import_guard:
    from tensorpack.predict import OfflinePredictor, PredictConfig  # pylint: disable=E0401
    from tensorpack.tfutils import SmartInit  # pylint: disable=E0401
    from tensorpack.train.model_desc import ModelDesc  # pylint: disable=E0401
    from tensorpack.utils.gpu import get_num_gpu  # pylint: disable=E0401

if not import_guard.is_successful():
    from ...utils.mocks import ModelDesc


class ModelDescWithConfig(ModelDesc, ABC):  # type: ignore
    """
    A wrapper for `Tensorpack ModelDesc` for bridging the gap between Tensorpack and DD API.

    Only for storing a configuration of hyperparameters and maybe training settings.


    """

    def __init__(self, config: AttrDict) -> None:
        """
        Args:
            config: Config setting.
        """
        super().__init__()
        self.cfg = config

    def get_inference_tensor_names(self) -> tuple[list[str], list[str]]:
        """
        Returns lists of tensor names to be used to create an inference callable.

        `build_graph` must create tensors of these names when called under inference context.

        Returns:
            Tuple of list input and list output names. The names must coincide with tensor within the model.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError()


class TensorpackPredictor(ABC):
    """
    The base class for wrapping a Tensorpack predictor. It takes a ModelDescWithConfig from Tensorpack and weights and
    builds a Tensorpack offline predictor (e.g. a default session will be generated when initializing).

    Two abstract methods need to be implemented:

        - `set_model` generates a ModelDescWithConfig from the input .yaml file and possible manual adaptions of
          the configuration.

        - `predict` the interface of this class for calling the OfflinePredictor and returning the results. This
          method will be used throughout in the context of pipeline components. If there are some pre- or
          post-processing steps, you can place them here. However, do not convert the returned results into DD objects
          as there is an explicit class available for this.
    """

    def __init__(self, model: ModelDescWithConfig, path_weights: PathLikeOrStr, ignore_mismatch: bool) -> None:
        """
        Args:
            model: Model, either as `ModelDescWithConfig` or derived from that class.
            path_weights: Model weights of the prediction config.
            ignore_mismatch: When True will ignore mismatches between checkpoint weights and models. This is needed
                if a pre-trained model is to be fine-tuned on a custom dataset.
        """
        self._model = model
        self.path_weights = Path(path_weights)
        self.ignore_mismatch = ignore_mismatch
        self._number_gpus = get_num_gpu()
        self.predict_config = self._build_config()
        self.tp_predictor = self.get_predictor()

    def get_predictor(self) -> OfflinePredictor:
        """
        Returns an `OfflinePredictor`.

        Returns:
            Returns an `OfflinePredictor`.
        """
        return OfflinePredictor(self.predict_config)

    def _build_config(self) -> PredictConfig:
        path_weights = os.fspath(self.path_weights) if os.fspath(self.path_weights) != "." else ""
        predict_config = PredictConfig(
            model=self._model,
            session_init=SmartInit(path_weights, ignore_mismatch=self.ignore_mismatch),
            input_names=self._model.get_inference_tensor_names()[0],
            output_names=self._model.get_inference_tensor_names()[1],
        )

        return predict_config

    @staticmethod
    @abstractmethod
    def get_wrapped_model(
        path_yaml: PathLikeOrStr, categories: Mapping[int, ObjectTypes], config_overwrite: Union[list[str], None]
    ) -> ModelDescWithConfig:
        """
        Implement the config generation, its modification and instantiate a version of the model.

        See `pipe.tpfrcnn.TPFrcnnDetector` for an example.

        Raises:
            NotImplementedError: If not implemented in subclass.

        Args:
            path_yaml: Path to the yaml file.
            categories: Mapping of categories.
            config_overwrite: List of config overwrites or None.

        Returns:
            An instance of `ModelDescWithConfig`.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, np_img: PixelValues) -> Any:
        """
        Implement how `self.tp_predictor` is invoked and raw prediction results are generated.

        Do use only raw objects and nothing, which is related to the DD API.

        Args:
            np_img: The input image as pixel values.

        Returns:
            Raw prediction results.

        Raises:
            NotImplementedError: If not implemented in subclass.
        """
        raise NotImplementedError()

    @property
    def model(self) -> ModelDescWithConfig:
        """
        model
        """
        return self._model
