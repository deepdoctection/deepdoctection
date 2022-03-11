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
Compatibility classes and methods related to Tensorpack package
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from tensorpack.predict import OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit

# pylint: disable=import-error
from tensorpack.train.model_desc import ModelDesc
from tensorpack.utils.gpu import get_num_gpu

from deepdoctection.utils.metacfg import AttrDict

# pylint: enable=import-error


class ModelDescWithConfig(ModelDesc, ABC):  # type: ignore  # pylint: disable=R0903
    """
    A wrapper for Tensorpack ModelDesc for bridging the gap between Tensorpack and DD API. Only for storing a
    configuration of hyperparameters and maybe training settings.
    """

    def __init__(self, config: AttrDict) -> None:
        """
        :param config: Config setting
        """
        super().__init__()
        self.cfg = config

    def get_inference_tensor_names(self) -> Tuple[List[str], List[str]]:
        """
        Returns lists of tensor names to be used to create an inference callable. "build_graph" must create tensors
        of these names when called under inference context.

        :return: Tuple of list input and list output names. The names must coincide with tensor within the model.
        """
        raise NotImplementedError


class TensorpackPredictor(ABC):
    """
    The base class for wrapping a Tensorpack predictor. It takes a ModelDescWithConfig from Tensorpack and weights and
    builds a Tensorpack offline predictor (e.g. a default session will be generated when initializing).

    Two abstract methods need to be implemented:

        - :meth:`set_model` generates a ModelDescWithConfig from the input .yaml file and possible manual adaptions of
          the configuration.

        - :meth:`predict` the interface of this class for calling the OfflinePredictor and returning the results. This
          method will be used throughout in the context of pipeline components. If there are some pre- or
          post-processing steps, you can place them here. However, do not convert the returned results into DD objects
          as there is an explicit class available for this.
    """

    def __init__(self, model: ModelDescWithConfig, path_weights: str, ignore_mismatch: bool) -> None:
        """
        :param model: Model, either as ModelDescWithConfig or derived from that class.
        :param path_weights: Model weights of the prediction config.
        :param ignore_mismatch: When True will ignore mismatches between checkpoint weights and models. This is needed
                                if a pre-trained model is to be fine-tuned on a custom dataset.
        """
        self._model = model
        self.path_weights = path_weights
        self.ignore_mismatch = ignore_mismatch
        self._number_gpus = get_num_gpu()
        self.predict_config = self._build_config()
        self.tp_predictor = self.get_predictor()

    def get_predictor(self) -> OfflinePredictor:
        """
        :return: Returns an OfflinePredictor.
        """
        return OfflinePredictor(self.predict_config)

    def _build_config(self) -> PredictConfig:
        predict_config = PredictConfig(
            model=self._model,
            session_init=SmartInit(self.path_weights, ignore_mismatch=self.ignore_mismatch),
            input_names=self._model.get_inference_tensor_names()[0],
            output_names=self._model.get_inference_tensor_names()[1],
        )

        return predict_config

    @staticmethod
    @abstractmethod
    def set_model(
        path_yaml: str, categories: Dict[str, str], config_overwrite: Union[List[str], None]
    ) -> ModelDescWithConfig:
        """
        Implement the config generation, its modification and instantiate a version of the model. See
        :class:`pipe.tpfrcnn.TPFrcnnDetector` for an example
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, np_img: Any) -> Any:
        """
        Implement, how :attr:`self.tp_predictor` is invoked and raw prediction results are generated. Do use only raw
        objects and nothing, which is related to the DD API.
        """
        raise NotImplementedError

    @property
    def model(self) -> ModelDescWithConfig:
        """
        model
        """
        return self._model
