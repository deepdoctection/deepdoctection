# -*- coding: utf-8 -*-
# File: doctrocr.py

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
Deepdoctection wrappers for DocTr OCR text line detection and text recognition models
"""

from pathlib import Path
from typing import Any, List, Literal, Mapping, Optional, Tuple
from zipfile import ZipFile

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import (
    doctr_available,
    get_doctr_requirement,
    get_pytorch_requirement,
    get_tensorflow_requirement,
    get_tf_addons_requirements,
    pytorch_available,
    tf_addons_available,
    tf_available,
)
from ..utils.settings import LayoutType, ObjectTypes, TypeOrStr
from .base import DetectionResult, ObjectDetector, PredictorBase, TextRecognizer
from .pt.ptutils import set_torch_auto_device

if doctr_available() and ((tf_addons_available() and tf_available()) or pytorch_available()):
    from doctr.models.detection.predictor import DetectionPredictor  # pylint: disable=W0611
    from doctr.models.detection.zoo import detection_predictor
    from doctr.models.recognition.predictor import RecognitionPredictor  # pylint: disable=W0611
    from doctr.models.recognition.zoo import recognition_predictor

if pytorch_available():
    import torch

if tf_available():
    import tensorflow as tf  # type: ignore  # pylint: disable=E0401


def _set_device_str(device: Optional[str] = None) -> str:
    if device is not None:
        if tf_available():
            device = "/" + device.replace("cuda", "gpu") + ":0"
    elif pytorch_available():
        device = set_torch_auto_device()
    else:
        device = "/gpu:0"  # we impose to install tensorflow-gpu because of Tensorpack models
    return device


def _load_model(path_weights: str, doctr_predictor: Any, device: str) -> None:
    if pytorch_available():
        state_dict = torch.load(path_weights, map_location=device)
        for key in list(state_dict.keys()):
            state_dict["model." + key] = state_dict.pop(key)
        doctr_predictor.load_state_dict(state_dict)
        doctr_predictor.to(device)
    elif tf_available():
        # Unzip the archive
        params_path = Path(path_weights).parent
        is_zip_path = path_weights.endswith(".zip")
        if is_zip_path:
            with ZipFile(path_weights, "r") as file:
                file.extractall(path=params_path)
                doctr_predictor.model.load_weights(params_path / "weights")
        else:
            doctr_predictor.model.load_weights(path_weights)


def doctr_predict_text_lines(np_img: ImageType, predictor: "DetectionPredictor", device: str) -> List[DetectionResult]:
    """
    Generating text line DetectionResult based on Doctr DetectionPredictor.

    :param np_img: Image in np.array.
    :param predictor: `doctr.models.detection.predictor.DetectionPredictor`
    :param device: Will only be used in tensorflow settings. Either /gpu:0 or /cpu:0
    :return: A list of text line detection results (without text).
    """
    if tf_available() and device is not None:
        with tf.device(device):
            raw_output = predictor([np_img])
    else:
        raw_output = predictor([np_img])
    detection_results = [
        DetectionResult(
            box=box[:4].tolist(), class_id=1, score=box[4], absolute_coords=False, class_name=LayoutType.word
        )
        for box in raw_output[0]
    ]
    return detection_results


def doctr_predict_text(
    inputs: List[Tuple[str, ImageType]], predictor: "RecognitionPredictor", device: str
) -> List[DetectionResult]:
    """
    Calls Doctr text recognition model on a batch of numpy arrays (text lines predicted from a text line detector) and
    returns the recognized text as DetectionResult

    :param inputs: list of tuples containing the annotation_id of the input image and the numpy array of the cropped
                   text line
    :param predictor: `doctr.models.detection.predictor.RecognitionPredictor`
    :param device: Will only be used in tensorflow settings. Either /gpu:0 or /cpu:0
    :return: A list of DetectionResult containing recognized text.
    """

    uuids, images = list(zip(*inputs))
    if tf_available() and device is not None:
        with tf.device(device):
            raw_output = predictor(list(images))
    else:
        raw_output = predictor(list(images))
    detection_results = [
        DetectionResult(score=output[1], text=output[0], uuid=uuid) for uuid, output in zip(uuids, raw_output)
    ]
    return detection_results


class DoctrTextlineDetector(ObjectDetector):
    """
    A deepdoctection wrapper of DocTr text line detector. We model text line detection as ObjectDetector
    and assume to use this detector in a ImageLayoutService.
    DocTr supports several text line detection implementations but provides only a subset of pre-trained models.
    The most usable one for document OCR for which a pre-trained model exists is DBNet as described in “Real-time Scene
    Text Detection with Differentiable Binarization”, with a ResNet-50 backbone. This model can be used in either
    Tensorflow or PyTorch.
    Some other pre-trained models exist that have not been registered in `ModelCatalog`. Please check the DocTr library
    and organize the download of the pre-trained model by yourself.

    **Example:**

                 path_weights_tl = ModelDownloadManager.maybe_download_weights_and_configs("doctr/db_resnet50/pt
                 /db_resnet50-ac60cadc.pt")
                 # Use "doctr/db_resnet50/tf/db_resnet50-adcafc63.zip" for Tensorflow

                 categories = ModelCatalog.get_profile("doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt").categories
                 det = DoctrTextlineDetector("db_resnet50",path_weights_tl,categories,"cpu")
                 layout = ImageLayoutService(det,to_image=True, crop_image=True)

                 path_weights_tr = dd.ModelDownloadManager.maybe_download_weights_and_configs("doctr/crnn_vgg16_bn
                 /pt/crnn_vgg16_bn-9762b0b0.pt")
                 rec = DoctrTextRecognizer("crnn_vgg16_bn", path_weights_tr, "cpu")
                 text = TextExtractionService(rec, extract_from_roi="word")

                 analyzer = DoctectionPipe(pipeline_component_list=[layout,text])

                 path = "/path/to/image_dir"
                 df = analyzer.analyze(path = path)

                 for dp in df:
                     ...


    """

    def __init__(
        self,
        architecture: str,
        path_weights: str,
        categories: Mapping[str, TypeOrStr],
        device: Optional[Literal["cpu", "cuda"]] = None,
    ) -> None:
        self.name = "doctr_text_detector"
        self.architecture = architecture
        self.path_weights = path_weights
        self.doctr_predictor = detection_predictor(
            arch=self.architecture, pretrained=False, pretrained_backbone=False
        )  # we will be loading the model
        # later because there is no easy way in doctr to load a model by giving only a path to its weights
        self.categories = categories  # type: ignore
        self.device_input = device
        self.device = _set_device_str(device)
        self.load_model()

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Prediction per image.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        detection_results = doctr_predict_text_lines(np_img, self.doctr_predictor, self.device)
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        if tf_available():
            return [get_tensorflow_requirement(), get_doctr_requirement(), get_tf_addons_requirements()]
        if pytorch_available():
            return [get_pytorch_requirement(), get_doctr_requirement()]
        raise ModuleNotFoundError("Neither Tensorflow nor PyTorch has been installed. Cannot use DoctrTextlineDetector")

    def clone(self) -> PredictorBase:
        return self.__class__(self.architecture, self.path_weights, self.categories, self.device_input)

    def possible_categories(self) -> List[ObjectTypes]:
        return [LayoutType.word]

    def load_model(self) -> None:
        """Loading model weights"""
        _load_model(self.path_weights, self.doctr_predictor, self.device)


class DoctrTextRecognizer(TextRecognizer):
    """
    A deepdoctection wrapper of DocTr text recognition predictor. The base class is a TextRecognizer that takes
    a batch of sub images (e.g. text lines from a text detector) and returns a list with text spotted in the sub images.
    DocTr supports several text recognition models but provides only a subset of pre-trained models.

    This model that is most suitable for document text recognition is the CRNN implementation with a VGG-16 backbone as
    described in “An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to
    Scene Text Recognition”. It can be used in either Tensorflow or PyTorch.

    For more details please check the official DocTr documentation by Mindee: https://mindee.github.io/doctr/

    **Example:**

                 path_weights_tl = ModelDownloadManager.maybe_download_weights_and_configs("doctr/db_resnet50/pt
                 /db_resnet50-ac60cadc.pt")
                 # Use "doctr/db_resnet50/tf/db_resnet50-adcafc63.zip" for Tensorflow

                 categories = ModelCatalog.get_profile("doctr/db_resnet50/pt/db_resnet50-ac60cadc.pt").categories
                 det = DoctrTextlineDetector("db_resnet50",path_weights_tl,categories,"cpu")
                 layout = ImageLayoutService(det,to_image=True, crop_image=True)

                 path_weights_tr = dd.ModelDownloadManager.maybe_download_weights_and_configs("doctr/crnn_vgg16_bn
                 /pt/crnn_vgg16_bn-9762b0b0.pt")
                 rec = DoctrTextRecognizer("crnn_vgg16_bn", path_weights_tr, "cpu")
                 text = TextExtractionService(rec, extract_from_roi="word")

                 analyzer = DoctectionPipe(pipeline_component_list=[layout,text])

                 path = "/path/to/image_dir"
                 df = analyzer.analyze(path = path)

                 for dp in df:
                     ...

    """

    def __init__(self, architecture: str, path_weights: str, device: Optional[Literal["cpu", "cuda"]] = None) -> None:

        self.name = "doctr_text_recognizer"
        self.architecture = architecture
        self.path_weights = path_weights
        self.device_input = device
        self.device = _set_device_str(device)
        self.doctr_predictor = recognition_predictor(arch=self.architecture, pretrained=True)
        self.load_model()

    def predict(self, images: List[Tuple[str, ImageType]]) -> List[DetectionResult]:
        """
        Prediction on a batch of text lines

        :param images: list of tuples with the annotation_id of the sub image and a numpy array
        :return: A list of DetectionResult
        """
        if images:
            return doctr_predict_text(images, self.doctr_predictor, self.device)
        return []

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        if tf_available():
            return [get_tensorflow_requirement(), get_doctr_requirement(), get_tf_addons_requirements()]
        if pytorch_available():
            return [get_pytorch_requirement(), get_doctr_requirement()]
        raise ModuleNotFoundError("Neither Tensorflow nor PyTorch has been installed. Cannot use DoctrTextRecognizer")

    def clone(self) -> PredictorBase:
        return self.__class__(self.architecture, self.path_weights, self.device_input)

    def load_model(self) -> None:
        """Loading model weights"""
        _load_model(self.path_weights, self.doctr_predictor, self.device)
