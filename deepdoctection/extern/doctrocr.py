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

"""
from typing import List, Tuple
from ..utils.settings import names
from ..utils.detection_types import Requirement, ImageType
from ..utils.file_utils import doctr_available, get_doctr_requirement, get_tf_addons_requirements, tf_addons_available
from .base import ObjectDetector, DetectionResult, PredictorBase, TextRecognizer

if doctr_available() and tf_addons_available():
    from doctr.models.detection.zoo import detection_predictor
    from doctr.models.detection.predictor import DetectionPredictor
    from doctr.models.recognition.zoo import recognition_predictor
    from doctr.models.recognition.predictor import RecognitionPredictor


def doctr_predict_text_lines(np_img: ImageType, predictor: "DetectionPredictor") -> List[DetectionResult]:
    raw_output = predictor([np_img])
    detection_results = [DetectionResult(box=box[:4].tolist(),
                                         class_id=1,
                                         score=box[4],
                                         absolute_coords=False,
                                         class_name=names.C.LINE) for box in raw_output[0]]
    return detection_results


def doctr_predict_text(inputs: List[Tuple[str,ImageType]], predictor: "RecognitionPredictor") -> List[DetectionResult]:
    uuids, images = list(zip(*inputs))
    raw_output = predictor(images)
    return raw_output


class DoctrTextlineDetector(ObjectDetector):

    def __init__(self):
        self.doctr_predictor = detection_predictor(pretrained=True)

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        detection_results = doctr_predict_text_lines(np_img,self.doctr_predictor)
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_doctr_requirement(), get_tf_addons_requirements()]

    def clone(self) -> PredictorBase:
        return self.__class__()


class DoctrTextRecognizer(TextRecognizer):

    def __init__(self):
        self.doctr_predictor = recognition_predictor(pretrained=True)

    def predict(self, images: List[Tuple[str,ImageType]]) -> List[DetectionResult]:
        return doctr_predict_text(images, self.doctr_predictor)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_doctr_requirement(),get_tf_addons_requirements()]

    def clone(self) -> PredictorBase:
        return self.__class__()
