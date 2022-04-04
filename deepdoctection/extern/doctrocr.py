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

from typing import List, Tuple

from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import doctr_available, get_doctr_requirement, get_tf_addons_requirements, tf_addons_available
from ..utils.settings import names
from .base import DetectionResult, ObjectDetector, PredictorBase, TextRecognizer

if doctr_available() and tf_addons_available():
    # pylint: disable=import-error
    from doctr.models.detection.predictor import DetectionPredictor  # pylint: disable=W0611
    from doctr.models.detection.zoo import detection_predictor
    from doctr.models.recognition.predictor import RecognitionPredictor  # pylint: disable=W0611
    from doctr.models.recognition.zoo import recognition_predictor

    # pylint: enable=import-error


def doctr_predict_text_lines(np_img: ImageType, predictor: "DetectionPredictor") -> List[DetectionResult]:
    """
    Generating text line DetectionResult based on Doctr DetectionPredictor.

    :param np_img: Image in np.array.
    :param predictor: `doctr.models.detection.predictor.DetectionPredictor`
    :return: A list of text line detection results (without text).
    """
    raw_output = predictor([np_img])
    detection_results = [
        DetectionResult(box=box[:4].tolist(), class_id=1, score=box[4], absolute_coords=False, class_name=names.C.WORD)
        for box in raw_output[0]
    ]
    return detection_results


def doctr_predict_text(inputs: List[Tuple[str, ImageType]], predictor: "RecognitionPredictor") -> List[DetectionResult]:
    """
    Calls Doctr text recognition model on a batch of numpy arrays (text lines predicted from a text line detector) and
    returns the recognized text as DetectionResult

    :param inputs: list of tuples containing the annotation_id of the input image and the numpy array of the cropped
                   text line

    :param predictor: `doctr.models.detection.predictor.RecognitionPredictor`
    :return: A list of DetectionResult containing recognized text.
    """

    uuids, images = list(zip(*inputs))
    raw_output = predictor(list(images))
    detection_results = [
        DetectionResult(score=output[1], text=output[0], uuid=uuid) for uuid, output in zip(uuids, raw_output)
    ]
    return detection_results


class DoctrTextlineDetector(ObjectDetector):
    """
    A deepdoctection wrapper of DocTr text line detector. We model text line detection as ObjectDetector
    and assume to use this detector in a ImageLayoutService. This model currently uses the default implementation
    DBNet as described in “Real-time Scene Text Detection with Differentiable Binarization”, using a ResNet-50 backbone.
    and can be used in either Tensorflow or PyTorch.

    Regarding the model we refer to the documentation https://mindee.github.io/doctr/models.html#

    **Example:**

         .. code-block:: python

                 path = "/path/to/image_dir"
                 det = DoctrTextlineDetector()
                 layout = ImageLayoutService(det,to_image=True, crop_image=True)
                 rec = DoctrTextRecognizer()
                 text = TextExtractionService(rec,extract_from_roi="LINE")
                 analyzer = DoctectionPipe(pipeline_component_list=[layout,text])
                 df = analyzer.analyze(path = path)

                 for dp in df:
                     ...


    """

    def __init__(self) -> None:
        self.doctr_predictor = detection_predictor(pretrained=True)

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Prediction per image.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        detection_results = doctr_predict_text_lines(np_img, self.doctr_predictor)
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_doctr_requirement(), get_tf_addons_requirements()]

    def clone(self) -> PredictorBase:
        return self.__class__()


class DoctrTextRecognizer(TextRecognizer):
    """
    A deepdoctection wrapper of DocTr text recognition predictor. The base class is a TextRecognizer that takes
    a batch of sub images (e.g. text lines from a text detector) and returns a list with text spotted in the sub images.
    This model currently uses the default implementation CRNN with a VGG-16 backbone as described in
    “An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene
    Text Recognition” and can be used in either Tensorflow or PyTorch.

    Regarding the model we refer to the documentation https://mindee.github.io/doctr/models.html#

    **Example:**

         .. code-block:: python

                 path = "/path/to/image_dir"
                 det = DoctrTextlineDetector()
                 layout = ImageLayoutService(det,to_image=True, crop_image=True)
                 rec = DoctrTextRecognizer()
                 text = TextExtractionService(rec,extract_from_roi="LINE")
                 analyzer = DoctectionPipe(pipeline_component_list=[layout,text])
                 df = analyzer.analyze(path = path)

                 for dp in df:
                     ...

    """

    def __init__(self) -> None:
        self.doctr_predictor = recognition_predictor(pretrained=True)

    def predict(self, images: List[Tuple[str, ImageType]]) -> List[DetectionResult]:
        """
        Prediction on a batch of text lines

        :param images: list of tuples with the annotation_id of the sub image and a numpy array
        :return: A list of DetectionResult
        """
        if images:
            return doctr_predict_text(images, self.doctr_predictor)
        return []

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_doctr_requirement(), get_tf_addons_requirements()]

    def clone(self) -> PredictorBase:
        return self.__class__()
