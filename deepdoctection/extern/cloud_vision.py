"""
Google Cloud Vision OCR engine for text extraction
"""
import sys
import traceback
from enum import Enum
from typing import List

from ..datapoint.convert import convert_np_array_to_b64_b
from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import cloud_vision_available, get_cloud_vision_requirement
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import LayoutType, ObjectTypes
from .base import DetectionResult, ObjectDetector, PredictorBase

if cloud_vision_available():
    from google.cloud.vision import AnnotateImageResponse, Image, ImageAnnotatorClient


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5


def _google_vision_to_detectresult(response: AnnotateImageResponse) -> List[DetectionResult]:
    all_results = []
    document = response.full_text_annotation
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    word = DetectionResult(
                        box=[
                            word.bounding_box.vertices[0].x,
                            word.bounding_box.vertices[0].y,
                            word.bounding_box.vertices[2].x,
                            word.bounding_box.vertices[2].y,
                        ],
                        score=word.confidence,
                        text=word_text,
                        class_id=1,
                        class_name=LayoutType.word,
                    )
                    all_results.append(word)
    return all_results


def predict_text(np_img: ImageType, client: ImageAnnotatorClient) -> List[DetectionResult]:
    """
    Calls Google Cloud Vision client (`document_text_detection`) and returns plain OCR results.
    Google Cloud account required.

    :param client: Google Cloud Vision client
    :param np_img: Image in np.array.
    :return: A list of google cloud vision extractions wrapped in DetectionResult
    """
    image = Image(content=convert_np_array_to_b64_b(np_img))
    try:
        response = client.document_text_detection(image=image)
    except Exception:
        _, exc_val, exc_tb = sys.exc_info()
        frame_summary = traceback.extract_tb(exc_tb)[0]
        log_dict = {
            "file_name": "NN",
            "error_type": type(exc_val).__name__,
            "error_msg": str(exc_val),
            "orig_module": frame_summary.filename,
            "line": frame_summary.lineno,
        }
        logger.warning(LoggingRecord("Error with Cloud Vision", log_dict))  # type: ignore
        return []
    all_results = _google_vision_to_detectresult(response)
    return all_results


class CloudVisionOCRDetector(ObjectDetector):
    """
    Text object detection using Google Cloud Vision API. Google Cloud Vision package required.

        cloud_vison_predictor = CloudVisionOCRDetector()
        detection_results = cloud_vison_predictor.predict(np_img)

    or
        cloud_vision_predictor = CloudVisionOCRDetector()
        cloud_vision  = TextExtractionService(cloud_vision_predictor)

        pipe = DoctectionPipe(cloud_vision)
        df = pipe.analyse(path = "path/to/document.pdf")
    """

    def __init__(self) -> None:
        self.name = "cloud_vision"
        self.model_id = self.get_model_id()
        self.client = ImageAnnotatorClient()
        self.categories = {"1": LayoutType.word}

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Transfer of a numpy array and call textract client. Return of the detection results.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        return predict_text(np_img, self.client)

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_cloud_vision_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__()

    def possible_categories(self) -> List[ObjectTypes]:
        return [LayoutType.text]
