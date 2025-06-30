# -*- coding: utf-8 -*-
# File: base.py

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
Base classes for unifying external predictors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union, overload

import numpy as np
from lazy_imports import try_import

from ..utils.identifier import get_uuid_from_str
from ..utils.logger import logger
from ..utils.settings import (
    DefaultType,
    ObjectTypes,
    TypeOrStr,
    get_type,
    token_class_tag_to_token_class_with_tag,
    token_class_with_tag_to_token_class_and_tag,
)
from ..utils.transform import BaseTransform, box_to_point4, point4_to_box
from ..utils.types import JsonDict, PixelValues, Requirement

if TYPE_CHECKING:
    with try_import() as import_guard:
        import torch


@dataclass
class ModelCategories:
    """
    Categories for models (except models for NER tasks) are managed in this class.
    Different to `DatasetCategories`, these members are immutable.

    Example:

        ```python
        categories = ModelCategories(init_categories={1: "text", 2: "title"})
        cats = categories.get_categories(as_dict=True)  # {1: LayoutType.text, 2: LayoutType.title}
        categories.filter_categories = [LayoutType.text]  # filter out text
        cats = categories.get_categories(as_dict=True)  # {2: LayoutType.title}
        ```

    """

    init_categories: Optional[Mapping[int, TypeOrStr]] = field(repr=False)
    _init_categories: MappingProxyType[int, ObjectTypes] = field(init=False, repr=False)
    _filter_categories: Sequence[ObjectTypes] = field(init=False, repr=False, default_factory=tuple)
    categories: MappingProxyType[int, ObjectTypes] = field(init=False)

    def __post_init__(self) -> None:
        """post init method"""
        if self.init_categories:
            self._init_categories = MappingProxyType({key: get_type(val) for key, val in self.init_categories.items()})
        else:
            self._init_categories = MappingProxyType({})
        self.categories = self._init_categories

    @overload
    def get_categories(self, as_dict: Literal[False]) -> tuple[ObjectTypes, ...]:
        ...

    @overload
    def get_categories(
        self, as_dict: Literal[True] = ..., name_as_key: Literal[False] = False
    ) -> MappingProxyType[int, ObjectTypes]:
        ...

    @overload
    def get_categories(self, as_dict: Literal[True], name_as_key: Literal[True]) -> MappingProxyType[ObjectTypes, int]:
        ...

    def get_categories(
        self, as_dict: bool = True, name_as_key: bool = False
    ) -> Union[MappingProxyType[int, ObjectTypes], MappingProxyType[ObjectTypes, int], tuple[ObjectTypes, ...]]:
        """
        Get the categories

        Args:
            as_dict: return as dict
            name_as_key: if `as_dict=True` and `name_as_key=True` will swap key and value

        Returns:
            categories dict
        """
        if as_dict:
            if name_as_key:
                return MappingProxyType(
                    {value: key for key, value in self._init_categories.items() if value not in self.filter_categories}
                )
            return MappingProxyType(
                {key: value for key, value in self._init_categories.items() if value not in self.filter_categories}
            )
        return tuple(val for val in self._init_categories.values() if val not in self.filter_categories)

    @property
    def filter_categories(self) -> Sequence[ObjectTypes]:
        """`filter_categories`"""
        return self._filter_categories

    @filter_categories.setter
    def filter_categories(self, categories: Sequence[ObjectTypes]) -> None:
        """`categories` setter"""
        self._filter_categories = categories
        self.categories = self.get_categories()

    def shift_category_ids(self, shift_by: int) -> MappingProxyType[int, ObjectTypes]:
        """
        Shift `category_id`s

        Example:
            ```python
            categories = ModelCategories(init_categories={"1": "text", "2": "title"})
            cats = categories.shift_category_ids(1) # {"2": LayoutType.text, "3": LayoutType.title}
            ```

         Args:
            shift_by: The value to shift the category id to the left or to the right

        Returns:
            shifted categories
        """
        return MappingProxyType({k + shift_by: v for k, v in self.get_categories().items()})


@dataclass
class NerModelCategories(ModelCategories):
    """
    Categories for models for NER tasks. It can handle the merging of token classes and bio tags to build a new set
    of categories.

    Example:
        ```python
        categories = NerModelCategories(categories_semantics=["question", "answer"], categories_bio=["B", "I"])
        cats = categories.get_categories(as_dict=True)  # {"1": TokenClassWithTag.b_question,
                                                           "2": TokenClassWithTag.i_question,
                                                           "3": TokenClassWithTag.b_answer,
                                                           "4": TokenClassWithTag.i_answer}
        ```

    You can also leave the categories unchanged:

    Example:
        ```python
        categories = NerModelCategories(init_categories={"1": "question", "2": "answer"})
        cats = categories.get_categories(as_dict=True)  # {"1": TokenClasses.question,
                                                           "2": TokenClasses.answer}
        ```
    """

    categories_semantics: Optional[Sequence[TypeOrStr]] = field(default=None)
    categories_bio: Optional[Sequence[TypeOrStr]] = field(default=None)
    _categories_semantics: tuple[ObjectTypes, ...] = field(init=False, repr=False)
    _categories_bio: tuple[ObjectTypes, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.init_categories:
            if not self.categories_semantics:
                raise ValueError("If categories is None then categories_semantics cannot be None")
            if not self.categories_bio:
                raise ValueError("If categories is None then categories_bio cannot be None")
        else:
            self._init_categories = MappingProxyType({key: get_type(val) for key, val in self.init_categories.items()})

        if self.categories_bio:
            self._categories_bio = tuple((get_type(cat) for cat in self.categories_bio))
        if self.categories_semantics:
            self._categories_semantics = tuple((get_type(cat) for cat in self.categories_semantics))
        if self.categories_bio and self.categories_semantics and self.init_categories:
            logger.info("Will disregard categories_bio and categories_semantics")

        if self.categories_bio and self.categories_semantics:
            self._init_categories = self.merge_bio_semantics_categories(
                self._categories_semantics, self._categories_bio
            )
        self.categories = self._init_categories

    @staticmethod
    def merge_bio_semantics_categories(
        categories_semantics: tuple[ObjectTypes, ...], categories_bio: tuple[ObjectTypes, ...]
    ) -> MappingProxyType[int, ObjectTypes]:
        """
        Merge bio and semantics categories

        Example:

            ```python
            categories = NerModelCategories(categories_semantics=["question", "answer"], categories_bio=["B", "I"])
            cats = categories.get_categories(as_dict=True)  # {"1": TokenClassWithTag.b_question,
                                                               "2": TokenClassWithTag.i_question,
                                                               "3": TokenClassWithTag.b_answer,
                                                               "4": TokenClassWithTag.i_answer}
            ```

        Args:
            categories_semantics: semantic categories (without tags)
            categories_bio: bio tags

        Returns:
            A mapping of categories with tags
        """
        categories_list = sorted(
            {
                token_class_tag_to_token_class_with_tag(token, tag)
                for token in categories_semantics
                for tag in categories_bio
            }
        )
        return MappingProxyType(dict(enumerate(categories_list, 1)))

    @staticmethod
    def disentangle_token_class_and_tag(category_name: ObjectTypes) -> Optional[tuple[ObjectTypes, ObjectTypes]]:
        """
        Disentangle token class and tag. It will return separate ObjectTypes for token class and tag.

        Example:

            ```python
             NerModelCategories.disentangle_token_class_and_tag(TokenClassWithTag.b_question)
             # (TokenClasses.question, TokenTags.begin)
            ```

        Args:
            category_name: A category name with token class and tag

        Returns:
            Tuple of disentangled token class and tag
        """
        return token_class_with_tag_to_token_class_and_tag(category_name)


class PredictorBase(ABC):
    """
    Abstract base class for all types of predictors (e.g. object detectors language models, ...)
    """

    name: str
    model_id: str

    def __new__(cls, *args, **kwargs):  # type: ignore # pylint: disable=W0613
        requirements = cls.get_requirements()
        name = cls.__name__ if hasattr(cls, "__name__") else cls.__class__.__name__
        if not all(requirement[1] for requirement in requirements):
            raise ImportError(
                "\n".join(
                    [f"{name} has the following dependencies:"]
                    + [requirement[2] for requirement in requirements if not requirement[1]]
                )
            )
        return super().__new__(cls)

    @classmethod
    @abstractmethod
    def get_requirements(cls) -> list[Requirement]:
        """
        Get a list of requirements for running the detector

        Returns:
            A list of requirements, where each requirement is a tuple of the form:
            (requirement_name, is_available, description)
            - `requirement_name`: The name of the requirement.
            - `is_available`: A boolean indicating whether the requirement is available.
            - `description`: A string describing the error code.
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> PredictorBase:
        """
        Clone an instance
        """
        raise NotImplementedError()

    def get_model_id(self) -> str:
        """
        Get the generating model

        Returns:
            A string representing the `model_id`, which is derived from the name of the predictor.

        Raises:
            ValueError: If the name is not set
        """
        if self.name is not None:
            return get_uuid_from_str(self.name)[:8]
        raise ValueError("name must be set before calling get_model_id")

    def clear_model(self) -> None:
        """
        Clear the inner model of the model wrapper if it has one. Needed for model updates during training.
        """
        raise NotImplementedError(
            "Maybe you forgot to implement this method in your pipeline component. This might "
            "be the case when you run evaluation during training and need to update the "
            "trained model in your pipeline component."
        )


@dataclass
class DetectionResult:
    """
    Simple mutable storage for detection results.

    Attributes:
        box: [ulx,uly,lrx,lry]
        class_id: category id
        score: prediction score
        mask: binary mask
        absolute_coords: absolute coordinates
        class_name: category name
        text: text string. Used for OCR predictors
        block: block number. For reading order from some ocr predictors
        line: line number. For reading order from some ocr predictors
        uuid: uuid. For assigning detection result (e.g. text to image annotations)
    """

    box: Optional[list[float]] = None
    class_id: Optional[int] = None
    score: Optional[float] = None
    mask: Optional[list[float]] = None
    absolute_coords: bool = True
    class_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    text: Optional[Union[str, ObjectTypes]] = None
    block: Optional[str] = None
    line: Optional[str] = None
    uuid: Optional[str] = None
    relationships: Optional[dict[str, Any]] = None
    angle: Optional[float] = None


class ObjectDetector(PredictorBase, ABC):
    """
    Abstract base class for object detection. This can be anything ranging from layout detection to OCR.
    Use this to connect external detectors with deepdoctection predictors on images.

    Example:
        ```python
        MyFancyTensorpackPredictor(TensorpackPredictor,ObjectDetector)
        ```

    and implement the `predict`.
    """

    categories: ModelCategories

    @abstractmethod
    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Abstract method predict

        Args:
            np_img: A numpy array representing the image to be processed by the predictor.
        """
        raise NotImplementedError()

    @property
    def accepts_batch(self) -> bool:
        """
        Whether to accept batches in `predict`
        """
        return False

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """
        `get_category_names`
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> ObjectDetector:
        """
        Clone an instance
        """
        raise NotImplementedError()


class PdfMiner(PredictorBase, ABC):
    """
    Abstract base class for mining information from PDF documents. Reads in a bytes stream from a PDF document page.
    Use this to connect external pdf miners and wrap them into deepdoctection predictors.

    Attributes:
        categories: ModelCategories
        _pdf_bytes: Optional[bytes]: Bytes of the PDF document page to be processed by the predictor.
    """

    categories: ModelCategories
    _pdf_bytes: Optional[bytes] = None

    @abstractmethod
    def predict(self, pdf_bytes: bytes) -> list[DetectionResult]:
        """
        Abstract method predict

        Args:
            pdf_bytes: A bytes stream representing the PDF document page to be processed by the predictor.

        Returns:
            A list of DetectionResult objects containing the results of the prediction.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_width_height(self, pdf_bytes: bytes) -> tuple[float, float]:
        """
        Abstract method get_width_height

        Args:
            pdf_bytes: A bytes stream representing the PDF document page.

        Returns:
            A tuple containing the width and height of the PDF document page.
        """
        raise NotImplementedError()

    def clone(self) -> PdfMiner:
        return self.__class__()

    @property
    def accepts_batch(self) -> bool:
        """
        Whether to accept batches in `predict`

        Returns:
            bool: True if the predictor accepts batches, False otherwise.
        """
        return False

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """
        `get_category_names`
        """
        raise NotImplementedError()


class TextRecognizer(PredictorBase, ABC):
    """
    Abstract base class for text recognition. In contrast to `ObjectDetector` one assumes that `predict` accepts
    batches of `np.arrays`. More precisely, when using `predict` pass a list of tuples with uuids (e.g. `image_id`,
    or `annotation_id`) or `np.array`s.
    """

    @abstractmethod
    def predict(self, images: list[tuple[str, PixelValues]]) -> list[DetectionResult]:
        """
        Abstract method predict

        Args:
            images: A list of tuples, where each tuple contains a unique identifier (e.g., `annotation_id`)
                    and a `np.array` representing the image to be processed by the predictor.
        """
        raise NotImplementedError()

    @property
    def accepts_batch(self) -> bool:
        """
        Whether to accept batches in `predict`
        """
        return True

    @staticmethod
    def get_category_names() -> tuple[ObjectTypes, ...]:
        """return category names"""
        return ()


@dataclass
class TokenClassResult:
    """
    Simple mutable storage for token classification results

    Attributes:
       id: uuid of token (not unique)
       token_id: token id
       token: token
       class_id: category id
       class_name: category name
       semantic_name: semantic name
       bio_tag: bio tag
       score: prediction score
       successor_uuid: uuid of the next token in the sequence
    """

    uuid: str
    token: str
    class_id: int
    class_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    semantic_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    bio_tag: ObjectTypes = DefaultType.DEFAULT_TYPE
    score: Optional[float] = None
    token_id: Optional[int] = None
    successor_uuid: Optional[str] = None


@dataclass
class SequenceClassResult:
    """
    Storage for sequence classification results

    Attributes:
        class_id: category_id
        class_name: category_name
        score: prediction score
        class_name_orig: original class name
    """

    class_id: int
    class_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    score: Optional[float] = None
    class_name_orig: Optional[str] = None


class LMTokenClassifier(PredictorBase, ABC):
    """
    Abstract base class for token classifiers. If you want to connect external token classifiers with deepdoctection
    predictors wrap them into a class derived from this class.
    """

    @abstractmethod
    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
        """
        Abstract method predict

        Args:
            encodings: A dictionary of encodings, where each key is a string representing the encoding type
                       (e.g., "input_ids", "attention_mask") and the value is a list of lists of strings or a
                       torch.Tensor representing the encoded input data.
        """
        raise NotImplementedError()

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Some models require that their inputs must be pre-processed in a specific way. Responsible for converting
        an `Image` datapoint into the input format in inference mode is a mapper function which is called
        in a pipeline component. The mapper function's name, which has to be used must be specified in the returned
        value of `image_to_features_mapping`.
        This mapper function is often implemented for various models and can therefore have various parameters.
        Some parameters can be inferred from the config file of the model parametrization. Some other might not be
        in the parametrization and therefore have to be specified here.

        This method therefore returns a dictionary that contains as keys some arguments of the function
        `image_to_features_mapping` and as values the values necessary for providing the model with the required input.
        """
        return {}

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """
        Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name either for
        training or inference purposes:

        `image_to_raw_features_mapping` is used for training and transforms an image into raw features that can be
        further processed through augmentation or batching. It should not be used when running inference, i.e. when
        running the model in a pipeline component.
        """
        return ""

    @staticmethod
    def image_to_features_mapping() -> str:
        """
        Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_features_mapping` is a mapping function that converts a single image into ready features that can
        be directly fed into the model. We use this function to determine the input format of the model in a pipeline
        component. Note that this function will also require specific parameters, which can be specified in
        `default_kwargs_for_image_to_features_mapping`.

        """
        return ""


class LMSequenceClassifier(PredictorBase, ABC):
    """
    Abstract base class for sequence classification. If you want to connect external sequence classifiers with
    deepdoctection predictors, wrap them into a class derived from this class.
    """

    @abstractmethod
    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> SequenceClassResult:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @staticmethod
    def default_kwargs_for_image_to_features_mapping() -> JsonDict:
        """
        Some models require that their inputs must be pre-processed in a specific way. Responsible for converting
        an `Image` datapoint into the input format in inference mode is a mapper function which is called
        in a pipeline component. The mapper function's name, which has to be used must be specified in the returned
        value of `image_to_features_mapping`.
        This mapper function is often implemented for various models and can therefore have various parameters.
        Some parameters can be inferred from the config file of the model parametrization. Some other might not be
        in the parametrization and therefore have to be specified here.

        This method therefore returns a dictionary that contains as keys some arguments of the function
        `image_to_features_mapping` and as values the values necessary for providing the model with the required input.
        """
        return {}

    @staticmethod
    def image_to_raw_features_mapping() -> str:
        """
        Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_raw_features_mapping` is used for training and transforms an image into raw features that can be
        further processed through augmentation or batching. It should not be used when running inference, i.e. when
        running the model in a pipeline component.
        """
        return ""

    @staticmethod
    def image_to_features_mapping() -> str:
        """
        Converting image into model features must often be divided into several steps. This is because the process
        method during training and serving might differ: For training there might be additional augmentation steps
        required or one might add some data batching. For this reason we have added two methods
        `image_to_raw_features_mapping`, `image_to_features_mapping` that return a mapping function name for either for
        training or inference purposes:

        `image_to_features_mapping` is a mapping function that converts a single image into ready features that can
        be directly fed into the model. We use this function to determine the input format of the model in a pipeline
        component. Note that this function will also require specific parameters, which can be specified in
        `default_kwargs_for_image_to_features_mapping`.

        """
        return ""


class LanguageDetector(PredictorBase, ABC):
    """
    Abstract base class for language detectors.
    """

    @abstractmethod
    def predict(self, text_string: str) -> DetectionResult:
        """
        Abstract method predict

        Args:
            text_string: A string representing the text to be processed by the predictor.

        Returns:
            A DetectionResult object containing the detected language information (ISO-639 code).
        """
        raise NotImplementedError()


class ImageTransformer(PredictorBase, ABC):
    """
    Abstract base class for transforming an image.
    """

    @abstractmethod
    def transform_image(self, np_img: PixelValues, specification: DetectionResult) -> PixelValues:
        """
        Abstract method transform

        Args:
            np_img: A `np.array` representing the image to be transformed.
            specification: A `DetectionResult` instance containing specifications for the transformation.

        Returns:
            A `np.array` representing the transformed image.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, np_img: PixelValues) -> DetectionResult:
        """
        Abstract method predict
        Args:
            np_img: A `np.array` representing the image to be processed by the predictor.

        Rweturns:
            A `DetectionResult` object containing the prediction results regarding the transformation.
        """
        raise NotImplementedError()

    def clone(self) -> ImageTransformer:
        return self.__class__()

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """returns category names"""
        raise NotImplementedError()

    def transform_coords(self, detect_results: Sequence[DetectionResult]) -> Sequence[DetectionResult]:
        """
        Transform coordinates aligned with the transform_image method.

        Args:
            detect_results: List of `DetectionResult`s

        Returns:
            List of DetectionResults. If you pass `uuid` it is possible to track the transformed bounding boxes.
        """

        raise NotImplementedError()

    def inverse_transform_coords(self, detect_results: Sequence[DetectionResult]) -> Sequence[DetectionResult]:
        """
        Inverse transform coordinates aligned with the `transform_image` method. Composing transform_coords with
        inverse_transform_coords should return the original coordinates.

        Args:
            detect_results: List of `DetectionResult`s

        Returns:
            List of `DetectionResult`s. If you pass `uuid` it is possible to track the transformed bounding boxes.
        """

        raise NotImplementedError()


class DeterministicImageTransformer(ImageTransformer):
    """
    A wrapper for BaseTransform classes that implements the ImageTransformer interface.

    This class provides a bridge between the BaseTransform system (which handles image and coordinate
    transformations like rotation, padding, etc.) and the predictors framework by implementing the
    ImageTransformer interface. It allows BaseTransform objects to be used within pipelines that
    expect ImageTransformer components.

    The transformer performs deterministic transformations on images and their associated coordinates,
    enabling operations like padding, rotation, and other geometric transformations while maintaining
    the relationship between image content and annotation coordinates.
    """

    def __init__(self, base_transform: BaseTransform) -> None:
        """
        Initialize the DeterministicImageTransformer with a BaseTransform instance.

        Args:
            base_transform: A BaseTransform instance that defines the actual transformation operations
        """
        self.base_transform = base_transform
        self.name = base_transform.__class__.__name__
        self.model_id = self.get_model_id()

    def transform_image(self, np_img: PixelValues, specification: DetectionResult) -> PixelValues:
        return self.base_transform.apply_image(np_img)

    def transform_coords(self, detect_results: Sequence[DetectionResult]) -> Sequence[DetectionResult]:
        boxes = np.array([detect_result.box for detect_result in detect_results])
        # boxes = box_to_point4(boxes)
        boxes = self.base_transform.apply_coords(boxes)
        # boxes = point4_to_box(boxes)
        detection_results = []
        for idx, detect_result in enumerate(detect_results):
            detection_results.append(
                DetectionResult(
                    box=boxes[idx, :].tolist(),
                    class_name=detect_result.class_name,
                    class_id=detect_result.class_id,
                    score=detect_result.score,
                    absolute_coords=detect_result.absolute_coords,
                    uuid=detect_result.uuid,
                )
            )
        return detection_results

    def inverse_transform_coords(self, detect_results: Sequence[DetectionResult]) -> Sequence[DetectionResult]:
        boxes = np.array([detect_result.box for detect_result in detect_results])
        boxes = box_to_point4(boxes)
        boxes = self.base_transform.inverse_apply_coords(boxes)
        boxes = point4_to_box(boxes)
        detection_results = []
        for idx, detect_result in enumerate(detect_results):
            detection_results.append(
                DetectionResult(
                    box=boxes[idx, :].tolist(),
                    class_id=detect_result.class_id,
                    score=detect_result.score,
                    absolute_coords=detect_result.absolute_coords,
                    uuid=detect_result.uuid,
                )
            )
        return detection_results

    def clone(self) -> DeterministicImageTransformer:
        return self.__class__(self.base_transform)

    def predict(self, np_img: PixelValues) -> DetectionResult:
        detect_result = DetectionResult()
        for init_arg in self.base_transform.get_init_args():
            setattr(detect_result, init_arg, getattr(self.base_transform, init_arg))
        return detect_result

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.base_transform.get_category_names()

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return []
