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
Abstract classes for unifying external base- and Doctection predictors
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional, Sequence, Union, overload

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
from ..utils.types import JsonDict, PixelValues, Requirement

if TYPE_CHECKING:
    with try_import() as import_guard:
        import torch


@dataclass
class ModelCategories:
    """
    Categories for models (except models for NER tasks) are managed in this class. Different to DatasetCategories,
    these members are immutable.

    **Example**:

        categories = ModelCategories(init_categories={1: "text", 2: "title"})
        cats = categories.get_categories(as_dict=True)  # {1: LayoutType.text, 2: LayoutType.title}
        categories.filter_categories = [LayoutType.text]  # filter out text
        cats = categories.get_categories(as_dict=True)  # {2: LayoutType.title}
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

        :param as_dict: return as dict
        :param name_as_key: if as_dict=`True` and name_as_key=`True` will swap key and value
        :return: categories dict
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
        """filter_categories"""
        return self._filter_categories

    @filter_categories.setter
    def filter_categories(self, categories: Sequence[ObjectTypes]) -> None:
        """categories setter"""
        self._filter_categories = categories
        self.categories = self.get_categories()

    def shift_category_ids(self, shift_by: int) -> MappingProxyType[int, ObjectTypes]:
        """
        Shift category ids

         **Example**:

            categories = ModelCategories(init_categories={"1": "text", "2": "title"})
            cats = categories.shift_category_ids(1) # {"2": LayoutType.text, "3": LayoutType.title}

        :param shift_by: The value to shift the category id to the left or to the right
        :return: shifted categories
        """
        return MappingProxyType({k + shift_by: v for k, v in self.get_categories().items()})


@dataclass
class NerModelCategories(ModelCategories):
    """
    Categories for models for NER tasks. It can handle the merging of token classes and bio tags to build a new set
    of categories.

    **Example**:

        categories = NerModelCategories(categories_semantics=["question", "answer"], categories_bio=["B", "I"])
        cats = categories.get_categories(as_dict=True)  # {"1": TokenClassWithTag.b_question,
                                                           "2": TokenClassWithTag.i_question,
                                                           "3": TokenClassWithTag.b_answer,
                                                           "4": TokenClassWithTag.i_answer}

    You can also leave the categories unchanged:

    **Example**:

        categories = NerModelCategories(init_categories={"1": "question", "2": "answer"})
        cats = categories.get_categories(as_dict=True)  # {"1": TokenClasses.question,
                                                           "2": TokenClasses.answer}
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

        **Example**:

            categories = NerModelCategories(categories_semantics=["question", "answer"], categories_bio=["B", "I"])
            cats = categories.get_categories(as_dict=True)  # {"1": TokenClassWithTag.b_question,
                                                               "2": TokenClassWithTag.i_question,
                                                               "3": TokenClassWithTag.b_answer,
                                                               "4": TokenClassWithTag.i_answer}
        :param categories_semantics: semantic categories (without tags)
        :param categories_bio: bio tags
        :return: A mapping of categories with tags
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

        **Example**:

             NerModelCategories.disentangle_token_class_and_tag(TokenClassWithTag.b_question)
             # (TokenClasses.question, TokenTags.begin)

        :param category_name: A category name with token class and tag
        :return: Tuple of disentangled token class and tag
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

    `box`: [ulx,uly,lrx,lry]

    `class_id`: category id

    `score`: prediction score

    `mask`: binary mask

    `absolute_coords` : absolute coordinates

    `class_name`: category name

    `text`: text string. Used for OCR predictors

    `block`: block number. For reading order from some ocr predictors

    `line`: line number. For reading order from some ocr predictors

    `uuid`: uuid. For assigning detection result (e.g. text to image annotations)


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

    **Example:**

            MyFancyTensorpackPredictor(TensorpackPredictor,ObjectDetector)

    and implement the `predict`.
    """

    categories: ModelCategories

    @abstractmethod
    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in `predict`
        """
        return False

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """
        Abstract method get_category_names
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
    Use this to connect external pdf miners and wrap them into Deep-Doctection predictors.
    """

    categories: ModelCategories
    _pdf_bytes: Optional[bytes] = None

    @abstractmethod
    def predict(self, pdf_bytes: bytes) -> list[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @abstractmethod
    def get_width_height(self, pdf_bytes: bytes) -> tuple[float, float]:
        """
        Abstract method get_width_height
        """
        raise NotImplementedError()

    def clone(self) -> PdfMiner:
        return self.__class__()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in `predict`
        """
        return False

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """
        Abstract method get_category_names
        """
        raise NotImplementedError()


class TextRecognizer(PredictorBase, ABC):
    """
    Abstract base class for text recognition. In contrast to ObjectDetector one assumes that `predict` accepts
    batches of numpy arrays. More precisely, when using `predict` pass a list of tuples with uuids (e.g. image_id,
    or annotation_id) or numpy arrays.
    """

    @abstractmethod
    def predict(self, images: list[tuple[str, PixelValues]]) -> list[DetectionResult]:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    @property
    def accepts_batch(self) -> bool:
        """
        whether to accept batches in `predict`
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

     `id`: uuid of token (not unique)

     `token_id`: token id

     `token`: token

     `class_id`: category id

     `class_name`: category name

     `semantic_name`: semantic name

     `bio_tag`: bio tag

     `score`: prediction score
    """

    uuid: str
    token: str
    class_id: int
    class_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    semantic_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    bio_tag: ObjectTypes = DefaultType.DEFAULT_TYPE
    score: Optional[float] = None
    token_id: Optional[int] = None


@dataclass
class SequenceClassResult:
    """
    Storage for sequence classification results

    `class_id`: category id
    `class_name`: category name
    `score`: prediction score
    `class_name_orig`: original class name
    """

    class_id: int
    class_name: ObjectTypes = DefaultType.DEFAULT_TYPE
    score: Optional[float] = None
    class_name_orig: Optional[str] = None


class LMTokenClassifier(PredictorBase, ABC):
    """
    Abstract base class for token classifiers. If you want to connect external token classifiers with Deepdoctection
    predictors wrap them into a class derived from this class. Note, that this class is still DL library agnostic.
    """

    @abstractmethod
    def predict(self, **encodings: Union[list[list[str]], torch.Tensor]) -> list[TokenClassResult]:
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
        """Converting image into model features must often be divided into several steps. This is because the process
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
        """Converting image into model features must often be divided into several steps. This is because the process
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
        """Converting image into model features must often be divided into several steps. This is because the process
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
        """Converting image into model features must often be divided into several steps. This is because the process
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
    Abstract base class for language detectors. The `predict` accepts a string of arbitrary length and returns an
    ISO-639 code for the detected language.
    """

    @abstractmethod
    def predict(self, text_string: str) -> DetectionResult:
        """
        Abstract method predict
        """
        raise NotImplementedError()


class ImageTransformer(PredictorBase, ABC):
    """
    Abstract base class for transforming an image. The `transform` accepts and returns a numpy array
    """

    @abstractmethod
    def transform(self, np_img: PixelValues, specification: DetectionResult) -> PixelValues:
        """
        Abstract method transform
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, np_img: PixelValues) -> DetectionResult:
        """
        Abstract method predict
        """
        raise NotImplementedError()

    def clone(self) -> ImageTransformer:
        return self.__class__()

    @abstractmethod
    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        """returns category names"""
        raise NotImplementedError()
