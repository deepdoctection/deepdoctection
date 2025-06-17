# -*- coding: utf-8 -*-
# File: annotation.py

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
Dataclass for `Annotation`s and their sub-classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Union, no_type_check

from ..utils.error import AnnotationError, UUIDError
from ..utils.identifier import get_uuid, is_uuid_like
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..utils.types import AnnotationDict
from .box import BoundingBox
from .convert import as_dict


@no_type_check
def ann_from_dict(cls, **kwargs: AnnotationDict):
    """
    A factory function to create subclasses of annotations from a given dict
    """
    _init_kwargs = {
        "external_id": kwargs.get("external_id"),
        "category_name": kwargs.get("category_name"),
        "category_id": kwargs.get("category_id", DEFAULT_CATEGORY_ID),
        "score": kwargs.get("score"),
        "service_id": kwargs.get("service_id"),
        "model_id": kwargs.get("model_id"),
        "session_id": kwargs.get("session_id"),
    }
    _init_kwargs["category_id"] = (
        int(_init_kwargs["category_id"]) if (_init_kwargs)["category_id"] not in ("None", "") else DEFAULT_CATEGORY_ID
    )
    ann = cls(**_init_kwargs)
    ann.active = kwargs.get("active")
    ann._annotation_id = kwargs.get("_annotation_id")  # pylint: disable=W0212
    if isinstance(kwargs.get("sub_categories"), dict):
        for key, value in kwargs["sub_categories"].items():
            if "value" in value:
                ann.dump_sub_category(key, ContainerAnnotation.from_dict(**value))
            else:
                ann.dump_sub_category(key, CategoryAnnotation.from_dict(**value))
    if isinstance(kwargs.get("relationships"), dict):
        for key, values in kwargs["relationships"].items():
            for value in values:
                ann.dump_relationship(key, value)
    return ann


@dataclass(frozen=True)
class AnnotationMap:
    """AnnotationMap to store all sub categories, relationship keys and summary keys of an annotation"""

    image_annotation_id: str
    sub_category_key: Optional[ObjectTypes] = None
    relationship_key: Optional[ObjectTypes] = None
    summary_key: Optional[ObjectTypes] = None


@dataclass
class Annotation(ABC):
    """
    Abstract base class for all types of annotations. This abstract base class only implements general methods for
    correctly assigning annotation_ids. It also has an active flag which is set to True. Only active annotations will be
    returned when querying annotations from image containers.

    Annotation id should never be assigned by yourself. One possibility of assigning an id depending on an external
    value is to set an external id, which in turn creates an annotation id as a function of this.
    An annotation id can only be explicitly set, provided the value is a md5 hash.

    Note:
         Ids will be generated automatically if the annotation object is dumped in a parent container,
         either an image or an annotation (e.g. sub-category). If no id is supplied, the `annotation_id` is created
         depending on the defining attributes (key and value pairs) as specified in the return value of
         `get_defining_attributes`.

    Attributes:
        active: Always set to `True`. You can change the value using `deactivate` .

        external_id: A string or integer value for generating an annotation id. Note, that the resulting annotation
                     id will not depend on the defining attributes.

        _annotation_id: Unique id for annotations. Will always be given as string representation of a md5-hash.
        service_id: Service that generated the annotation. This will be the name of a pipeline component
        model_id: Model that generated the annotation. This will be the name of a model in a component
        session_id: Session id for the annotation. This will be the id of the session in which the annotation was
                    created.
    """

    active: bool = field(default=True, init=False, repr=True)
    external_id: Optional[Union[str, int]] = field(default=None, init=True, repr=False)
    _annotation_id: Optional[str] = field(default=None, init=False, repr=True)
    service_id: Optional[str] = field(default=None)
    model_id: Optional[str] = field(default=None)
    session_id: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        """
        Will check, if the external id provided is an uuid. If not will use the external id as seed for defining an
        `uuid`.
        """

        if self.external_id is not None:
            external_id = str(self.external_id)
            if is_uuid_like(external_id):
                self.annotation_id = external_id
            else:
                self.annotation_id = get_uuid(external_id)
        self._assert_attributes_have_str()

    @property
    def annotation_id(self) -> str:
        """
        `annotation_id`
        """
        if self._annotation_id:
            return self._annotation_id
        raise AnnotationError("Dump annotation first or pass external_id to create an annotation id")

    @annotation_id.setter
    def annotation_id(self, input_id: str) -> None:
        """
        `annotation_id` setter
        """
        if self._annotation_id is not None:
            raise AnnotationError("Annotation_id already defined and cannot be reset")
        if is_uuid_like(input_id):
            self._annotation_id = input_id
        elif isinstance(input_id, property):
            pass
        else:
            raise AnnotationError("Annotation_id must be uuid3 string")

    @abstractmethod
    def get_defining_attributes(self) -> list[str]:
        """
        Defining attributes of an annotation instance are attributes, of which you think that they uniquely
        describe the annotation object. If you do not provide an external id, only the defining attributes will be used
        for generating the annotation id.

        Returns:
            A list of attributes.
        """
        raise NotImplementedError()

    def _assert_attributes_have_str(self, state_id: bool = False) -> None:
        defining_attributes = self.get_state_attributes() if state_id else self.get_defining_attributes()
        for attr in defining_attributes:
            if not hasattr(eval("self." + attr), "__str__"):  # pylint: disable=W0123
                raise AnnotationError(f"Attribute {attr} must have __str__ method")

    @staticmethod
    def set_annotation_id(annotation: CategoryAnnotation, *container_id_context: Optional[str]) -> str:
        """
        Defines the `annotation_id` by attributes of the annotation class as well as by external parameters given by a
        tuple or list of container id contexts.

        Args:
            annotation: The annotation instance for which the id should be generated.
            container_id_context: A tuple/list of strings on which you want the resulting annotation id to depend on.

        Returns:
            A uuid that uniquely characterizes the annotation.
        """
        for container_id in container_id_context:
            if container_id is not None:
                assert is_uuid_like(container_id), f"{container_id} is not a uuid"
        attributes = annotation.get_defining_attributes()
        attributes_values = [str(getattr(annotation, attribute)) for attribute in attributes]
        return get_uuid(*attributes_values, *container_id_context)  # type: ignore

    def as_dict(self) -> AnnotationDict:
        """
        Returning the full dataclass as dict. Uses the custom `convert.as_dict` to disregard attributes defined by
        `remove_keys`.

        Returns:
            A custom dict.
        """

        img_dict = as_dict(self, dict_factory=dict)

        return img_dict

    def deactivate(self) -> None:
        """
        Sets `active` to False. When calling `Image.get_annotations()` it will be filtered.
        """
        self.active = False

    @classmethod
    @abstractmethod
    def from_dict(cls, **kwargs: AnnotationDict) -> Annotation:
        """
        Method to initialize a derived class from dict.

        Args:
            kwargs: dict with `Annotation` attributes

        Returns:
            Annotation instance
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_state_attributes() -> list[str]:
        """
        Similar to `get_defining_attributes` but for `state_id`

        Returns:
            A list of attributes.
        """
        raise NotImplementedError()

    @property
    def state_id(self) -> str:
        """
        Different to `annotation_id` this id does depend on every defined state attributes and might therefore change
        over time.

        Returns:
            Annotation state instance
        """
        container_ids = []
        attributes = self.get_state_attributes()
        for attribute in attributes:
            attr = getattr(self, attribute)
            if isinstance(attr, dict):
                for key, value in attr.items():
                    if isinstance(value, Annotation):
                        container_ids.extend([key, value.state_id])
                    elif isinstance(value, list):
                        container_ids.extend([str(element) for element in value])
                    else:
                        raise TypeError(f"Cannot determine __str__ or annotation_id for element in {attribute}")
            elif isinstance(attr, list):
                for element in attr:
                    if isinstance(element, Annotation):
                        container_ids.append(element.state_id)
                    if isinstance(element, str):
                        container_ids.append(element)
                    else:
                        container_ids.append(str(element))
            elif hasattr(attr, "state_id"):
                container_ids.append(attr.state_id)
            else:
                container_ids.append(str(attr))
        return get_uuid(self.annotation_id, *container_ids)


DEFAULT_CATEGORY_ID = -1


@dataclass
class CategoryAnnotation(Annotation):
    """
    A general class for storing categories (labels/classes) as well as sub categories (sub-labels/subclasses),
    relationships and prediction scores.

    Sub-categories and relationships are stored in a dict, which are populated via the `dum_sub_category` or
    `dump_relationship`. If a key is already available as a sub-category, it must be explicitly removed using the
    `remove_sub_category` before replacing the sub-category.

    Note:
        Sub categories are only accepted as category annotations. Relationships, on the other hand, are only
        managed by passing the `annotation_id`.

    Attributes:
        category_name: String will be used for selecting specific annotations. Use upper case strings.
        category_id: When setting a value will accept strings and ints. Will be stored as string.
        score: Score of a prediction.
        sub_categories: Do not access the dict directly. Rather use the access `get_sub_category` resp.
                        `dump_sub_category`.
        relationships: Do not access the dict directly either. Use `get_relationship` or
                       `dump_relationship` instead.
    """

    category_name: TypeOrStr = field(default=DefaultType.DEFAULT_TYPE)
    _category_name: ObjectTypes = field(default=DefaultType.DEFAULT_TYPE, init=False)
    category_id: int = field(default=DEFAULT_CATEGORY_ID)
    score: Optional[float] = field(default=None)
    sub_categories: dict[ObjectTypes, CategoryAnnotation] = field(default_factory=dict, init=False, repr=True)
    relationships: dict[ObjectTypes, list[str]] = field(default_factory=dict, init=False, repr=True)

    @property  # type: ignore
    def category_name(self) -> ObjectTypes:
        """`category_name`"""
        return self._category_name

    @category_name.setter
    def category_name(self, category_name: TypeOrStr) -> None:
        """category name setter"""
        if not isinstance(category_name, property):
            self._category_name = get_type(category_name)

    def __post_init__(self) -> None:
        self._assert_attributes_have_str(state_id=True)
        super().__post_init__()

    def dump_sub_category(
        self, sub_category_name: TypeOrStr, annotation: CategoryAnnotation, *container_id_context: Optional[str]
    ) -> None:
        """
        Storage of sub categories. As sub categories usually only depend on very few attributes and the parent
        category cannot yet be stored in a comprehensive container, it is possible to include a context of the
        annotation id in order to ensure that the sub-category annotation id is unambiguously created.

        Args:
            sub_category_name: key for defining the sub category.
            annotation: Annotation instance to dump
            container_id_context: Tuple/list of context ids.
        """

        if sub_category_name in self.sub_categories:
            raise AnnotationError(
                f"sub category {sub_category_name} already defined: "
                f"annotation_id: {self.annotation_id}, "
                f"category_name: {self.category_name}, "
                f"category_id: {self.category_id}"
            )

        if self._annotation_id is not None:
            if annotation._annotation_id is None:  # pylint: disable=W0212
                annotation.annotation_id = self.set_annotation_id(annotation, self.annotation_id, *container_id_context)
        else:
            tmp_annotation_id = self.set_annotation_id(self)
            if annotation._annotation_id is None:  # pylint: disable=W0212
                annotation.annotation_id = annotation.set_annotation_id(
                    annotation, tmp_annotation_id, *container_id_context
                )
        self.sub_categories[get_type(sub_category_name)] = annotation

    def get_sub_category(self, sub_category_name: ObjectTypes) -> CategoryAnnotation:
        """
        Return a sub category by its key.

        Args:
            sub_category_name: The key of the sub-category.

        Returns:
            sub category as `CategoryAnnotation`
        """
        return self.sub_categories[sub_category_name]

    def remove_sub_category(self, key: ObjectTypes) -> None:
        """
        Removes a sub category with a given key. Necessary to call, when you want to replace an already dumped sub
        category.

        Args:
            key: A key to a sub category.
        """

        if key in self.sub_categories:
            self.sub_categories.pop(key)

    def dump_relationship(self, key: TypeOrStr, annotation_id: str) -> None:
        """
        Dumps an `annotation_id` to a given key, in order to store relations between annotations.

        Note:
            The referenced annotation must be stored elsewhere.

        Args:
            key: The key, where to place the `annotation_id`.
            annotation_id: The `annotation_id` to dump
        """
        if not is_uuid_like(annotation_id):
            raise UUIDError("Annotation_id must be uuid")

        key_type = get_type(key)
        if key not in self.relationships:
            self.relationships[key_type] = []
        if annotation_id not in self.relationships[key_type]:
            self.relationships[key_type].append(annotation_id)

    def get_relationship(self, key: ObjectTypes) -> list[str]:
        """
        Returns a list of annotation ids stored with a given relationship key.

        Args:
            key: The key for the required relationship.

        Returns:
            A (possibly) empty list of `annotation_id`s.
        """
        if key in self.relationships:
            return self.relationships[key]
        return []

    def remove_relationship(self, key: ObjectTypes, annotation_ids: Optional[Union[list[str], str]] = None) -> None:
        """
        Remove relationship by some given keys and ids. If no annotation ids are provided all relationship according
        to the key will be removed.

        Args:
            key: A relationship key.
            annotation_ids: A single annotation_id or a list. Will remove only the relation with given
                            `annotation_ids`, provided not None is passed as argument.
        """

        if annotation_ids is not None:
            if isinstance(annotation_ids, str):
                annotation_ids = [annotation_ids]
            for ann_id in annotation_ids:
                try:
                    self.relationships[key].remove(ann_id)
                except ValueError:
                    logger.warning(LoggingRecord(f"Relationship {key} cannot be removed because it does not exist"))
        else:
            if key in self.relationships:
                self.relationships[key].clear()

    def get_defining_attributes(self) -> list[str]:
        return ["category_name", "category_id"]

    @staticmethod
    def remove_keys() -> list[str]:
        """
        A list of attributes to suspend from as_dict creation.

        Returns:
            list of attributes.
        """
        return ["_category_name"]

    @classmethod
    def from_dict(cls, **kwargs: AnnotationDict) -> CategoryAnnotation:
        category_ann = ann_from_dict(cls, **kwargs)
        return category_ann

    @staticmethod
    def get_state_attributes() -> list[str]:
        return ["active", "sub_categories", "relationships"]


@dataclass
class ImageAnnotation(CategoryAnnotation):
    """
    A general class for storing annotations related to object detection tasks. In addition to the inherited attributes,
    the class contains a bounding box and an image attribute. The image attribute is optional and is suitable for
    generating an image from the annotation and then saving it there. Compare with the method `image.Image.
    image_ann_to_image`, which naturally populates this attribute.

    Attributes:
        bounding_box: Regarding the coordinate system, if you have to define a prediction, use the system of the
                      image where the object has been detected.

        image: Image, defined by the bounding box and cropped from its parent image. Populate this attribute with
               `Image.image_ann_to_image`.
    """

    bounding_box: Optional[BoundingBox] = field(default=None)
    image: Optional[Image] = field(default=None, init=False, repr=False)  # type: ignore  # pylint: disable=E0602

    def get_defining_attributes(self) -> list[str]:
        return ["category_name", "bounding_box"]

    @classmethod
    def from_dict(cls, **kwargs: AnnotationDict) -> ImageAnnotation:
        image_ann = ann_from_dict(cls, **kwargs)
        if box_kwargs := kwargs.get("bounding_box"):
            image_ann.bounding_box = BoundingBox.from_dict(**box_kwargs)
        return image_ann

    @staticmethod
    def get_state_attributes() -> list[str]:
        return ["active", "sub_categories", "relationships", "image"]

    def get_bounding_box(self, image_id: Optional[str] = None) -> BoundingBox:
        """Get bounding from image embeddings or, if not available or if `image_id` is not provided,
        from `bounding_box`.

        Returns:
            BoundingBox: A bounding box of the annotation.

        Raises:
            ValueError: If no bounding box is available."""
        if self.image and image_id:
            box = self.image.get_embedding(image_id)
        else:
            box = self.bounding_box
        if box:
            return box
        raise AnnotationError(f"bounding_box has not been initialized for {self.annotation_id}")

    def get_summary(self, key: ObjectTypes) -> CategoryAnnotation:
        """Get summary sub categories from `image`.

        Returns:
            CategoryAnnotation: A summary sub category of the image.

        Raises:
            ValueError: If `key` is not available"""

        if self.image:
            return self.image.summary.get_sub_category(key)
        raise AnnotationError(f"Summary does not exist for {self.annotation_id} and key: {key}")

    def get_annotation_map(self) -> defaultdict[str, list[AnnotationMap]]:
        """
        Returns:
             A `defaultdict` with `annotation_id`s as keys and a list of `AnnotationMap` instances as values for all
             sub categories, relationships and image summaries.
        """
        annotation_id_dict = defaultdict(list)
        annotation_id_dict[self.annotation_id].append(AnnotationMap(image_annotation_id=self.annotation_id))
        for sub_cat_key in self.sub_categories:
            sub_cat = self.get_sub_category(sub_cat_key)
            annotation_id_dict[sub_cat.annotation_id].append(
                AnnotationMap(image_annotation_id=self.annotation_id, sub_category_key=sub_cat_key)
            )
        if self.image is not None:
            for summary_cat_key in self.image.summary.sub_categories:
                summary_cat = self.get_summary(summary_cat_key)
                annotation_id_dict[summary_cat.annotation_id].append(
                    AnnotationMap(image_annotation_id=self.annotation_id, summary_key=summary_cat_key)
                )
        for rel_key in self.relationships:
            for rel_ann_ids in self.get_relationship(rel_key):
                annotation_id_dict[rel_ann_ids].append(
                    AnnotationMap(image_annotation_id=self.annotation_id, relationship_key=rel_key)
                )
        return annotation_id_dict


@dataclass
class ContainerAnnotation(CategoryAnnotation):
    """
    A dataclass for transporting values along with categorical attributes. Use these types of annotations as special
    types of sub categories.

    Attributes:
        value: Attribute to store the value. Use strings.
    """

    value: Optional[Union[list[str], str]] = field(default=None)

    def get_defining_attributes(self) -> list[str]:
        return ["category_name", "value"]

    @classmethod
    def from_dict(cls, **kwargs: AnnotationDict) -> ContainerAnnotation:
        container_ann = ann_from_dict(cls, **kwargs)
        value = kwargs.get("value", "")
        container_ann.value = value if isinstance(value, (int, float, str)) else list(value)
        return container_ann
