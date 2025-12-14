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
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar, Union, Type

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from ..utils.error import AnnotationError, UUIDError
from ..utils.identifier import get_uuid, is_uuid_like
from ..utils.logger import LoggingRecord, logger
from ..utils.object_types import DefaultType, ObjectTypes, TypeOrStr, get_type
from ..utils.types import AnnotationDict
from .box import BoundingBox


@dataclass(frozen=True)
class AnnotationMap:
    """AnnotationMap to store all sub categories, relationship keys and summary keys of an annotation"""

    image_annotation_id: str
    sub_category_key: Optional[ObjectTypes] = None
    relationship_key: Optional[ObjectTypes] = None
    summary_key: Optional[ObjectTypes] = None


DEFAULT_CATEGORY_ID = -1

T = TypeVar("T", str, list[str], None)
A = TypeVar("A", bound="Annotation")

class Annotation(BaseModel, ABC):
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

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
    }

    active: bool = Field(default=True)
    external_id: Optional[Union[str, int]] = Field(default=None)
    _annotation_id: Optional[str] = PrivateAttr(default=None)
    service_id: Optional[str] = Field(default=None)
    model_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)

    def __init__(self, **data: Any) -> None:
        """
        Accept `_annotation_id` in kwargs (e.g. CategoryAnnotation(**item)),
        remove it before BaseModel initialization and set the PrivateAttr after.
        """
        _annotation_id = data.pop("_annotation_id", None)
        super().__init__(**data)
        if _annotation_id is not None:
            object.__setattr__(self, "_annotation_id", _annotation_id)

    @model_validator(mode="after")
    def _setup_annotation_id(self) -> Annotation:
        """Set up annotation_id from external_id if provided."""
        if self.external_id is not None and self._annotation_id is None:
            external_id = str(self.external_id)
            if is_uuid_like(external_id):
                object.__setattr__(self, "_annotation_id", external_id)
            else:
                object.__setattr__(self, "_annotation_id", get_uuid(external_id))
        return self

    @property
    def annotation_id(self) -> str:
        """Get annotation_id."""
        if self._annotation_id:
            return self._annotation_id
        raise AnnotationError("Dump annotation first or pass external_id to create an annotation id")

    @annotation_id.setter
    def annotation_id(self, input_id: str) -> None:
        """Set annotation_id."""
        if self._annotation_id is not None:
            raise AnnotationError("Annotation_id already defined and cannot be reset")
        if is_uuid_like(input_id):
            self._annotation_id = input_id
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
        filtered_context = []
        for container_id in container_id_context:
            if container_id is not None:
                assert is_uuid_like(container_id), f"{container_id} is not a uuid"
                filtered_context.append(container_id)

        attributes = annotation.get_defining_attributes()
        attributes_values = [str(getattr(annotation, attribute)) for attribute in attributes]

        return get_uuid(*attributes_values, *filtered_context)

    def as_dict(self) -> AnnotationDict:
        """Return the model as dict."""
        return self.model_dump(by_alias=True, exclude_none=False)

    def deactivate(self) -> None:
        """
        Sets `active` to False. When calling `Image.get_annotations()` it will be filtered.
        """
        self.active = False

    @classmethod
    def from_dict(cls: Type[A], **kwargs: AnnotationDict) -> A:
        """
        Method to initialize a derived class from dict.

        Args:
            kwargs: dict with `Annotation` attributes

        Returns:
            Annotation instance
        """
        return cls(**kwargs)

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
        """Generate state_id from state attributes."""
        container_ids = []
        attributes = self.get_state_attributes()

        for attribute in attributes:
            attr = getattr(self, attribute)
            if isinstance(attr, dict):
                for key, value in attr.items():
                    if isinstance(value, Annotation):
                        container_ids.extend([str(key), value.state_id])
                    elif isinstance(value, list):
                        container_ids.extend([str(element) for element in value])
                    else:
                        raise TypeError(f"Cannot determine __str__ or annotation_id for element in {attribute}")
            elif isinstance(attr, list):
                for element in attr:
                    if isinstance(element, Annotation):
                        container_ids.append(element.state_id)
                    elif isinstance(element, str):
                        container_ids.append(element)
                    else:
                        container_ids.append(str(element))
            elif hasattr(attr, "state_id"):
                container_ids.append(attr.state_id)
            else:
                container_ids.append(str(attr))

        return get_uuid(self.annotation_id, *container_ids)


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

    category_name: TypeOrStr = Field(default=DefaultType.DEFAULT_TYPE)
    category_id: int = Field(default=DEFAULT_CATEGORY_ID)
    score: Optional[float] = Field(default=None)
    sub_categories: dict[ObjectTypes, CategoryAnnotation] = Field(default_factory=dict)
    relationships: dict[ObjectTypes, list[str]] = Field(default_factory=dict)

    @field_validator("category_name", mode="before")
    @classmethod
    def _validate_category_name(cls, v: TypeOrStr) -> ObjectTypes:
        """Convert string to ObjectTypes if needed."""
        if isinstance(v, str):
            return get_type(v)
        return v

    @field_validator("category_id", mode="before")
    @classmethod
    def _validate_category_id(cls, v: Any) -> int:
        """Convert category_id to int, handling special cases."""
        if v in ("None", "", None):
            return DEFAULT_CATEGORY_ID
        return int(v)

    @field_validator("score", mode="before")
    @classmethod
    def _validate_score(cls, v: Any) -> Optional[float]:
        """Validate score is between 0 and 1 and limit to 8 digits."""
        if v is None:
            return None
        score = float(v)
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be between 0 and 1 (inclusive), got {score}")
        return round(score, 8)

    @field_validator("sub_categories", mode="before")
    @classmethod
    def _coerce_sub_categories(cls, v: Any) -> dict[ObjectTypes, CategoryAnnotation]:
        """
        Accept dicts with string or ObjectTypes keys and values that are either:
          - CategoryAnnotation1/ContainerAnnotation1 instances, or
          - dict payloads for those types.
        Rule: if 'value' in payload -> ContainerAnnotation1, else CategoryAnnotation1.
        """
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("sub_categories must be a dict")

        out: dict[ObjectTypes, CategoryAnnotation] = {}
        for key, val in v.items():
            key_type: ObjectTypes = get_type(key) if isinstance(key, str) else key  # keep enum keys as-is

            if isinstance(val, CategoryAnnotation):
                # also covers ContainerAnnotation1 since it inherits CategoryAnnotation1
                out[key_type] = val
            elif isinstance(val, dict):
                # decide target type without ann_from_dict
                if "value" in val:
                    out[key_type] = ContainerAnnotation(**val)
                else:
                    out[key_type] = CategoryAnnotation(**val)
            else:
                raise TypeError("sub_categories values must be dict or CategoryAnnotation1/ContainerAnnotation1")
        return out

    # python
    @field_validator("relationships", mode="before")
    @classmethod
    def _coerce_relationships(cls, v: Any) -> dict[ObjectTypes, list[str]]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("relationships must be a dict")
        new: dict[ObjectTypes, list[str]] = {}
        for key, val in v.items():
            key_type = get_type(key) if isinstance(key, str) else key
            if not isinstance(val, list):
                raise TypeError("relationships values must be a list of ids")
            # remove duplicates (via set) and convert back to list
            deduped = list({str(x) for x in val})
            # validate UUIDs
            for sid in deduped:
                if not is_uuid_like(sid):
                    raise UUIDError(f"Relationship id must be uuid: {sid}")
            new[key_type] = deduped
        return new

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
        key = get_type(sub_category_name)

        if key in self.sub_categories:
            raise AnnotationError(
                f"sub category {sub_category_name} already defined: "
                f"annotation_id: {self.annotation_id}, "
                f"category_name: {self.category_name}, "
                f"category_id: {self.category_id}"
            )

        if self._annotation_id is not None:
            if annotation._annotation_id is None:
                annotation.annotation_id = self.set_annotation_id(annotation, self.annotation_id, *container_id_context)
        else:
            tmp_annotation_id = self.set_annotation_id(self)
            if annotation._annotation_id is None:
                annotation.annotation_id = annotation.set_annotation_id(
                    annotation, tmp_annotation_id, *container_id_context
                )

        self.sub_categories[key] = annotation

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
        sub_categories = getattr(self, "sub_categories", None)
        if (
            sub_categories is not None
            and isinstance(sub_categories, dict)
            and hasattr(sub_categories, "pop")
            and key in sub_categories
        ):
            sub_categories.pop(key)

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
        if key_type not in self.relationships:
            self.relationships[key_type] = []
        if annotation_id not in self.relationships[key_type]:
            self.relationships[key_type].append(annotation_id)

    def get_relationship(self, key: TypeOrStr) -> list[str]:
        """
        Returns a list of annotation ids stored with a given relationship key.

        Args:
            key: The key for the required relationship.

        Returns:
            A (possibly) empty list of `annotation_id`s.
        """
        relationships = getattr(self, "relationships", None)
        if relationships is not None and isinstance(relationships, dict) and hasattr(relationships, "get"):
            return relationships.get(get_type(key), [])
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
    def get_state_attributes() -> list[str]:
        return ["active", "sub_categories", "relationships"]

    def __repr__(self) -> str:
        return (
            f"CategoryAnnotation(annotation_id: {self._annotation_id}, category_name={self.category_name},"
            f"category_id={self.category_id}, score={self.score}, sub_categories={self.sub_categories},"
            f" relationships={self.relationships})"
        )

    def __str__(self) -> str:
        return repr(self)


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

    model_config = {
        "arbitrary_types_allowed": False,
        "validate_assignment": True,
    }

    bounding_box: Optional[BoundingBox] = Field(default=None)
    image: Optional[Any] = Field(default=None)

    @field_validator("image", mode="before")
    @classmethod
    def _coerce_image(cls, v: Any) -> Optional[Any]:
        """
        Coerce dict payloads into an Image instance. Import `Image` locally to avoid
        circular import between `image.py` and `annotation.py`.
        """
        if v is None:
            return None

        try:
            from .image import Image  # pylint: disable=C0415 # local import to avoid circular import
        except (ImportError, ModuleNotFoundError):
            return v
        if isinstance(v, Image):
            return v
        if isinstance(v, dict):
            return Image(**v)
        raise TypeError("image must be Image or dict")

    @field_validator("bounding_box", mode="before")
    @classmethod
    def _coerce_bounding_box(cls, v: Any) -> Optional[BoundingBox]:
        if v is None or isinstance(v, BoundingBox):
            return v
        if isinstance(v, dict):
            # ensure proper init from dict payload
            return BoundingBox(**v)
        raise TypeError("bounding_box must be a BoundingBox or a dict")

    def get_defining_attributes(self) -> list[str]:
        return ["category_name", "bounding_box"]

    @staticmethod
    def get_state_attributes() -> list[str]:
        return ["active", "sub_categories", "relationships", "image"]

    def get_bounding_box(self, image_id: Optional[str] = None) -> BoundingBox:
        """
        Get bounding from image embeddings or, if not available or if `image_id` is not provided,
        from `bounding_box`.
        """
        image_obj = self.image  # use local variable so static analyzers treat it as instance value
        if image_obj and image_id:
            if hasattr(image_obj, "get_embedding"):
                box = getattr(image_obj, "get_embedding")(image_id)
            else:
                box = self.bounding_box
        else:
            box = self.bounding_box

        if box:
            return box
        raise AnnotationError(f"bounding_box has not been initialized for {self.annotation_id}")

    def get_summary(self, key: ObjectTypes) -> CategoryAnnotation:
        """
        Get summary sub categories from `image`.

        Returns:
            CategoryAnnotation: A summary sub category of the image.

        Raises:
            AnnotationError: If `key` is not available or image/summary missing.
        """
        image_obj = self.image
        if image_obj and hasattr(image_obj, "summary"):
            summary = getattr(image_obj, "summary")
            return summary.get_sub_category(key)
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

        image_obj = self.image
        if image_obj is not None and hasattr(image_obj, "summary"):
            summary_obj = getattr(image_obj, "summary")
            for summary_cat_key in summary_obj.sub_categories:
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

    def __repr__(self) -> str:
        return (
            f"ImageAnnotation(annotation_id: {self._annotation_id}, category_name={self.category_name},"
            f"category_id={self.category_id}, score={self.score}, bounding_box: {self.bounding_box}, "
            f"sub_categories={self.sub_categories}, relationships={self.relationships})"
        )

    def __str__(self) -> str:
        return repr(self)


class ContainerAnnotation(CategoryAnnotation):
    """
    Container annotation with typed value.
    Rules:
    - If initialized with a value and no type is set, infer the type from the value (keep native int/float).
    - Calling set_type(<type>) enforces that type; existing value is converted if possible.
    - Calling set_type(None) disables validation and coerces current value to its string (or list[str]) form.
    """

    value: Optional[Union[list[str], str, int, float]] = Field(default=None)
    value_type: Optional[Literal["str", "int", "float", "list[str]"]] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _coerce_or_infer_value_validator(self) -> ContainerAnnotation:
        return self._coerce_or_infer_value()

    def _coerce_or_infer_value(self) -> ContainerAnnotation:
        effective_type = self.value_type
        # No explicit type: infer once
        if effective_type is None:
            if self.value is None:
                return self
            if isinstance(self.value, int):
                self.value_type = "int"
            elif isinstance(self.value, float):
                self.value_type = "float"
            elif isinstance(self.value, str):
                self.value_type = "str"
            elif isinstance(self.value, list):
                if all(isinstance(el, str) for el in self.value):
                    self.value_type = "list[str]"
                else:
                    object.__setattr__(self, "value", [str(el) for el in self.value])
                    self.value_type = "list[str]"
            return self

        # Explicit type: enforce / convert
        if self.value is None:
            return self

        if effective_type == "int":
            if isinstance(self.value, int):
                return self
            v = getattr(self, "value", None)
            if isinstance(v, int):
                return self
            if isinstance(v, str) and v.isdigit():
                object.__setattr__(self, "value", int(v))
                return self
            raise TypeError(f"value must be int when type='int', got {type(self.value).__name__}")

        if effective_type == "float":
            if isinstance(self.value, float):
                return self
            if isinstance(self.value, int):
                object.__setattr__(self, "value", float(self.value))
                return self
            v = self.value
            if isinstance(v, str):
                try:
                    object.__setattr__(self, "value", float(v))
                    return self
                except ValueError:
                    pass
            raise TypeError(f"value must be float when type='float', got {type(self.value).__name__}")

        if effective_type == "str":
            if not isinstance(self.value, str):
                object.__setattr__(self, "value", str(self.value))
            return self

        if effective_type == "list[str]":
            if not isinstance(self.value, list):
                raise TypeError("value must be list[str] when type='list[str]'")
            if not all(isinstance(el, str) for el in self.value):
                object.__setattr__(self, "value", [str(el) for el in self.value])
            return self

        raise ValueError(f"Unsupported type {effective_type}")

    def set_type(self, value_type: Literal["str", "int", "float", "list[str]"]) -> None:
        """
        Set and enforce the value type for this ContainerAnnotation and coerce the current value.

        Args:
            value_type: One of `"str"`, `"int"`, `"float"`, or `"list[str]"`. (The current implementation rejects
             `None`.)

        Raises:
            ValueError: If `type` is `None` or not one of the allowed values.
            TypeError: May be raised by `_coerce_or_infer_value` if the existing `value` cannot be converted.
        """
        if value_type is None:
            raise ValueError(f"type cannot be None, current value_type is {self.value_type}")

        allowed = {"str", "int", "float", "list[str]"}
        if value_type not in allowed:
            raise ValueError(f"type must be one of {sorted(allowed)}")
        self.value_type = value_type
        self._coerce_or_infer_value()

    def get_defining_attributes(self) -> list[str]:
        return ["category_name", "value"]

    def __repr__(self) -> str:
        return (
            f"ContainerAnnotation(annotation_id: {self.annotation_id}, category_name={self.category_name},"
            f"category_id={self.category_id}, score={self.score}, sub_categories={self.sub_categories},"
            f" relationships={self.relationships})"
        )
