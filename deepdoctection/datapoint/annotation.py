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
Dataclass for annotations and their derived classes.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..utils import get_uuid, is_uuid_like
from .box import BoundingBox
from .convert import as_dict


@dataclass  # type: ignore
class Annotation(ABC):
    """
    Abstract base class for all types of annotations. This abstract base class only implements general methods for
    correctly assigning annotation_ids. It also has an active flag which is set to True. Only active annotations will be
    returned when querying annotations from image containers.

    Annotation id should never be assigned by yourself. One possibility of assigning an id depending on an external
    value is to set an external id, which in turn creates an annotation id as a function of this.
    An annotation id can only be explicitly set, provided the value is a md5 hash.

    Note that otherwise ids will be generated automatically if the annotation object is dumped in a parent container,
    either an image or annotation (e.g. sub-category). If no id is supplied, the annotation id is created depending
    on the defining attributes (key and value pairs) as specified in the return value of
    :meth:`get_defining_attributes`.

    :attr:`active`: Always set to "True". You can change the value using :meth:`deactivate` .

    :attr:`external_id`: A string or integer value for generating an annotation id. Note, that the resulting annotation
    id will not depend on the defining attributes.

    :attr:`annotation_id`: Unique id for annotations. Will always be given as string representation of a md5-hash.
    """

    active: bool = field(default=True, init=False, repr=True)
    external_id: Optional[Union[str, int]] = field(default=None, init=True, repr=False)
    annotation_id: str = field(init=False, repr=True)
    _annotation_id: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Will check, if the external id provided is an uuid. If not will use the external id as seed for defining an
        uuid.
        """

        if self.external_id is not None:
            external_id = str(self.external_id)
            if is_uuid_like(external_id):
                self.annotation_id = external_id
            else:
                self.annotation_id = get_uuid(external_id)
        self._assert_defining_attributes_have_str()

    @property  # type: ignore
    def annotation_id(self) -> str:  # pylint: disable=E0102
        """
        annotation_id
        """
        if self._annotation_id:
            return self._annotation_id
        raise ValueError("Dump annotation first or pass external_id to create an annotation id")

    @annotation_id.setter
    def annotation_id(self, input_id: str) -> None:
        """
        annotation_id setter
        """
        if self._annotation_id is not None:
            raise AssertionError("annotation_id already defined and cannot be reset")
        if is_uuid_like(input_id):
            self._annotation_id = input_id
        elif isinstance(input_id, property):
            pass
        else:
            raise ValueError("annotation_id must be uuid3 string")

    @abstractmethod
    def get_defining_attributes(self) -> List[str]:  # type: ignore
        """
        Defining attributes of an Annotation instance are attributes, of which you think that they uniquely
        describe the annotation object. If you do not provide an external id, only the defining attributes will be used
        for generating the annotation id.

        :return: A list of attributes.
        """
        NotImplemented  # pylint: disable=W0104

    def _assert_defining_attributes_have_str(self) -> None:
        for attr in self.get_defining_attributes():
            assert hasattr(
                eval("self." + attr), "__str__"  # pylint: disable=W0123
            ), f"attribute {attr} must have __str__ method"

    @staticmethod
    def set_annotation_id(annotation: "CategoryAnnotation", *container_id_context: Optional[str]) -> str:
        """
        Defines the annotation_id by attributes of the annotation class as well as by external parameters given by a
        tuple or list of container id contexts.

        :param annotation: The annotation instance for which the id should be generated.
        :param container_id_context: A tuple/list of strings on which you want the resulting annotation id to depend on.
        :return: A uuid that uniquely characterizes the annotation.
        """
        for container_id in container_id_context:
            if container_id is not None:
                assert is_uuid_like(container_id), f"{container_id} is not a uuid"
        attributes = annotation.get_defining_attributes()
        attributes_values = [str(getattr(annotation, attribute)) for attribute in attributes]
        return get_uuid(*attributes_values, *container_id_context)  # type: ignore

    def as_dict(self) -> Dict[str, Any]:
        """
        Returning the full dataclass as dict. Uses the custom :func:`convert.as_dict` to disregard attributes defined by
        :meth:`remove_keys`.

        :return: A custom dict.
        """

        img_dict = as_dict(self, dict_factory=dict)

        return img_dict

    def deactivate(self) -> None:
        """
        Sets :attr:`active` to False. When calling :meth:`Image.get_annotations` it will be filtered.
        """
        self.active = False

    @abstractmethod
    def get_export(self) -> Dict[str, Any]:  # type: ignore  # pylint: disable=C0116
        """
        Generate a dictionary representing the object that can be saved to a file. See some
        built-in examples e.g. :meth:`CategoryAnnotation.get_export`
        """
        NotImplemented  # pylint: disable=W0104


@dataclass
class CategoryAnnotation(Annotation):
    """
    A general class for storing categories (labels/classes) as well as sub categories (sub-labels/subclasses),
    relationships and prediction scores.

    Subcategories and relationships are stored in a dict, which are populated via the :meth:`dum_sub_category` or
    :meth:`dump_relationship`. If a key is already available as a sub-category, it must be explicitly removed using the
    :meth:`remove_sub_category` before replacing the sub-category.

    Note that subcategories are only accepted as category annotations. Relationships, on the other hand, are only
    managed by passing the annotation id.

    :attr:`category_name`: String will be used for selecting specific annotations. Use upper case strings.

    :attr:`category_id`: When setting a value will accept strings and ints. Will be stored as string.

    :attr:`score`: Score of a prediction.

    :attr:`sub_categories`: Do not access the dict directly. Rather use the access :meth:`get_sub_category` resp.
    :meth:`dump_sub_category`.

    :attr:`relationships`: Do not access the dict directly either. Use :meth:`get_relationship` or
    :meth:`dump_relationship` instead.
    """

    category_name: str = field(default_factory=str)
    _category_id: str = field(default="-1", init=False, repr=False)
    category_id: Union[str, int] = field(default="-1")
    score: Optional[float] = field(default=None)
    _sub_categories: Dict[str, "CategoryAnnotation"] = field(default_factory=dict, init=False, repr=False)
    _relationships: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=False)
    sub_categories: Dict[str, "CategoryAnnotation"] = field(default_factory=dict, init=False, repr=True)
    relationships: Dict[str, List[str]] = field(default_factory=dict, init=False, repr=True)

    def __post_init__(self) -> None:
        assert self.category_name
        super().__post_init__()

    @property  # type: ignore
    def category_id(self) -> str:  # pylint: disable=E0102
        """
        category_id
        """
        return self._category_id

    @category_id.setter
    def category_id(self, input_id: Optional[Union[str, int]]) -> None:
        self._category_id = str(input_id)

    @property  # type: ignore
    def sub_categories(self) -> Dict[str, "CategoryAnnotation"]:  # pylint: disable=E0102
        """
        sub_categories
        """
        return self._sub_categories

    @sub_categories.setter
    def sub_categories(self, inputs: Dict[str, "CategoryAnnotation"]) -> None:  # pylint: disable=E0102
        """
        sub_categories setter
        """

    @property  # type: ignore
    def relationships(self) -> Dict[str, List[str]]:  # pylint: disable=E0102
        """
        relationships
        """
        return self._relationships

    @relationships.setter
    def relationships(self, inputs: Dict[str, List[str]]) -> None:
        """
        relationships setter
        """

    def dump_sub_category(
        self, sub_category_name: str, annotation: "CategoryAnnotation", *container_id_context: Optional[str]
    ) -> None:
        """
        Storage of sub-categories. Since sub-categories usually only depend on very few attributes and the parent
        category cannot yet be stored in a comprehensive container, it is possible to include a context of the
        annotation id in order to ensure that the sub-category annotation id is unambiguously created.

        :param sub_category_name: key for defining the sub category.
        :param annotation: Annotation instance to dump
        :param container_id_context: Tuple/list of context ids.
        """

        assert sub_category_name not in self._sub_categories, (
            f"{sub_category_name} as sub category already defined for " f"{self.annotation_id}"
        )
        if self._annotation_id is not None:
            annotation.annotation_id = self.set_annotation_id(annotation, self.annotation_id, *container_id_context)
        else:
            tmp_annotation_id = self.set_annotation_id(self)
            annotation.annotation_id = annotation.set_annotation_id(
                annotation, tmp_annotation_id, *container_id_context
            )
        self._sub_categories[sub_category_name] = annotation

    def get_sub_category(self, sub_category_name: str) -> "CategoryAnnotation":
        """
        Return a sub category by its key.

        :param sub_category_name: The key of the sub-category.

        :return: sub category as CategoryAnnotation
        """
        return self._sub_categories[sub_category_name]

    def remove_sub_category(self, key: str) -> None:
        """
        Removes a sub category with a given key. Necessary to call, when you want to replace an already dumped sub
        category.

        :param key: A key to a sub category.
        """

        self._sub_categories.pop(key)

    def dump_relationship(self, key: str, annotation_id: str) -> None:
        """
        Dumps an annotation id to a given key, in order to store relations between annotations. Note, that the
        referenced annotation must be stored elsewhere.

        :param key: The key, where to place the annotation id.
        :param annotation_id: An annotation id
        """
        assert is_uuid_like(annotation_id), "annotation_id must be uuid"
        if key not in self._relationships:
            self._relationships[key] = []
        if annotation_id not in self._relationships[key]:
            self._relationships[key].append(annotation_id)

    def get_relationship(self, key: str) -> List[str]:
        """
        Returns a list of annotation ids stored with a given relationship key.

        :param key: The key for the required relationship.
        :return: Get a (possibly) empty list of annotation ids.
        """
        if key in self._relationships:
            return self._relationships[key]
        return []

    def remove_relationship(self, key: str, annotation_ids: Optional[Union[List[str], str]] = None) -> None:
        """
        Remove relationship by some given keys and ids. If no annotation ids are provided all relationship according
        to the key will be removed.

        :param key: A relationship key.
        :param annotation_ids: A single annotation_id or a list. Will remove only the relation with given
                               annotation_ids, provided not None is passed as argument.
        """

        if annotation_ids is not None:
            if isinstance(annotation_ids, str):
                annotation_ids = [annotation_ids]
            for ann_id in annotation_ids:
                try:
                    self._relationships[key].remove(ann_id)
                except ValueError:
                    pass
        else:
            self._relationships[key].clear()

    def get_defining_attributes(self) -> List[str]:
        return ["category_name", "category_id"]

    @staticmethod
    def remove_keys() -> List[str]:
        """
        A list of attributes to suspend from as_dict creation.

        :return: List of attributes.
        """
        return ["_annotation_id", "_category_id", "_relationships", "_sub_categories", "external_id"]

    def get_export(self) -> Dict[str, Any]:
        """
        Exporting annotations as dictionary.

        :return: dict that e.g. can be saved to a file.
        """
        ann_copy = deepcopy(self)
        ann_copy._sub_categories = {}  # pylint: disable=W0212
        export_dict = self.as_dict()

        export_dict["sub_categories"] = {}

        for key, value in self._sub_categories.items():
            export_dict["sub_categories"][key] = value.get_export()
        try:
            int(export_dict["category_id"])
        except ValueError:
            export_dict["category_id"] = None

        return export_dict


@dataclass
class ImageAnnotation(CategoryAnnotation):
    """
    A general class for storing annotations related to object detection tasks. In addition to the inherited attributes,
    the class contains a bounding box and an image attribute. The image attribute is optional and is suitable for
    generating an image from the annotation and then saving it there. Compare with the method :meth:`image.Image.
    image_ann_to_image`, which naturally populates this attribute.

    :attr:`bounding_box`: Regarding the coordinate system, if you have to define a prediction, use the system of the
                          image where the object has been detected.

    :attr:`image`: Image, defined by the bounding box and cropped from its parent image. Populate this attribute with
                   :meth:`Image.image_ann_to_image`.
    """

    bounding_box: Optional[BoundingBox] = field(default=None)
    image: Optional["Image"] = field(default=None, init=False, repr=False)  # type:ignore

    def get_defining_attributes(self) -> List[str]:
        return ["category_name", "bounding_box"]

    def get_export(self) -> Dict[str, str]:
        """
        Exporting annotations as dictionary.

        :return: dict that e.g. can be saved to a file.
        """

        export_dict = super().get_export()
        if export_dict["image"] is not None:
            export_dict["embeddings"] = export_dict["image"]["embeddings"]
        export_dict.pop("image")

        return export_dict


@dataclass
class SummaryAnnotation(ImageAnnotation):
    """
    A dataclass for adding summaries. The various summaries can be stored as sub categories.

    Summary annotations should be stored in the attribute provided: :attr:`image.Image.summary`  and should not be
    dumped as a category.
    """

    def __post_init__(self) -> None:
        self.category_name = "SUMMARY"
        super().__post_init__()


@dataclass
class ContainerAnnotation(CategoryAnnotation):
    """
    A dataclass for transporting values along with categorical attributes. Use these types of annotations as special
    types of sub categories.

    :attr value: Attribute to store the value. Use strings.
    """

    value: Optional[Union[List[str], str]] = field(default=None)

    def get_defining_attributes(self) -> List[str]:
        return ["category_name", "value"]
