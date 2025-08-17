# -*- coding: utf-8 -*-
# File: image.py

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
Dataclass `Image`
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from os import environ, fspath
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict, Union, no_type_check

import numpy as np
from numpy import uint8

from ..utils.error import AnnotationError, BoundingBoxError, ImageError, UUIDError
from ..utils.identifier import get_uuid, is_uuid_like
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import ObjectTypes, SummaryType, get_type
from ..utils.types import ImageDict, PathLikeOrStr, PixelValues
from .annotation import Annotation, AnnotationMap, BoundingBox, CategoryAnnotation, ImageAnnotation
from .box import crop_box_from_image, global_to_local_coords, intersection_box
from .convert import as_dict, convert_b64_to_np_array, convert_np_array_to_b64, convert_pdf_bytes_to_np_array_v2


class MetaAnnotationDict(TypedDict):
    """MetaAnnotationDict"""

    image_annotations: list[str]
    sub_categories: dict[str, dict[str, list[str]]]
    relationships: dict[str, list[str]]
    summaries: list[str]


@dataclass(frozen=True)
class MetaAnnotation:
    """
    An immutable dataclass that stores information about what `Image` are being
    modified through a pipeline component.

    Attributes:
        image_annotations: Tuple of `ObjectTypes` representing image annotations.
        sub_categories: Dictionary mapping `ObjectTypes` to dicts of `ObjectTypes` to sets of `ObjectTypes`
        for sub-categories.
        relationships: Dictionary mapping `ObjectTypes` to sets of `ObjectTypes` for relationships.
        summaries: Tuple of `ObjectTypes` representing summaries.
    """

    image_annotations: tuple[ObjectTypes, ...] = field(default=())
    sub_categories: dict[ObjectTypes, dict[ObjectTypes, set[ObjectTypes]]] = field(default_factory=dict)
    relationships: dict[ObjectTypes, set[ObjectTypes]] = field(default_factory=dict)
    summaries: tuple[ObjectTypes, ...] = field(default=())

    def as_dict(self) -> MetaAnnotationDict:
        """
        Returns the MetaAnnotation as a dictionary, with all `ObjectTypes` converted to strings.

        Returns:
            A dictionary representation of the MetaAnnotation where all `ObjectTypes` are converted to strings.
        """
        return {
            "image_annotations": [obj.value for obj in self.image_annotations],
            "sub_categories": {
                outer_key.value: {
                    inner_key.value: [val.value for val in inner_values]
                    for inner_key, inner_values in outer_value.items()
                }
                for outer_key, outer_value in self.sub_categories.items()
            },
            "relationships": {key.value: [val.value for val in values] for key, values in self.relationships.items()},
            "summaries": [obj.value for obj in self.summaries],
        }


@dataclass
class Image:
    """
    The image object is the enclosing data class that is used in the core data model to manage, retrieve or store
    all information during processing. It contains metadata belonging to the image, but also the image itself
    and annotations, either given as ground truth or determined via a processing path. In addition, there are
    storage options in order not to recalculate coordinates in relation to other images.

    Data points from datasets must be mapped in this format so that the processing tools (pipeline components) can
    be called up without further adjustment.

    An image can be provided with an `image_id` by providing the `external_id`, which can be clearly identified
    as a `md5` hash. If such an id is not given, an `image_id` will be derived from `file_name` and, if necessary,
    from `location`.

    All other attributes represent containers (lists or dicts) that can be populated and managed using their own method.

    In `image`, the image may be saved as `np.array`. Allocation as `base64` encoding string or as pdf bytes are
    possible and are converted via a `image.setter`. Other formats are rejected.
    If an image of a given size is added, the width and height of the image are determined.

    Using `embeddings`, various bounding boxes can be saved that describe the position of the image as a
    sub-image. The bounding box is accessed in relation to the embedding image via the `annotation_id`.
    Embeddings are often used in connection with annotations in which the `image` is populated.

    All `ImageAnnotations` of the image are saved in the list annotations. Other types of annotation are
    not permitted.

    Args:
        file_name: Should be equal to the name of a physical file representing the image. If the image is part
                   of a larger document (e.g. pdf-document) the file_name should be populated as a concatenation of
                   the document file and its page number.
        location: Full path to the document or to the physical file. Loading functions from disk use this attribute.
        document_id: A unique identifier for the document. If not set, it will be set to the `image_id`.
        page_number: The page number of the image in the document. If not set, it will be set to 0.
        external_id: A string or integer value for generating an `image_id`.
        _image_id: A unique identifier for the image. If not set, it will be set to a generated `uuid`.
        _image: The image as a numpy array. If not set, it will be set to None. Do not set this attribute directly.
        _bbox: The bounding box of the image. If not set, it will be set to None. Do not set this attribute directly.
        embeddings: A dictionary of `image_id` to `BoundingBox`es. If not set, it will be set to an empty dict.
        annotations: A list of `ImageAnnotation` objects. Use `get_annotation` to retrieve annotations.
        _annotation_ids: A list of `annotation_id`s. Used internally to ensure uniqueness of annotations.
        _summary: A `CategoryAnnotation` for image-level informations. If not set, it will be set to None.

    """

    file_name: str
    location: str = field(default="")
    document_id: str = field(default="", init=False, repr=True)
    page_number: int = field(default=0, init=False, repr=False)
    external_id: Optional[Union[str, int]] = field(default=None, repr=False)
    _image_id: Optional[str] = field(default=None, init=False, repr=True)
    _image: Optional[PixelValues] = field(default=None, init=False, repr=False)
    _bbox: Optional[BoundingBox] = field(default=None, init=False, repr=False)
    embeddings: dict[str, BoundingBox] = field(default_factory=dict, init=False, repr=True)
    annotations: list[ImageAnnotation] = field(default_factory=list, init=False, repr=True)
    _annotation_ids: list[str] = field(default_factory=list, init=False, repr=False)
    _summary: Optional[CategoryAnnotation] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.external_id is not None:
            external_id = str(self.external_id)
            if is_uuid_like(external_id):
                self.image_id = external_id
            else:
                self.image_id = get_uuid(external_id)
        else:
            self.image_id = get_uuid(str(self.location), self.file_name)
        self.document_id = self.image_id

    @property
    def image_id(self) -> str:
        """
        `image_id`
        """
        if self._image_id is not None:
            return self._image_id
        raise ImageError("image_id not set")

    @image_id.setter
    def image_id(self, input_id: str) -> None:
        """
        `image_id` setter
        """
        if self._image_id is not None:
            raise ImageError("image_id already defined and cannot be reset")
        if is_uuid_like(input_id):
            self._image_id = input_id
        elif isinstance(input_id, property):
            pass
        else:
            raise UUIDError("image_id must be uuid3 string")

    @property
    def image(self) -> Optional[PixelValues]:
        """
        image
        """
        return self._image

    @image.setter
    def image(self, image: Optional[Union[str, PixelValues, bytes]]) -> None:
        """
        Sets the image for internal storage. Will convert to numpy array before storing internally.

        Note:
            If the input is an np.array, ensure that the image is in BGR-format as this is the standard
            format for the whole package.

        Args:
            image: Accepts `np.array`s, `base64` encodings or `bytes` generated from pdf documents.
                   Everything else will be rejected.
        """

        if isinstance(image, property):
            pass
        elif isinstance(image, str):
            self._image = convert_b64_to_np_array(image)
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()
        elif isinstance(image, bytes):
            self._image = convert_pdf_bytes_to_np_array_v2(image, dpi=int(environ["DPI"]))
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()
        else:
            if not isinstance(image, np.ndarray):
                raise ImageError(f"Cannot load image is of type: {type(image)}")
            self._image = image.astype(uint8)
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()

    @property
    def summary(self) -> CategoryAnnotation:
        """summary"""
        if self._summary is None:
            self._summary = CategoryAnnotation(category_name=SummaryType.SUMMARY)
            if self._summary._annotation_id is None:  # pylint: disable=W0212
                self._summary.annotation_id = self.define_annotation_id(self._summary)
        return self._summary

    @summary.setter
    def summary(self, summary_annotation: CategoryAnnotation) -> None:
        """summary setter"""
        if self._summary is not None:
            raise ImageError("Image.summary already defined and cannot be reset")
        if summary_annotation._annotation_id is None:  # pylint: disable=W0212
            summary_annotation.annotation_id = self.define_annotation_id(summary_annotation)
        self._summary = summary_annotation

    @property
    def pdf_bytes(self) -> Optional[bytes]:
        """
        `pdf_bytes`. This attribute will be set dynamically and is not part of the core Image data model
        """
        if hasattr(self, "_pdf_bytes"):
            return getattr(self, "_pdf_bytes")
        return None

    @pdf_bytes.setter
    def pdf_bytes(self, pdf_bytes: bytes) -> None:
        """
        `pdf_bytes` setter
        """
        assert isinstance(pdf_bytes, bytes)
        if not hasattr(self, "_pdf_bytes"):
            setattr(self, "_pdf_bytes", pdf_bytes)

    def clear_image(self, clear_bbox: bool = False) -> None:
        """
        Removes the `Image.image`. Useful, if the image must be a lightweight object.

        Args:
            clear_bbox: If set to `True` it will remove the image width and height. This is necessary,
                        if the image is going to be replaced with a transform. It will also remove the self
                        embedding entry
        """
        self._image = None
        if clear_bbox:
            self._bbox = None
            self.embeddings.pop(self.image_id)

    def get_image(self) -> _Img:  # type: ignore # pylint: disable=E0602
        """
        Get the image either in base64 string representation or as `np.array`.

        Example:
            ```python
            image.get_image().to_np_array()
            ```

            or

            ```python
            image.get_image().to_b64()
            ```

        Returns:
            Desired image encoding representation
        """

        class _Img:
            """
            Helper class. Do not use it.
            """

            def __init__(self, img: Optional[PixelValues]):
                self.img = img

            def to_np_array(self) -> Optional[PixelValues]:
                """
                Returns image as numpy array

                :return: array
                """
                return self.img

            def to_b64(self) -> Optional[str]:
                """
                Returns image as b64 encoded string

                :return: b64 encoded string
                """
                if self.img is not None:
                    return convert_np_array_to_b64(self.img)
                return self.img

        return _Img(self.image)

    @property
    def width(self) -> float:
        """
        `width`
        """
        if self._bbox is None:
            raise ImageError("Width not available. Call set_width_height first")
        return self._bbox.width

    @property
    def height(self) -> float:
        """
        `height`
        """
        if self._bbox is None:
            raise ImageError("Height not available. Call set_width_height first")
        return self._bbox.height

    def set_width_height(self, width: float, height: float) -> None:
        """
        Defines bounding box of the image if not already set. Use this, if you do not want to keep the image separated
        for memory reasons.

        Args:
            width: width of image
            height: height of image
        """
        if self._bbox is None:
            self._bbox = BoundingBox(ulx=0.0, uly=0.0, height=height, width=width, absolute_coords=True)
            self._self_embedding()

    def set_embedding(self, image_id: str, bounding_box: BoundingBox) -> None:
        """
        Set embedding pair. Pass an image_id and a bounding box defining the spacial position of this image with
        respect to the embedding image.

        Args:
            image_id: A uuid of the embedding image.
            bounding_box: bounding box of this image in terms of the embedding image.
        """
        if not isinstance(bounding_box, BoundingBox):
            raise BoundingBoxError(f"Bounding box must be of type BoundingBox, is of type {type(bounding_box)}")
        self.embeddings[image_id] = bounding_box

    def get_embedding(self, image_id: str) -> BoundingBox:
        """
        Returns the bounding box according to the `image_id`.

        Args:
            image_id: uuid string of the embedding image

        Returns:
            The bounding box of this instance in terms of the embedding image
        """

        return self.embeddings[image_id]

    def remove_embedding(self, image_id: str) -> None:
        """
        Remove an embedding from the image.

        Args:
            image_id: `uuid` string of the embedding image
        """
        if image_id in self.embeddings:
            self.embeddings.pop(image_id)

    def _self_embedding(self) -> None:
        if self._bbox is not None:
            self.set_embedding(self.image_id, self._bbox)

    def dump(self, annotation: ImageAnnotation) -> None:
        """
        Dump an annotation to the Image dataclass. This is the central method for associating an annotation with
        an image. It gives the annotation an `annotation_id` in relation to the `image_id` in order to ensure uniqueness
        across all images.

        Args:
            annotation: image annotation to store
        """
        if not isinstance(annotation, ImageAnnotation):
            raise AnnotationError(
                f"Annotation must be of type ImageAnnotation: "
                f"{annotation.annotation_id} but is of type {str(type(annotation))}"
            )
        if annotation._annotation_id is None:  # pylint: disable=W0212
            annotation.annotation_id = self.define_annotation_id(annotation)
        if annotation.annotation_id in self._annotation_ids:
            raise ImageError(f"Cannot dump annotation with already taken " f"id {annotation.annotation_id}")
        self._annotation_ids.append(annotation.annotation_id)
        self.annotations.append(annotation)

    def get_annotation(
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        service_ids: Optional[Union[str, Sequence[str]]] = None,
        model_id: Optional[Union[str, Sequence[str]]] = None,
        session_ids: Optional[Union[str, Sequence[str]]] = None,
        ignore_inactive: bool = True,
    ) -> list[ImageAnnotation]:
        """
        Selection of annotations from the annotation container. Filter conditions can be defined by specifying
        the `annotation_id` or `category_name`.
        Only annotations that have  active = 'True' are returned. If more than one condition is provided, only
        annotations will be returned that satisfy all conditions.
        If no condition is provided, it will return all active annotations.

        Args:
            category_names: A single name or list of names
            annotation_ids: A single id or list of ids
            service_ids: A single service name or list of service names
            model_id: A single model name or list of model names
            session_ids: A single session id or list of session ids
            ignore_inactive: If set to `True` only active annotations are returned.

        Returns:
            A (possibly empty) list of `ImageAnnotation`s
        """

        if category_names is not None:
            category_names = (
                (get_type(category_names),)
                if isinstance(category_names, str)
                else tuple(get_type(cat_name) for cat_name in category_names)
            )

        ann_ids = [annotation_ids] if isinstance(annotation_ids, str) else annotation_ids
        service_ids = [service_ids] if isinstance(service_ids, str) else service_ids
        model_id = [model_id] if isinstance(model_id, str) else model_id
        session_id = [session_ids] if isinstance(session_ids, str) else session_ids

        if ignore_inactive:
            anns: Union[list[ImageAnnotation], filter[ImageAnnotation]] = filter(lambda x: x.active, self.annotations)
        else:
            anns = self.annotations

        if category_names is not None:
            anns = filter(lambda x: x.category_name in category_names, anns)

        if ann_ids is not None:
            anns = filter(lambda x: x.annotation_id in ann_ids, anns)

        if service_ids is not None:
            anns = filter(lambda x: x.service_id in service_ids, anns)

        if model_id is not None:
            anns = filter(lambda x: x.model_id in model_id, anns)

        if session_id is not None:
            anns = filter(lambda x: x.session_id in session_id, anns)

        return list(anns)

    def as_dict(self) -> dict[str, Any]:
        """
        Returns the full image dataclass as dict. Uses the custom `convert.as_dict` to disregard attributes
        defined by `remove_keys`.

        Returns:
            A custom `dict`.
        """

        img_dict = as_dict(self, dict_factory=dict)
        if self.image is not None:
            img_dict["_image"] = convert_np_array_to_b64(self.image)
        else:
            img_dict["_image"] = None
        return img_dict

    def as_json(self) -> str:
        """
        Returns the full image dataclass as json string.

        Returns:
            A `JSON` object.
        """

        return json.dumps(self.as_dict(), indent=4)

    @staticmethod
    def remove_keys() -> list[str]:
        """
        A list of attributes to suspend from `as_dict` creation.
        """

        return ["_annotation_ids", "_category_name"]

    def define_annotation_id(self, annotation: Annotation) -> str:
        """
        Generate a uuid for a given annotation. To guarantee uniqueness the generation depends on the datapoint
        `image_id` as well as on the annotation.

        Args:
            annotation:  An annotation to generate the `uuid` for

        Returns:
            uuid string
        """

        attributes = annotation.get_defining_attributes()
        attributes_values = [
            str(getattr(annotation, attribute))
            if attribute != "bounding_box"
            else getattr(annotation, "bounding_box").get_legacy_string()
            for attribute in attributes
        ]
        return get_uuid(*attributes_values, str(self.image_id))

    def remove(
        self,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        service_ids: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        """
        Instead of removing consider deactivating annotations.

        Calls `List.remove`.

        Args:
            annotation_ids: The annotation to remove
            service_ids: The service id to remove

        Raises:
            ValueError: If the annotation or service id is not found in the image.
        """
        ann_id_to_annotation_maps = self.get_annotation_id_to_annotation_maps()

        if annotation_ids is not None:
            annotation_ids = [annotation_ids] if isinstance(annotation_ids, str) else annotation_ids

            for ann_id in annotation_ids:
                if ann_id not in ann_id_to_annotation_maps:
                    raise ImageError(f"Annotation with id {ann_id} not found")
                annotation_maps = ann_id_to_annotation_maps[ann_id]

                for annotation_map in annotation_maps:
                    self._remove_by_annotation_id(ann_id, annotation_map)

        if service_ids is not None:
            service_ids = [service_ids] if isinstance(service_ids, str) else service_ids
            service_id_to_annotation_id = self.get_service_id_to_annotation_id()

            for service_id in service_ids:
                if service_id not in service_id_to_annotation_id:
                    logger.info(
                        LoggingRecord(
                            f"Service_id {service_id} for image_id: {self.image_id} not found. Skipping removal."
                        )
                    )

                annotation_ids = service_id_to_annotation_id.get(service_id, [])

                for ann_id in annotation_ids:
                    if ann_id not in ann_id_to_annotation_maps:
                        raise ImageError(f"Annotation with id {ann_id} not found")
                    annotation_maps = ann_id_to_annotation_maps[ann_id]

                    for annotation_map in annotation_maps:
                        self._remove_by_annotation_id(ann_id, annotation_map)

    def _remove_by_annotation_id(self, annotation_id: str, location_dict: AnnotationMap) -> None:
        image_annotation_id = location_dict.image_annotation_id
        annotations = self.get_annotation(annotation_ids=image_annotation_id)
        if not annotations:
            return
        # There can only be one annotation with a given id
        annotation = annotations[0]

        if (
            location_dict.sub_category_key is None
            and location_dict.relationship_key is None
            and location_dict.summary_key is None
        ):
            self.annotations.remove(annotation)
            self._annotation_ids.remove(annotation.annotation_id)

        sub_category_key = location_dict.sub_category_key

        if sub_category_key is not None:
            annotation.remove_sub_category(sub_category_key)

        relationship_key = location_dict.relationship_key

        if relationship_key is not None:
            annotation.remove_relationship(relationship_key, annotation_id)

        summary_key = location_dict.summary_key
        if summary_key is not None:
            if annotation.image is not None:
                annotation.image.summary.remove_sub_category(summary_key)

    def image_ann_to_image(self, annotation_id: str, crop_image: bool = False) -> None:
        """
        This method is an operation that changes the state of an underlying dumped image annotation and that
        manages `ImageAnnotation.image`. An image object is generated, which is interpreted as part of the image
        by the bounding box. The image is cut out and the determinable fields such as height, width and the embeddings
        are determined. The partial image is not saved if `crop_image = 'False'` is set.

        Args:
            annotation_id: An annotation id of the image annotations.
            crop_image: Whether to store the cropped image as `np.array`.
        """

        ann = self.get_annotation(annotation_ids=annotation_id)[0]

        new_image = Image(file_name=self.file_name, location=self.location, external_id=annotation_id)

        if self._bbox is None or ann.bounding_box is None:
            raise ImageError(f"Bounding box for image and ImageAnnotation ({annotation_id}) must be set")

        new_bounding_box = intersection_box(self._bbox, ann.bounding_box, self.width, self.height)
        if new_bounding_box.absolute_coords:
            width = new_bounding_box.width
            height = new_bounding_box.height
        else:
            width = new_bounding_box.width * self.width
            height = new_bounding_box.height * self.height
        new_image.set_width_height(width, height)
        new_image.set_embedding(self.image_id, bounding_box=new_bounding_box)

        if crop_image and self.image is not None:
            new_image.image = crop_box_from_image(self.image, ann.bounding_box, self.width, self.height)
        elif crop_image and self.image is None:
            raise ImageError("crop_image = True requires self.image to be not None")

        ann.image = new_image

    def maybe_ann_to_sub_image(self, annotation_id: str, category_names: Union[str, list[str]]) -> None:
        """
        Provides a supplement to `image_ann_to_image` and mainly operates on the `ImageAnnotation.image` of
        the image annotation. The aim is to assign image annotations from this image one hierarchy level lower to the
        image of the image annotation. All annotations of this image are also dumped onto the image of the image
        annotation, provided that their bounding boxes are completely in the box of the annotation under consideration.

        Args:
            annotation_id: image annotation you want to assign image annotation from this image. Note, that the
                           annotation must have a not None `image`.
            category_names: Filter the proposals of all image categories of this image by some given category names.
        """

        ann = self.get_annotation(annotation_ids=annotation_id)[0]
        if ann.image is None:
            raise ImageError("When adding sub images to ImageAnnotation then ImageAnnotation.image must not be None")
        box = ann.get_bounding_box(self.image_id).to_list("xyxy")
        proposals = self.get_annotation(category_names)
        points = np.array([prop.get_bounding_box(self.image_id).center for prop in proposals])
        if not points.size:
            return
        ann_ids = np.array([prop.annotation_id for prop in proposals])
        indices = np.where(
            (box[0] < points[:, 0]) & (box[1] < points[:, 1]) & (box[2] > points[:, 0]) & (box[3] > points[:, 1])
        )[0]
        selected_ids = ann_ids[indices]
        sub_images = self.get_annotation(annotation_ids=selected_ids.tolist())
        ann_box = ann.get_bounding_box(self.image_id)
        if not ann_box.absolute_coords:
            ann_box = ann_box.transform(self.width, self.height, absolute_coords=True)
        for sub_image in sub_images:
            if sub_image.image is None:
                raise ImageError(
                    "When setting an embedding to ImageAnnotation then ImageAnnotation.image must not be None"
                )
            sub_image_box = sub_image.get_bounding_box(self.image_id)
            if not sub_image_box.absolute_coords:
                sub_image_box = sub_image_box.transform(self.width, self.height, absolute_coords=True)
            sub_image.image.set_embedding(
                annotation_id,
                global_to_local_coords(sub_image_box, ann_box),
            )
            ann.image.dump(sub_image)

    def remove_image_from_lower_hierarchy(self, pixel_values_only: bool = False) -> None:
        """Will remove all images from image annotations."""
        for ann in self.annotations:
            if pixel_values_only:
                if ann.image is not None:
                    ann.image.clear_image()
            else:
                absolute_bounding_box = ann.get_bounding_box(self.image_id)
                ann.bounding_box = absolute_bounding_box
                ann.image = None

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> Image:
        """
        Create `Image` instance from dict.

        Args:
            kwargs: dict with  `Image` attributes and nested dicts for initializing annotations,

        Returns:
            Initialized image
        """
        image = cls(kwargs.get("file_name"), kwargs.get("location"), kwargs.get("external_id"))
        image._image_id = kwargs.get("_image_id")
        _image = kwargs.get("_image")
        image.page_number = int(kwargs.get("page_number", 0))
        image.document_id = kwargs.get("document_id", image._image_id)
        if _image is not None:
            image.image = _image
        if box_kwargs := kwargs.get("_bbox"):
            image._bbox = BoundingBox.from_dict(**box_kwargs)
        for image_id, box_dict in kwargs.get("embeddings").items():
            image.set_embedding(image_id, BoundingBox.from_dict(**box_dict))
        for ann_dict in kwargs.get("annotations"):
            image_ann = ImageAnnotation.from_dict(**ann_dict)
            if "image" in ann_dict:
                image_dict = ann_dict["image"]
                if image_dict:
                    image_ann.image = cls.from_dict(**image_dict)
            image.dump(image_ann)
        if summary_dict := kwargs.get("_summary", kwargs.get("summary")):
            image.summary = CategoryAnnotation.from_dict(**summary_dict)
            image.summary.category_name = SummaryType.SUMMARY

        return image

    @classmethod
    @no_type_check
    def from_file(cls, file_path: str) -> Image:
        """
        Create `Image` instance from `.json` file.

        Args:
            file_path: file_path

        Returns:
            Initialized image
        """
        with open(file_path, "r", encoding="UTF-8") as file:
            image = Image.from_dict(**json.load(file))
        return image

    @staticmethod
    def get_state_attributes() -> list[str]:
        """
        Returns the list of attributes that define the `state_id` of an image.

        Returns:
            list of attributes
        """
        return ["annotations", "embeddings", "_image", "_summary"]

    @property
    def state_id(self) -> str:
        """
        Different to `image_id` this id does depend on every state attributes and might therefore change
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
                        container_ids.append(str(value))
            elif isinstance(attr, list):
                for element in attr:
                    if isinstance(element, Annotation):
                        container_ids.append(element.state_id)
                    elif isinstance(element, str):
                        container_ids.append(element)
                    else:
                        container_ids.append(str(element))
            elif isinstance(attr, np.ndarray):
                container_ids.append(convert_np_array_to_b64(attr))
            else:
                container_ids.append(str(attr))
        return get_uuid(self.image_id, *container_ids)

    def save(
        self,
        image_to_json: bool = True,
        highest_hierarchy_only: bool = False,
        path: Optional[PathLikeOrStr] = None,
        dry: bool = False,
    ) -> Optional[Union[ImageDict, str]]:
        """
        Export image as dictionary. As `np.array` cannot be serialized `image` values will be converted into
        `base64` encodings.

        Args:
            image_to_json: If `True` will save the image as b64 encoded string in output
            highest_hierarchy_only: If True it will remove all image attributes of ImageAnnotations
            path: Path to save the .json file to. If `None` results will be saved in the folder of the original
                  document.
            dry: Will run dry, i.e. without saving anything but returning the `dict`

        :return: optional dict
        """

        def set_image_keys_to_none(d):  # type: ignore
            if isinstance(d, dict):
                for key, value in d.items():
                    if key == "_image":
                        d[key] = None
                    else:
                        set_image_keys_to_none(value)
            elif isinstance(d, list):
                for item in d:
                    set_image_keys_to_none(item)

        if path is None:
            path = Path(self.location)
        path = Path(path)
        if path.is_dir():
            path = path / self.image_id
        suffix = path.suffix
        if suffix:
            path_json = fspath(path).replace(suffix, ".json")
        else:
            path_json = fspath(path) + ".json"
        if highest_hierarchy_only:
            self.remove_image_from_lower_hierarchy()
        else:
            self.remove_image_from_lower_hierarchy(pixel_values_only=True)
        export_dict = self.as_dict()
        export_dict["location"] = fspath(export_dict["location"])
        if not image_to_json:
            set_image_keys_to_none(export_dict)
        if dry:
            return export_dict
        with open(path_json, "w", encoding="UTF-8") as file:
            json.dump(export_dict, file, indent=2)
        return path_json

    def get_categories_from_current_state(self) -> set[str]:
        """
        Returns:
            All active dumped categories
        """
        return {ann.category_name for ann in self.get_annotation()}

    def get_service_id_to_annotation_id(self) -> defaultdict[str, list[str]]:
        """
        Returns:
            A dictionary with `service_id`s as keys and lists of annotation ids that have been generated by the
            service
        """
        service_id_dict = defaultdict(list)
        for ann in self.get_annotation():
            if ann.service_id:
                service_id_dict[ann.service_id].append(ann.annotation_id)
            for sub_cat_key in ann.sub_categories:
                sub_cat = ann.get_sub_category(sub_cat_key)
                if sub_cat.service_id:
                    service_id_dict[sub_cat.service_id].append(sub_cat.annotation_id)
            if ann.image is not None:
                for summary_cat_key in ann.image.summary.sub_categories:
                    summary_cat = ann.get_summary(summary_cat_key)
                    if summary_cat.service_id:
                        service_id_dict[summary_cat.service_id].append(summary_cat.annotation_id)

        return service_id_dict

    def get_annotation_id_to_annotation_maps(self) -> defaultdict[str, list[AnnotationMap]]:
        """
        Returns a dictionary with annotation ids as keys and lists of `AnnotationMap` as values. The range of ids
        is the union of all `ImageAnnotation`, `CategoryAnnotation` and `ContainerAnnotation` of the image.

        Returns:
            `defaultdict` with `annotation_id`s as keys and lists of `AnnotationMap` as values
        """
        all_ann_id_dict = defaultdict(list)
        for ann in self.get_annotation():
            ann_id_dict = ann.get_annotation_map()
            for key, val in ann_id_dict.items():
                all_ann_id_dict[key].extend(val)

        return all_ann_id_dict
