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
Dataclass Image
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, no_type_check

import numpy as np
from numpy import uint8

from ..utils.detection_types import ImageType
from ..utils.identifier import get_uuid, is_uuid_like
from ..utils.settings import ObjectTypes, get_type
from .annotation import Annotation, BoundingBox, ImageAnnotation, SummaryAnnotation
from .box import crop_box_from_image, global_to_local_coords, intersection_box
from .convert import as_dict, convert_b64_to_np_array, convert_np_array_to_b64, convert_pdf_bytes_to_np_array_v2


@dataclass
class Image:
    """
    The image object is the enclosing data class that is used in the core data model to manage, retrieve or store
    all information during processing. It contains metadata belonging to the image, but also the image itself
    and annotations, either given as ground truth or determined via a processing path. In addition, there are
    storage options in order not to recalculate coordinates in relation to other images.

    Data points from datasets must be mapped in this format so that the processing tools (pipeline components) can
    be called up without further adjustment.

    In the case of full pipelines, the image data model is also the highest hierarchy class in which document pages
    including their discovered features can be processed.

    An image can be provided with an image_id by providing the external_id, which can be clearly identified
    as a md5 hash string. If such an id is not given, an image_id is derived from the file_name and, if necessary,
    from the given location.

    When initializing the object, the following arguments can be specified:

    :attr:`file_name`: Should be equal to the name of a physical file representing the image. If the image is part
    of a larger document (e.g. pdf-document) the file_name should be populated as a concatenation of the document file
    and its page number.

    :attr:`location`: Full path to the document or to the physical file. Loading functions from disk use this attribute.

    :attr:`external_id`: A string or integer value for generating an image id.

    All other attributes represent containers (lists or dicts) that can be populated and managed using their own method.

    In :attr:`image`, the image may be saved as np.array. Allocation as base64 encoding string or as pdf bytes are
    possible and are converted via a :meth:`image.setter`. Other formats are rejected. As a result of the transfer,
    the width and height of the image are determined. These are accessible via :attr:`width` or :attr:`height`.
    Using :attr:`embeddings`, various bounding boxes can be saved that describe the position of the image as a
    sub-image of another image. The bounding box is accessed in relation to the embedding image via the annotation_id.
    Embeddings are often used in connection with annotations in which :attr:`image` is populated.

    All ImageAnnotations associated with the image are used in the list annotations. Other types of annotation are
    not permitted and must either be transported as  sub-category of an ImageAnnotation or placed as a summary
    annotation in the :attr:`summary`.
    """

    file_name: str
    location: str = field(default="")
    external_id: Optional[Union[str, int]] = field(default=None, repr=False)
    _image_id: Optional[str] = field(default=None, init=False, repr=True)
    _image: Optional[ImageType] = field(default=None, init=False, repr=False)
    _bbox: Optional[BoundingBox] = field(default=None, init=False, repr=False)
    embeddings: Dict[str, BoundingBox] = field(default_factory=dict, init=False, repr=True)
    annotations: List[ImageAnnotation] = field(default_factory=list, init=False, repr=True)
    _annotation_ids: List[str] = field(default_factory=list, init=False, repr=False)
    _summary: Optional[SummaryAnnotation] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.external_id is not None:
            external_id = str(self.external_id)
            if is_uuid_like(external_id):
                self.image_id = external_id
            else:
                self.image_id = get_uuid(external_id)
        else:
            self.image_id = get_uuid(str(self.location), self.file_name)

    @property
    def image_id(self) -> str:
        """
        image_id
        """
        if self._image_id is not None:
            return self._image_id
        raise ValueError("image_id not set")

    @image_id.setter
    def image_id(self, input_id: str) -> None:
        """
        image_id setter
        """
        if self._image_id is not None:
            raise ValueError("image_id already defined and cannot be reset")
        if is_uuid_like(input_id):
            self._image_id = input_id
        elif isinstance(input_id, property):
            pass
        else:
            raise ValueError("image_id must be uuid3 string")

    @property
    def image(self) -> Optional[ImageType]:
        """
        image
        """
        return self._image

    @image.setter
    def image(self, image: Optional[Union[str, ImageType, bytes]]) -> None:
        """
        Sets the image for internal storage. Will convert to numpy array before storing internally.
        Note: If the input is an np.array, ensure that the image is in BGR-format as this is the standard
        format for the whole package.
        :param image: Accepts numpy arrays, base64 encodings or bytes generated from pdf documents.
                      Everything else will be rejected.
        """

        if isinstance(image, property):
            pass
        elif isinstance(image, str):
            self._image = convert_b64_to_np_array(image)
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()
        elif isinstance(image, bytes):
            self._image = convert_pdf_bytes_to_np_array_v2(image, dpi=300)
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()
        else:
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Cannot load image is of type: {type(image)}")
            self._image = image.astype(uint8)
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()

    @property
    def summary(self) -> Optional[SummaryAnnotation]:
        """summary"""
        return self._summary

    @summary.setter
    def summary(self, summary_annotation: SummaryAnnotation) -> None:
        """summary setter"""
        if summary_annotation._annotation_id is None:  # pylint: disable=W0212
            summary_annotation.annotation_id = self.define_annotation_id(summary_annotation)
        self._summary = summary_annotation

    @property
    def pdf_bytes(self) -> Optional[bytes]:
        """
        pdf_bytes. This attribute will be set dynamically and is not part of the core Image data model
        """
        if hasattr(self, "_pdf_bytes"):
            return getattr(self, "_pdf_bytes")
        return None

    @pdf_bytes.setter
    def pdf_bytes(self, pdf_bytes: bytes) -> None:
        """
        pdf_bytes setter
        """
        assert isinstance(pdf_bytes, bytes)
        if not hasattr(self, "_pdf_bytes"):
            setattr(self, "_pdf_bytes", pdf_bytes)

    def clear_image(self, clear_bbox: bool = False) -> None:
        """
        Removes the :attr:`Image.image`. Useful, if the image must be a lightweight object.

        :param clear_bbox: If set to `True` it will remove the image width and height. This is necessary,
                           if the image is going to be replaced with a transform. It will also remove the self
                           embedding entry
        """
        self._image = None
        if clear_bbox:
            self._bbox = None
            self.embeddings.pop(self.image_id)

    def get_image(self) -> "_Img":  # type: ignore
        """
        Get the image either in base64 string representation or as np.array.

        .. code-block:: python

            image.get_image().to_np_array()

        or

        .. code-block:: python

            image.get_image().to_b64()

        :return: desired image encoding representation
        """

        class _Img:
            """
            Helper class. Do not use it in your code.
            """

            def __init__(self, img: Optional[ImageType]):
                self.img = img

            def to_np_array(self) -> Optional[ImageType]:
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
        width
        """
        if self._bbox is None:
            raise ValueError("Width not available. Call set_width_height first")
        return self._bbox.width

    @property
    def height(self) -> float:
        """
        height
        """
        if self._bbox is None:
            raise ValueError("Height not available. Call set_width_height first")
        return self._bbox.height

    def set_width_height(self, width: float, height: float) -> None:
        """
        Defines bounding box of the image if not already set. Use this, if you do not want to keep the image separated
        for memory reasons.

        :param width: width of image
        :param height: height of image
        """
        if self._bbox is None:
            self._bbox = BoundingBox(ulx=0.0, uly=0.0, height=height, width=width, absolute_coords=True)
            self._self_embedding()

    def set_embedding(self, image_id: str, bounding_box: BoundingBox) -> None:
        """
        Set embedding pair. Pass an image_id and a bounding box defining the spacial position of this image with
        respect to the embedding image.

        :param image_id: A uuid of the embedding image.
        :param bounding_box: bounding box of this image in terms of the embedding image.
        """
        if not isinstance(bounding_box, BoundingBox):
            raise TypeError(f"Bounding box must be of type BoundingBox, is of type {type(bounding_box)}")
        self.embeddings[image_id] = bounding_box

    def get_embedding(self, image_id: str) -> BoundingBox:
        """
        Returns the bounding box according to the image_id.

        :param image_id: uuid string of the embedding image
        :return: The bounding box of this instance in terms of the embedding image
        """

        return self.embeddings[image_id]

    def _self_embedding(self) -> None:
        if self._bbox is not None:
            self.set_embedding(self.image_id, self._bbox)

    def dump(self, annotation: ImageAnnotation) -> None:
        """
        Dump an annotation to the Image dataclass. This is the central method for associating an annotation with
        an image. It gives the annotation an annotation_id in relation to the image_id in order to ensure uniqueness
        across all images.

        :param annotation: image annotation to store
        """
        if not isinstance(annotation, ImageAnnotation):
            raise TypeError(
                f"Annotation must be of type ImageAnnotation: "
                f"{annotation.annotation_id} but is of type {str(type(annotation))}"
            )
        if annotation._annotation_id is None:  # pylint: disable=W0212
            annotation.annotation_id = self.define_annotation_id(annotation)
        if annotation.annotation_id in self._annotation_ids:
            raise ValueError(f"Cannot dump annotation with already taken " f"id {annotation.annotation_id}")
        self._annotation_ids.append(annotation.annotation_id)
        self.annotations.append(annotation)

    def get_annotation(
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        annotation_types: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[ImageAnnotation]:
        """
        Selection of annotations from the annotation container. Filter conditions can be defined by specifying
        the annotation_id or the category name. (Since only image annotations are currently allowed in the container,
        annotation_type is a redundant filter condition.) Only annotations that have :attr: active = 'True' are
        returned. If more than one condition is provided, only annotations will be returned that satisfy all conditions.
        If no condition is provided, it will return all active annotations.

        :param category_names: A single name or list of names
        :param annotation_ids: A single id or list of ids
        :param annotation_types: A type name or list of type names.
        :return: A (possibly empty) list of Annotations
        """

        cat_names = [category_names] if isinstance(category_names, (ObjectTypes, str)) else category_names
        if cat_names is not None:
            cat_names = [get_type(cat_name) for cat_name in cat_names]
        ann_ids = [annotation_ids] if isinstance(annotation_ids, str) else annotation_ids
        ann_types = [annotation_types] if isinstance(annotation_types, str) else annotation_types

        anns = filter(lambda x: x.active, self.annotations)

        if ann_types is not None:
            for type_name in ann_types:
                anns = filter(lambda x: isinstance(x, eval(type_name)), anns)  # pylint: disable=W0123, W0640

        if cat_names is not None:
            anns = filter(lambda x: x.category_name in cat_names, anns)  # type:ignore

        if ann_ids is not None:
            anns = filter(lambda x: x.annotation_id in ann_ids, anns)  # type:ignore

        return list(anns)

    def get_annotation_iter(
        self,
        category_names: Optional[Union[str, ObjectTypes, Sequence[Union[str, ObjectTypes]]]] = None,
        annotation_ids: Optional[Union[str, Sequence[str]]] = None,
        annotation_types: Optional[Union[str, Sequence[str]]] = None,
    ) -> Iterable[ImageAnnotation]:
        """
        Get annotation as an iterator. Same as :meth:`get_annotation` but returns an iterator instead of a list.

        :param category_names: A single name or list of names
        :param annotation_ids: A single id or list of ids
        :param annotation_types: A type name or list of type names.

        :return: A (possibly empty) list of annotations
        """

        return iter(self.get_annotation(category_names, annotation_ids, annotation_types))

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the full image dataclass as dict. Uses the custom :func:`convert.as_dict` to disregard attributes
        defined by :meth:`remove_keys`.

        :return:  A custom dict.
        """

        img_dict = as_dict(self, dict_factory=dict)
        if self.image is not None:
            img_dict["_image"] = convert_np_array_to_b64(self.image)
        else:
            img_dict["_image"] = None
        return img_dict

    @staticmethod
    def remove_keys() -> List[str]:
        """
        A list of attributes to suspend from as_dict creation.
        """

        return []

    def define_annotation_id(self, annotation: Annotation) -> str:
        """
        Generate a uuid for a given annotation. To guarantee uniqueness the generation depends on the datapoint
        image_id as well as on the annotation.

        :param annotation:  An annotation to generate the uuid for
        :return: uuid string
        """

        attributes = annotation.get_defining_attributes()
        attributes_values = [str(getattr(annotation, attribute)) for attribute in attributes]
        return get_uuid(*attributes_values, str(self.image_id))

    def remove(self, annotation: ImageAnnotation) -> None:
        """
        Instead of removing consider deactivating annotations.

        Calls :meth:`List.remove`. Make sure, the element is in the list for otherwise a ValueError will be raised.

        :param annotation: The annotation to remove
        """

        self.annotations.remove(annotation)
        self._annotation_ids.remove(annotation.annotation_id)

    def image_ann_to_image(self, annotation_id: str, crop_image: bool = False) -> None:
        """
        This method is an operation that changes the state of an underlying dumped image annotation and that
        manages :attr:`ImageAnnotation.image`. An image object is generated, which is interpreted as part of the image
        by the bounding box. The image is cut out and the determinable fields such as height, width and the embeddings
        are determined. The partial image is not saved if crop_image = 'False' is set.

        :param annotation_id: An annotation id of the image annotations.
        :param crop_image: Whether to store the cropped image as np.array.
        """

        ann = self.get_annotation(annotation_ids=annotation_id)[0]

        new_image = Image(file_name=self.file_name, location=self.location, external_id=annotation_id)

        if self._bbox is None or ann.bounding_box is None:
            raise ValueError(f"Bounding box for image and ImageAnnotation ({annotation_id}) must be set")

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
            raise ValueError("crop_image = True requires self.image to be not None")

        ann.image = new_image

    def maybe_ann_to_sub_image(self, annotation_id: str, category_names: Union[str, List[str]]) -> None:
        """
        Provides a supplement to :meth:`image_ann_to_image` and mainly operates on the: attr:`ImageAnnotation.image` of
        the image annotation. The aim is to assign image annotations from this image one hierarchy level lower to the
        image of the image annotation. All annotations of this image are also dumped onto the image of the image
        annotation, provided that their bounding boxes are completely in the box of the annotation under consideration.

        :param annotation_id: image annotation you want to assign image annotation from this image. Note, that the
                              annotation must have a not None :attr:`image`.
        :param category_names: Filter the proposals of all image categories of this image by some given category names.
        """

        ann = self.get_annotation(annotation_ids=annotation_id)[0]
        if ann.image is None:
            raise ValueError("When adding sub images to ImageAnnotation then ImageAnnotation.image must not be None")
        assert ann.bounding_box is not None
        box = ann.bounding_box.to_list("xyxy")
        proposals = self.get_annotation(category_names)
        points = np.array(
            [prop.image.get_embedding(self.image_id).center for prop in proposals if prop.image is not None]
        )
        ann_ids = np.array([prop.annotation_id for prop in proposals])
        indices = np.where(
            (box[0] < points[:, 0]) & (box[1] < points[:, 1]) & (box[2] > points[:, 0]) & (box[3] > points[:, 1])
        )[0]
        selected_ids = ann_ids[indices]
        sub_images = self.get_annotation(annotation_ids=selected_ids.tolist())
        for sub_image in sub_images:
            if sub_image.image is None:
                raise ValueError(
                    "When setting an embedding to ImageAnnotation then ImageAnnotation.image must not " "be None"
                )
            sub_image.image.set_embedding(
                annotation_id,
                global_to_local_coords(
                    sub_image.image.get_embedding(self.image_id), ann.image.get_embedding(self.image_id)
                ),
            )
            ann.image.dump(sub_image)

    def remove_image_from_lower_hierachy(self) -> None:
        """Will remove all images from image annotations."""
        for ann in self.annotations:
            if ann.image is not None:
                absolute_bounding_box = ann.image.get_embedding(self.image_id)
                ann.bounding_box = absolute_bounding_box
                ann.image = None

    @classmethod
    @no_type_check
    def from_dict(cls, **kwargs) -> "Image":
        """
        Create :class:`Image` instance from dict.

        :param kwargs: dict with  :class:`Image` attributes and nested dicts for initializing annotations,
        :return: Initialized image
        """
        image = cls(kwargs.get("file_name"), kwargs.get("location"), kwargs.get("external_id"))
        image._image_id = kwargs.get("_image_id")
        _image = kwargs.get("_image")
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
            image.summary = SummaryAnnotation.from_dict(**summary_dict)
        return image
