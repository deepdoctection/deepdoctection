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
from typing import Iterable

import numpy as np

from ..utils.detection_types import ImageType
from .annotation import *  # pylint: disable=W0401, W0614
from .box import crop_box_from_image, global_to_local_coords, intersection_box
from .convert import convert_b64_to_np_array, convert_np_array_to_b64, convert_pdf_bytes_to_np_array_v2


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
    location: str = field(default_factory=str)
    external_id: Optional[Union[str, int]] = field(default=None, init=True, repr=False)
    image_id: str = field(init=False, repr=True)
    _image_id: Optional[str] = field(default=None, init=False, repr=False)
    image: Optional[ImageType] = field(init=False, repr=False)
    _image: Optional[ImageType] = field(default=None, init=False, repr=False)
    _bbox: Optional[BoundingBox] = field(default=None, init=False, repr=False)
    _embeddings: Dict[str, BoundingBox] = field(default_factory=dict, init=False, repr=False)
    embeddings: Dict[str, BoundingBox] = field(default_factory=dict, init=False, repr=True)
    annotations: List[ImageAnnotation] = field(default_factory=list, init=False, repr=True)
    _annotations: List[ImageAnnotation] = field(default_factory=list, init=False, repr=False)
    _annotation_ids: List[str] = field(default_factory=list, init=False, repr=False)
    summary: Optional[SummaryAnnotation] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.external_id is not None:
            external_id = str(self.external_id)
            if is_uuid_like(external_id):
                self.image_id = external_id
            else:
                self.image_id = get_uuid(external_id)
        else:
            self.image_id = get_uuid(self.location, self.file_name)

    @property  # type: ignore
    def image_id(self) -> str:  # pylint: disable=E0102
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
            raise AssertionError("image_id already defined and cannot be reset")
        if is_uuid_like(input_id):
            self._image_id = input_id
        elif isinstance(input_id, property):
            pass
        else:
            raise ValueError("image_id must be uuid3 string")

    @property  # type: ignore
    def image(self) -> Optional[ImageType]:  # pylint: disable=E0102
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
            assert isinstance(image, np.ndarray), f"cannot load image is of type: {type(image)}"
            self._image = image
            self.set_width_height(self._image.shape[1], self._image.shape[0])
            self._self_embedding()

    @property  # type: ignore
    def embeddings(self) -> Dict[str, BoundingBox]:  # pylint: disable=E0102
        """
        embeddings. Rather use :meth:`get_embedding`
        """
        return self._embeddings

    @embeddings.setter
    def embeddings(self, inputs: Dict[str, BoundingBox]) -> None:
        """
        embedding setter. Only defined for technical reasons. Cannot change the underlying attribute from here.
        """

    @property  # type: ignore
    def annotations(self) -> List[ImageAnnotation]:  # pylint: disable=E0102
        """
        annotations. Rather use :meth:`get_annotations` directly or with some specified filter conditions.
        """
        return self._annotations

    @annotations.setter
    def annotations(self, inputs: List[ImageAnnotation]) -> None:
        """
        annotations setter. Only defined for technical reasons. Cannot change the underlying attribute from here.
        """

    @property
    def pdf_bytes(self) -> Optional[bytes]:
        """
        pdf_bytes. This attribute will be set dynamically and is not part of the core Image data model
        """
        if hasattr(self, "_pdf_bytes"):
            return self._pdf_bytes  # type: ignore
        return None

    @pdf_bytes.setter
    def pdf_bytes(self, pdf_bytes: bytes) -> None:
        """
        pdf_bytes setter
        """
        assert isinstance(pdf_bytes, bytes)
        if not hasattr(self, "_pdf_bytes"):
            setattr(self, "_pdf_bytes", pdf_bytes)

    def clear_image(self) -> None:
        """
        Removes the :attr:`Image.image`. Useful, if the image must be a lightweight object.
        """
        self._image = None

    def get_image(self, type_id: str) -> Optional[Union[ImageType, str]]:
        """
        Get the image either in base64 string representation or as np.array.

        :param type_id: "b64" or "np"
        :return: desired image encoding representation
        """
        assert type_id in ["b64", "np"], type_id
        if type_id == "b64" and self.image is not None:
            return convert_np_array_to_b64(self.image)
        return self.image

    @property
    def width(self) -> float:  # pylint: disable=R1710
        """
        width
        """
        if self._bbox is None:
            raise ValueError("Width not available. Call set_width_height first")
        return self._bbox.width

    @property
    def height(self) -> float:  # pylint: disable=R1710
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
        assert isinstance(bounding_box, BoundingBox), "bounding box must be of type BoundingBox"
        self._embeddings[image_id] = bounding_box

    def get_embedding(self, image_id: str) -> BoundingBox:
        """
        Returns the bounding box according to the image_id.

        :param image_id: uuid string of the embedding image
        :return: The bounding box of this instance in terms of the embedding image
        """

        return self._embeddings[image_id]

    def _self_embedding(self) -> None:
        if self._bbox is not None:
            assert isinstance(self.image_id, str)
            self.set_embedding(self.image_id, self._bbox)

    def dump(self, annotation: ImageAnnotation) -> None:
        """
        Dump an annotation to the Image dataclass. This is the central method for associating an annotation with
        an image. It gives the annotation an annotation_id in relation to the image_id in order to ensure uniqueness
        across all images.

        :param annotation: image annotation to store
        """
        assert isinstance(annotation, ImageAnnotation), (
            f"Annotation must be of type ImageAnnotation: "
            f"{annotation.annotation_id} but is of type {str(type(annotation))}"
        )
        if annotation._annotation_id is None:  # pylint: disable=W0212
            annotation.annotation_id = self._define_annotation_id(annotation)
        assert annotation.annotation_id not in self._annotation_ids, (
            f"Cannot dump annotation with already taken " f"id {annotation.annotation_id}"
        )
        self._annotation_ids.append(annotation.annotation_id)
        self._annotations.append(annotation)

    def get_annotation(
        self,
        category_names: Optional[Union[str, List[str]]] = None,
        annotation_ids: Optional[Union[str, List[str]]] = None,
        annotation_types: Optional[Union[str, List[str]]] = None,
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

        cat_names = [category_names] if isinstance(category_names, str) else category_names
        ann_ids = [annotation_ids] if isinstance(annotation_ids, str) else annotation_ids
        ann_types = [annotation_types] if isinstance(annotation_types, str) else annotation_types

        anns = filter(lambda x: x.active, self._annotations)

        if ann_types is not None:
            for type_name in ann_types:
                anns = filter(lambda x: isinstance(x, eval(type_name)), anns)  # pylint: disable=W0123, W0640

        if cat_names is not None:
            anns = filter(lambda x: x.category_name in cat_names, anns)  # type: ignore

        if ann_ids is not None:
            anns = filter(lambda x: x.annotation_id in annotation_ids, anns)  # type: ignore

        return list(anns)

    def get_annotation_iter(
        self,
        category_names: Optional[Union[str, List[str]]] = None,
        annotation_ids: Optional[Union[str, List[str]]] = None,
        annotation_types: Optional[Union[str, List[str]]] = None,
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
        return img_dict

    @staticmethod
    def remove_keys() -> List[str]:
        """
        A list of attributes to suspend from as_dict creation.
        """

        return ["external_id", "_image_id", "_image", "_bbox", "_embeddings", "_annotations"]

    def _define_annotation_id(self, annotation: Annotation) -> str:
        attributes = annotation.get_defining_attributes()
        attributes_values = [str(getattr(annotation, attribute)) for attribute in attributes]
        assert self.image_id is not None
        return get_uuid(*attributes_values, str(self.image_id))

    def remove(self, annotation: ImageAnnotation) -> None:
        """
        Calls :meth:`List.remove`. Make sure, the element is in the list for otherwise a ValueError will be raised.

        :param annotation: The annotation to remove
        """

        self._annotations.remove(annotation)

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
        assert isinstance(
            ann, ImageAnnotation
        ), f"Annotation must be of type ImageAnnotation: {annotation_id} but is of type {str(type(ann))}"

        new_image = Image(file_name=self.file_name, location=self.location, external_id=annotation_id)

        assert self._bbox is not None and ann.bounding_box is not None
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
            raise ValueError("crop_image = 'True' requires self.image to be not None")

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
        assert ann.image is not None, "adding sub images to annotations requires ann to have an image"
        assert ann.bounding_box is not None
        box = ann.bounding_box.to_list("xyxy")
        proposals = self.get_annotation(category_names)
        points = np.array([prop.image.get_embedding(self.image_id).center for prop in proposals])  # type: ignore
        ann_ids = np.array([prop.annotation_id for prop in proposals])
        indices = np.where(
            (box[0] < points[:, 0]) & (box[1] < points[:, 1]) & (box[2] > points[:, 0]) & (box[3] > points[:, 1])
        )[0]
        selected_ids = ann_ids[indices]
        sub_images = self.get_annotation(annotation_ids=selected_ids.tolist())
        for sub_image in sub_images:
            assert sub_image.image is not None
            sub_image.image.set_embedding(
                annotation_id,
                global_to_local_coords(
                    sub_image.image.get_embedding(self.image_id), ann.image.get_embedding(self.image_id)
                ),
            )
            ann.image.dump(sub_image)

    def get_export(self) -> Dict[str, Any]:
        """
        Exporting image as dictionary. As numpy array cannot be serialized :attr:`image` values will be converted into
        base64 encodings.

        :return: Dict that e.g. can be saved to a file.
        """

        image_copy = deepcopy(self)
        image_copy._annotations = []  # pylint: disable=W0212
        image_copy.image = np.ones((1, 1, 3), dtype=np.float32)
        export_dict = image_copy.as_dict()
        export_dict["annotations"] = []
        export_dict["image"] = self.get_image(type_id="b64")
        for ann in self._annotations:
            export_dict["annotations"].append(ann.get_export())

        return export_dict
