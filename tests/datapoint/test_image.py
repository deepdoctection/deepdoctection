# -*- coding: utf-8 -*-
# File: test_image.py

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
Testing the module datapoint.image
"""

from typing import Union

from numpy import float32, ones
from numpy.testing import assert_array_equal
from pytest import mark, raises

from deepdoctection.datapoint import BoundingBox, CategoryAnnotation, Image, ImageAnnotation
from deepdoctection.utils import get_uuid

from ..test_utils import anns_to_ids
from .conftest import TestPdfPage, WhiteImage  # pylint: disable=W0611


class TestImage:
    """
    Testing Image methods
    """

    @staticmethod
    @mark.parametrize(
        "location, file_name, external_id, expected",
        [
            (WhiteImage.loc, WhiteImage.file_name, None, WhiteImage.get_image_id("d")),
            (
                WhiteImage.loc,
                WhiteImage.file_name,
                WhiteImage.external_id,
                WhiteImage.get_image_id("n"),
            ),
            (WhiteImage.loc, WhiteImage.file_name, WhiteImage.uuid, WhiteImage.get_image_id("u")),
        ],
    )
    def test_image_id_is_correctly_assigned(
        location: str, file_name: str, external_id: Union[str, None], expected: str
    ) -> None:
        """
        Image_id is assigned as expected, no matter what external_id is provided
        :param location: loc
        :param file_name: file name
        :param external_id: external id
        :param expected: image_id
        :return:
        """
        # Arrange
        test_image = Image(file_name=file_name, location=location, external_id=external_id)

        # Assert
        assert test_image.image_id == expected

    @staticmethod
    def test_image_id_cannot_be_reassigned(image: WhiteImage) -> None:
        """
        Once image_id is assigned, it cannot be reassigned
        :param image: test image
        """
        # Arrange
        test_image = Image(file_name=image.file_name, location=image.loc, external_id=image.external_id)

        # Act and assert
        with raises(AssertionError):
            test_image.image_id = "ec2aac06-c261-3669-b8bd-4486a54ce740"

    @staticmethod
    def test_image_stores_correct_image_and_meta(image: WhiteImage) -> None:
        """
        Image stores image given as np.array correctly. It returns height as width as expected
        :param image: Test image
        """
        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)

        # Act
        test_image.image = image.get_image_as_np_array()

        # Assert
        assert_array_equal(test_image.get_image(type_id="np"), image.get_image_as_np_array())  # type: ignore
        assert test_image.height == image.get_bounding_box().height
        assert test_image.width == image.get_bounding_box().width

    @staticmethod
    def test_image_has_its_own_embedding_entry(image: WhiteImage) -> None:
        """
        Image has embedding k/v entry image_id/self._bbox
        :param image: Test image
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)

        # Act
        test_image.image = image.get_image_as_np_array()

        # Assert
        assert test_image.get_embedding(test_image.image_id) == test_image._bbox  # pylint: disable=W0212

    @staticmethod
    def test_image_stores_correct_image_from_pdf(pdf_page: TestPdfPage) -> None:
        """

        :param pdf_page: Test pdf_page, represented in bytes
        """

        # Arrange
        test_image = Image(location=pdf_page.loc, file_name=pdf_page.file_name)

        # Act
        test_image.image = pdf_page.get_image_as_pdf_bytes()  # type: ignore

        # Assert
        assert test_image.height == pdf_page.np_array_shape[0]
        assert test_image.width == pdf_page.np_array_shape[1]

    @staticmethod
    def test_image_returns_image_representation(image: WhiteImage) -> None:
        """
        Image stores and returns image given as b64 string correctly. It returns height as width as expected.
        :param image: Test image
        """
        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        test_image.image = image.get_image_as_b64_string()  # type: ignore

        # Assert
        assert_array_equal(test_image.get_image(type_id="np"), image.get_image_as_np_array())  # type: ignore
        assert test_image.get_image(type_id="b64") == image.get_image_as_b64_string()
        assert test_image.height == image.get_bounding_box().height
        assert test_image.width == image.get_bounding_box().width

    @staticmethod
    def test_dump_cat_and_check_ann_id(image: WhiteImage) -> None:
        """
        Categories are dumped and annotation ids are correctly assigned.
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat = ImageAnnotation(
            category_name="FOO",
            category_id="1",
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )
        sub_cat_1 = CategoryAnnotation(category_name="BAK", category_id="2")

        # Act
        cat.dump_sub_category("BAK", sub_cat_1, test_image.image_id)
        test_image.dump(cat)

        # Assert
        assert test_image.image_id == get_uuid(image.loc + image.file_name)
        assert (
            cat.annotation_id == "85444154-9593-345a-8c1d-b6ec0020d63c"
        )  # get_uuid("FOOBounding Box ulx: 1.0 uly: 1.0 lrx: 2.0 lry: 3.090c05f37-a017-39cc-a178-b84f9d14ff48")
        assert sub_cat_1.annotation_id == "e6b7d9a9-c0fe-3e55-9a9a-7dfab2a44e41"

    @staticmethod
    def test_dump_same_annotation_not_possible(image: WhiteImage) -> None:
        """
        Same image annotations cannot be dumped twice
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat = ImageAnnotation(
            category_name="FOO",
            category_id="1",
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )

        # Act and Assert
        test_image.dump(cat)

        with raises(AssertionError):
            test_image.dump(cat)

    @staticmethod
    def test_get_annotation(image: WhiteImage) -> None:
        """
        Annotations are returned by conditions.
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat_1 = ImageAnnotation(
            category_name="FOO",
            category_id="1",
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )
        cat_2 = ImageAnnotation(
            category_name="BAK",
            category_id="2",
            bounding_box=BoundingBox(ulx=2.0, uly=2.0, width=2.0, height=2.0, absolute_coords=True),
        )
        cat_3 = ImageAnnotation(
            category_name="BAK",
            category_id="1",
            bounding_box=BoundingBox(ulx=1.5, uly=2.4, width=3.0, height=9.0, absolute_coords=True),
        )

        test_image.dump(cat_1)
        test_image.dump(cat_2)
        test_image.dump(cat_3)

        # Act
        filtered_anns_1 = test_image.get_annotation(category_names="FOO")
        filtered_anns_1_ids = anns_to_ids(filtered_anns_1)
        filtered_anns_2 = test_image.get_annotation(category_names="BLA")
        filtered_anns_2_ids = anns_to_ids(filtered_anns_2)
        filtered_anns_3 = test_image.get_annotation(annotation_ids=[cat_1.annotation_id, cat_3.annotation_id])

        filtered_anns_3_ids = anns_to_ids(filtered_anns_3)
        filtered_anns_4 = test_image.get_annotation(
            annotation_ids=[cat_2.annotation_id, cat_3.annotation_id],
        )
        filtered_anns_4_ids = anns_to_ids(filtered_anns_4)
        filtered_anns_5 = test_image.get_annotation_iter()
        filtered_anns_5_ids = anns_to_ids(filtered_anns_5)

        # Assert
        assert set(filtered_anns_1_ids) == {cat_1.annotation_id}
        assert set(filtered_anns_2_ids) == set()
        assert set(filtered_anns_3_ids) == {cat_1.annotation_id, cat_3.annotation_id}
        assert set(filtered_anns_4_ids) == {cat_2.annotation_id, cat_3.annotation_id}
        assert set(filtered_anns_5_ids) == {cat_1.annotation_id, cat_2.annotation_id, cat_3.annotation_id}

    @staticmethod
    def test_image_ann_to_image(image: WhiteImage) -> None:
        """
        test meth: image_ann_to_image add attr: image to ImageAnnotation and generates Image instance correctly
        """
        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        test_image.image = ones((24, 85, 3), dtype=float32)
        cat_1 = ImageAnnotation(
            category_name="FOO",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        test_image.dump(cat_1)
        annotation_id = cat_1.annotation_id

        # Act
        test_image.image_ann_to_image(annotation_id=annotation_id, crop_image=True)

        # Assert
        assert hasattr(cat_1, "image")
        assert cat_1.image.height == 4  # type: ignore # pylint: disable=E1101
        assert cat_1.image.width == 10  # type: ignore # pylint: disable=E1101
        assert cat_1.image.get_embedding(test_image.image_id) == BoundingBox(  # type: ignore  # pylint: disable=E1101
            ulx=15.0, uly=20.0, width=10.0, height=4.0, absolute_coords=True
        )  # pylint: disable=E1101
        assert cat_1.image.image.shape == (4, 10, 3)  # type: ignore # pylint: disable=E1101

    @staticmethod
    def test_image_with_anns_can_be_exported(image: WhiteImage) -> None:
        """
        test meth: get_export returns a dictionary
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        test_image.image = ones((24, 85, 3), dtype=float32)
        cat_1 = ImageAnnotation(
            category_name="FOO",
            bounding_box=BoundingBox(ulx=15.0, uly=20.0, width=10.0, height=8.0, absolute_coords=True),
        )
        test_image.dump(cat_1)

        # Act
        output = test_image.get_export()

        # Assert
        assert (
            "location" in output
            and "file_name" in output
            and "annotations" in output
            and "image" in output
            and "image_id" in output
            and "embeddings" in output
        )
        assert len(output["annotations"]) == 1

        ann = output["annotations"][0]
        assert (
            "active" in ann
            and "annotation_id" in ann
            and "category_name" in ann
            and "category_id" in ann
            and "score" in ann
        )
