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

from collections import defaultdict
from typing import Union

from numpy import float32, ones
from numpy.testing import assert_array_equal
from pytest import mark, raises

from deepdoctection.dataflow import DataFlow, MapData, SerializerJsonlines
from deepdoctection.datapoint import AnnotationMap, BoundingBox, CategoryAnnotation, Image, ImageAnnotation
from deepdoctection.utils import get_uuid
from deepdoctection.utils.error import ImageError
from deepdoctection.utils.settings import get_type

from ..test_utils import anns_to_ids, collect_datapoint_from_dataflow, get_test_path
from .conftest import TestPdfPage, WhiteImage


class TestImage:
    """
    Testing Image methods
    """

    @staticmethod
    @mark.basic
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
    @mark.basic
    def test_image_id_cannot_be_reassigned(image: WhiteImage) -> None:
        """
        Once image_id is assigned, it cannot be reassigned
        :param image: test image
        """
        # Arrange
        test_image = Image(file_name=image.file_name, location=image.loc, external_id=image.external_id)

        # Act and assert
        with raises(ImageError):
            test_image.image_id = "ec2aac06-c261-3669-b8bd-4486a54ce740"

    @staticmethod
    @mark.basic
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
        assert_array_equal(test_image.get_image().to_np_array(), image.get_image_as_np_array())
        assert test_image.height == image.get_bounding_box().height
        assert test_image.width == image.get_bounding_box().width

    @staticmethod
    @mark.basic
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
    @mark.basic
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
    @mark.basic
    def test_image_returns_image_representation(image: WhiteImage) -> None:
        """
        Image stores and returns image given as b64 string correctly. It returns height as width as expected.
        :param image: Test image
        """
        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        test_image.image = image.get_image_as_b64_string()  # type: ignore

        # Assert
        assert_array_equal(test_image.get_image().to_np_array(), image.get_image_as_np_array())
        assert test_image.get_image().to_b64() == image.get_image_as_b64_string()
        assert test_image.height == image.get_bounding_box().height
        assert test_image.width == image.get_bounding_box().width

    @staticmethod
    @mark.basic
    def test_dump_cat_and_check_ann_id(image: WhiteImage) -> None:
        """
        Categories are dumped and annotation ids are correctly assigned.
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat = ImageAnnotation(
            category_name="FOO",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )
        sub_cat_1 = CategoryAnnotation(category_name="BAK", category_id=2)

        # Act
        cat.dump_sub_category(get_type("BAK"), sub_cat_1, test_image.image_id)
        test_image.dump(cat)

        # Assert
        assert test_image.image_id == get_uuid(image.loc + image.file_name)
        assert (
            cat.annotation_id == "c8c58404-62c3-3e66-b302-ebd3202a778d"
        )  # get_uuid("FOOBounding Box ulx: 1.0 uly: 1.0 lrx: 2.0 lry: 3.090c05f37-a017-39cc-a178-b84f9d14ff48")
        assert sub_cat_1.annotation_id == "7f76354b-1d4c-3874-84c7-a6d0c5701987"

    @staticmethod
    @mark.basic
    def test_dump_same_annotation_not_possible(image: WhiteImage) -> None:
        """
        Same image annotations cannot be dumped twice
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat = ImageAnnotation(
            category_name="FOO",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )

        # Act and Assert
        test_image.dump(cat)

        with raises(ImageError):
            test_image.dump(cat)

    @staticmethod
    @mark.basic
    def test_get_annotation(image: WhiteImage) -> None:
        """
        Annotations are returned by conditions.
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat_1 = ImageAnnotation(
            category_name="FOO",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )
        cat_2 = ImageAnnotation(
            category_name="BAK",
            category_id=2,
            bounding_box=BoundingBox(ulx=2.0, uly=2.0, width=2.0, height=2.0, absolute_coords=True),
        )
        cat_3 = ImageAnnotation(
            category_name="BAK",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.5, uly=2.4, width=3.0, height=9.0, absolute_coords=True),
        )
        cat_4 = ImageAnnotation(
            category_name="BLI",
            category_id=3,
            bounding_box=BoundingBox(ulx=1.5, uly=2.4, width=3.0, height=9.0, absolute_coords=True),
            service_id="test_service",
        )
        cat_5 = ImageAnnotation(
            category_name="BLU",
            category_id=5,
            bounding_box=BoundingBox(ulx=1.5, uly=2.4, width=3.0, height=9.0, absolute_coords=True),
            model_id="test_model",
        )

        test_image.dump(cat_1)
        test_image.dump(cat_2)
        test_image.dump(cat_3)
        test_image.dump(cat_4)
        test_image.dump(cat_5)

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

        filtered_anns_5 = test_image.get_annotation()
        filtered_anns_5_ids = anns_to_ids(filtered_anns_5)

        filtered_anns_6 = test_image.get_annotation(service_id="test_service")
        filtered_anns_6_ids = anns_to_ids(filtered_anns_6)

        filtered_anns_7 = test_image.get_annotation(model_id="test_model")
        filtered_anns_7_ids = anns_to_ids(filtered_anns_7)

        filtered_anns_8 = test_image.get_annotation(service_id="test_model", annotation_ids=[cat_2.annotation_id])
        filtered_anns_8_ids = anns_to_ids(filtered_anns_8)

        # Assert
        assert set(filtered_anns_1_ids) == {cat_1.annotation_id}
        assert set(filtered_anns_2_ids) == set()
        assert set(filtered_anns_3_ids) == {cat_1.annotation_id, cat_3.annotation_id}
        assert set(filtered_anns_4_ids) == {cat_2.annotation_id, cat_3.annotation_id}
        assert set(filtered_anns_5_ids) == {
            cat_1.annotation_id,
            cat_2.annotation_id,
            cat_3.annotation_id,
            cat_4.annotation_id,
            cat_5.annotation_id,
        }
        assert set(filtered_anns_6_ids) == {cat_4.annotation_id}
        assert set(filtered_anns_7_ids) == {cat_5.annotation_id}
        assert set(filtered_anns_8_ids) == set()

    @staticmethod
    @mark.basic
    def test_image_ann_to_image(image: WhiteImage) -> None:
        """
        test  image_ann_to_image add attr: image to ImageAnnotation and generates Image instance correctly
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
        assert cat_1.image
        assert cat_1.image.height == 4
        assert cat_1.image.width == 10
        assert cat_1.get_bounding_box(test_image.image_id) == BoundingBox(
            ulx=15.0, uly=20.0, width=10.0, height=4.0, absolute_coords=True
        )
        assert cat_1.image.image.shape == (4, 10, 3)

    @staticmethod
    @mark.basic
    def test_load_image_from_dict() -> None:
        """
        test class  from_dict returns a image
        """
        df: DataFlow
        df = SerializerJsonlines.load(get_test_path() / "test_image.jsonl")
        df = MapData(df, lambda dp: Image.from_dict(**dp))
        image_list = collect_datapoint_from_dataflow(df)
        assert len(image_list) == 1

    @staticmethod
    @mark.basic
    def test_load_image_from_file() -> None:
        """
        test class from_file returns an image
        """
        test_file_path = get_test_path() / "test_image.json"
        image = Image.from_file(test_file_path.as_posix())
        assert isinstance(image, Image)

    @staticmethod
    @mark.basic
    def test_load_image_from_legacy_test_file() -> None:
        """
        test class from_file returns an image
        """
        test_file_path = get_test_path() / "legacy_test_image.json"
        image = Image.from_file(test_file_path.as_posix())
        assert isinstance(image, Image)

    @staticmethod
    @mark.basic
    def test_state_id_changes_when_annotations_added(image: WhiteImage) -> None:
        """
        state_id changes when annotations are added
        """

        # Arrange
        test_image = Image(location=image.loc, file_name=image.file_name)
        cat_1 = ImageAnnotation(
            category_name="FOO",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.0, uly=1.0, width=1.0, height=2.0, absolute_coords=True),
        )
        cat_2 = ImageAnnotation(
            category_name="BAK",
            category_id=2,
            bounding_box=BoundingBox(ulx=2.0, uly=2.0, width=2.0, height=2.0, absolute_coords=True),
        )
        cat_3 = ImageAnnotation(
            category_name="BAK",
            category_id=1,
            bounding_box=BoundingBox(ulx=1.5, uly=2.4, width=3.0, height=9.0, absolute_coords=True),
        )

        test_image.dump(cat_1)
        test_image.dump(cat_2)

        # Assert
        assert test_image.state_id == "bbd3ee7f-442e-3141-b808-ede127f153f5"

        # Act
        test_image.dump(cat_3)
        assert test_image.state_id == "6a949ef7-70e6-3e56-9d94-074882cf6a53"

    @staticmethod
    @mark.basic
    def test_get_annotation_id_to_annotation_maps(
        dp_image_with_layout_and_word_annotations: Image, annotation_maps: defaultdict[str, list[AnnotationMap]]
    ) -> None:
        """
        get_annotation_id_to_annotation_maps
        """

        # Arrange
        dp = dp_image_with_layout_and_word_annotations
        expected_annotation_maps = annotation_maps

        # Act
        ann_maps = dp.get_annotation_id_to_annotation_maps()

        # Assert
        assert ann_maps == expected_annotation_maps

    @staticmethod
    @mark.basic
    def test_get_service_id_to_annotation_id(
        dp_image_with_layout_and_word_annotations: Image, service_id_to_ann_id: dict[str, list[str]]
    ) -> None:
        """
        get_service_id_to_annotation_id
        """

        # Arrange
        dp = dp_image_with_layout_and_word_annotations
        expected_service_id_to_ann_id = service_id_to_ann_id

        # Act
        service_id_to_ann_id = dp.get_service_id_to_annotation_id()
        print(service_id_to_ann_id)
        # Assert
        assert service_id_to_ann_id == expected_service_id_to_ann_id

    @staticmethod
    @mark.basic
    def test_remove_by_annotation_id(dp_image_with_layout_and_word_annotations: Image) -> None:
        """
        remove
        """

        # Arrange
        dp = dp_image_with_layout_and_word_annotations
        anns = dp.get_annotation(annotation_ids="51fca38d-b181-3ea2-9c97-7e265febcc86")
        assert anns

        # Act
        dp.remove(annotation_ids=["51fca38d-b181-3ea2-9c97-7e265febcc86", "1413d499-ce19-3a50-861c-7d8c5a7ba772"])

        # Assert
        anns = dp.get_annotation(annotation_ids="51fca38d-b181-3ea2-9c97-7e265febcc86")
        assert not anns

    @staticmethod
    @mark.basic
    def test_remove_by_service_id(dp_image_with_layout_and_word_annotations: Image) -> None:
        """
        remove
        """

        # Arrange
        dp = dp_image_with_layout_and_word_annotations

        # Act
        dp.remove(service_ids="test_service")

        # Assert
        anns = dp.get_annotation(service_id="test_service")
        assert not anns
