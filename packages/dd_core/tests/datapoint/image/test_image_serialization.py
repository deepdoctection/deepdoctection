# -*- coding: utf-8 -*-
# File: test_image_serialization.py

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
Testing Image serialization (as_dict, as_json, from_file, save)
"""

from pathlib import Path

from dd_core.datapoint import BoundingBox, Image, ImageAnnotation

from ..conftest import WhiteImage


class TestImageSerialization:
    """Test Image serialization and deserialization"""

    @staticmethod
    def test_as_dict_contains_basic_fields(white_image: WhiteImage) -> None:
        """as_dict() contains basic fields"""
        img = Image(file_name=white_image.file_name, location=white_image.location)

        result = img.as_dict()

        assert "file_name" in result
        assert "location" in result
        assert "document_id" in result
        assert "page_number" in result
        assert "embeddings" in result
        assert "annotations" in result
        assert "_bbox" in result
        assert "_image" in result
        assert "_summary" in result

    @staticmethod
    def test_as_dict_with_image(white_image: WhiteImage) -> None:
        """as_dict() converts image to base64"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image

        result = img.as_dict()

        assert "_image" in result
        assert isinstance(result["_image"], str)

    @staticmethod
    def test_roundtrip_as_dict_recreates_image(white_image: WhiteImage) -> None:
        """Image can be recreated from as_dict() output"""
        img1 = Image(
            file_name=white_image.file_name, location=white_image.location, external_id=white_image.external_id
        )
        img1.image = white_image.image

        data = img1.as_dict()
        img2 = Image(**data)

        assert img2.file_name == img1.file_name
        assert img2.location == img1.location
        assert img2.image_id == img1.image_id

    @staticmethod
    def test_roundtrip_preserves_annotations(white_image: WhiteImage) -> None:
        """Annotations are preserved in roundtrip"""
        img1 = Image(file_name=white_image.file_name)
        box = BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        ann = ImageAnnotation(category_name="test_cat_1", bounding_box=box)
        img1.dump(ann)

        data = img1.as_dict()
        img2 = Image(**data)

        assert len(img2.annotations) == 1
        assert img2.annotations[0].category_name.value == "test_cat_1"  # type: ignore
        assert img2.annotations[0].bounding_box == box

    @staticmethod
    def test_save_creates_json_file(white_image: WhiteImage, tmp_path: Path) -> None:
        """save() creates JSON file"""
        img = Image(file_name=white_image.file_name, location=str(tmp_path))
        img.image = white_image.image

        result = img.save(path=tmp_path, dry=False)
        assert isinstance(result, str)

        assert result is not None
        assert Path(result).exists()
        assert Path(result).suffix == ".json"

    @staticmethod
    def test_save_without_image_to_json(white_image: WhiteImage, tmp_path: Path) -> None:
        """save(image_to_json=False) excludes image data"""
        img = Image(file_name=white_image.file_name)
        img.image = white_image.image

        result = img.save(path=tmp_path, image_to_json=False, dry=True)

        assert isinstance(result, dict)
        assert result["_image"] is None

    @staticmethod
    def test_from_file_loads_image(white_image: WhiteImage, tmp_path: Path) -> None:
        """from_file() loads Image from JSON file"""
        img1 = Image(file_name=white_image.file_name, location=str(tmp_path))
        img1.image = white_image.image
        file_path = img1.save(path=tmp_path, dry=False)

        assert isinstance(file_path, str)
        img2 = Image.from_file(file_path)

        assert isinstance(img2, Image)
        assert img2.file_name == img1.file_name

    @staticmethod
    def test_roundtrip_preserves_embeddings(white_image: WhiteImage) -> None:
        """Embeddings are preserved in roundtrip"""
        img1 = Image(file_name=white_image.file_name)
        img1.set_width_height(100, 100)
        bbox = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        img1.set_embedding(img1.image_id, bbox)

        data = img1.as_dict()
        img2 = Image(**data)

        assert img1.image_id in img2.embeddings
        retrieved_bbox = img2.get_embedding(img1.image_id)
        assert retrieved_bbox.ulx == 10
        assert retrieved_bbox.uly == 20

    @staticmethod
    def test_roundtrip_preserves_page_number(white_image: WhiteImage) -> None:
        """page_number is preserved in roundtrip"""
        img1 = Image(file_name=white_image.file_name, page_number=5)

        data = img1.as_dict()
        img2 = Image(**data)

        assert img2.page_number == 5

    @staticmethod
    def test_save_highest_hierarchy_only(white_image: WhiteImage, tmp_path: Path) -> None:
        """save(highest_hierarchy_only=True) removes nested images"""
        img = Image(file_name=white_image.file_name, location=str(tmp_path))
        img.image = white_image.image

        ann = ImageAnnotation(
            category_name="test_cat_1",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True),
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

        result = img.save(path=tmp_path, highest_hierarchy_only=True, dry=True)
        assert isinstance(result, dict)

        # Check that annotation image is removed
        assert result["annotations"][0]["image"] is None

    @staticmethod
    def test_roundtrip_with_multiple_annotations(white_image: WhiteImage) -> None:
        """Multiple annotations are preserved in roundtrip"""
        img1 = Image(file_name=white_image.file_name)

        for i in range(1, 3):
            ann = ImageAnnotation(
                category_name=f"test_cat_{i}",
                bounding_box=BoundingBox(ulx=i * 10, uly=i * 10, width=20, height=20, absolute_coords=True),
            )
            img1.dump(ann)

        data = img1.as_dict()
        img2 = Image(**data)

        assert len(img2.annotations) == 2
