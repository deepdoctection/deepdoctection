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

import json
from pathlib import Path

import numpy as np
from pytest import mark

from dd_datapoint.datapoint import BoundingBox, Image, ImageAnnotation

from .conftest import WhiteImage


class TestImageSerialization:
    """Test Image serialization and deserialization"""

    @staticmethod
    @mark.basic
    def test_as_dict_returns_dict(image: WhiteImage):
        """as_dict() returns dictionary"""
        img = Image(file_name=image.file_name, location=image.loc)

        result = img.as_dict()

        assert isinstance(result, dict)

    @staticmethod
    @mark.basic
    def test_as_dict_contains_basic_fields(image: WhiteImage):
        """as_dict() contains basic fields"""
        img = Image(file_name=image.file_name, location=image.loc)

        result = img.as_dict()

        assert "file_name" in result
        assert "location" in result
        assert "embeddings" in result
        assert "annotations" in result

    @staticmethod
    @mark.basic
    def test_as_dict_with_image(image: WhiteImage):
        """as_dict() converts image to base64"""
        img = Image(file_name=image.file_name)
        img.image = image.get_image_as_np_array()

        result = img.as_dict()

        assert "_image" in result
        assert isinstance(result["_image"], str)  # base64 string

    @staticmethod
    @mark.basic
    def test_as_dict_embeddings_are_dicts(image: WhiteImage):
        """as_dict() converts BoundingBox embeddings to dicts"""
        img = Image(file_name=image.file_name)
        img.set_width_height(100, 100)
        bbox = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        img.set_embedding("parent", bbox)

        result = img.as_dict()

        assert "embeddings" in result
        assert "parent" in result["embeddings"]
        assert isinstance(result["embeddings"]["parent"], dict)

    @staticmethod
    @mark.basic
    def test_as_json_returns_string(image: WhiteImage):
        """as_json() returns JSON string"""
        img = Image(file_name=image.file_name, location=image.loc)

        result = img.as_json()

        assert isinstance(result, str)

    @staticmethod
    @mark.basic
    def test_as_json_is_valid_json(image: WhiteImage):
        """as_json() returns valid JSON"""
        img = Image(file_name=image.file_name, location=image.loc)

        result = img.as_json()
        parsed = json.loads(result)

        assert isinstance(parsed, dict)
        assert parsed["file_name"] == image.file_name

    @staticmethod
    @mark.basic
    def test_as_json_is_formatted(image: WhiteImage):
        """as_json() returns formatted (indented) JSON"""
        img = Image(file_name=image.file_name)

        result = img.as_json()

        # Indented JSON has newlines
        assert "\n" in result

    @staticmethod
    @mark.basic
    def test_roundtrip_as_dict_recreates_image(image: WhiteImage):
        """Image can be recreated from as_dict() output"""
        img1 = Image(file_name=image.file_name, location=image.loc, external_id=image.external_id)
        img1.image = image.get_image_as_np_array()

        data = img1.as_dict()
        img2 = Image(**data)

        assert img2.file_name == img1.file_name
        assert img2.location == img1.location
        assert img2.image_id == img1.image_id

    @staticmethod
    @mark.basic
    def test_roundtrip_preserves_annotations(image: WhiteImage):
        """Annotations are preserved in roundtrip"""
        img1 = Image(file_name=image.file_name)
        ann = ImageAnnotation(
            category_name="TEST",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img1.dump(ann)

        data = img1.as_dict()
        img2 = Image(**data)

        assert len(img2.annotations) == 1
        assert img2.annotations[0].category_name.value == "TEST"

    @staticmethod
    @mark.basic
    def test_save_creates_json_file(image: WhiteImage, tmp_path: Path):
        """save() creates JSON file"""
        img = Image(file_name=image.file_name, location=str(tmp_path))
        img.image = image.get_image_as_np_array()

        result = img.save(path=tmp_path, dry=False)

        assert result is not None
        assert Path(result).exists()
        assert Path(result).suffix == ".json"

    @staticmethod
    @mark.basic
    def test_save_dry_returns_dict(image: WhiteImage):
        """save(dry=True) returns dict without saving"""
        img = Image(file_name=image.file_name)

        result = img.save(dry=True)

        assert isinstance(result, dict)

    @staticmethod
    @mark.basic
    def test_save_without_image_to_json(image: WhiteImage, tmp_path: Path):
        """save(image_to_json=False) excludes image data"""
        img = Image(file_name=image.file_name)
        img.image = image.get_image_as_np_array()

        result = img.save(path=tmp_path, image_to_json=False, dry=True)

        assert result["_image"] is None

    @staticmethod
    @mark.basic
    def test_from_file_loads_image(image: WhiteImage, tmp_path: Path):
        """from_file() loads Image from JSON file"""
        img1 = Image(file_name=image.file_name, location=str(tmp_path))
        img1.image = image.get_image_as_np_array()
        file_path = img1.save(path=tmp_path, dry=False)

        img2 = Image.from_file(file_path)

        assert isinstance(img2, Image)
        assert img2.file_name == img1.file_name

    @staticmethod
    @mark.basic
    def test_location_converted_to_string_in_serialization(image: WhiteImage):
        """Location is converted to string in serialization"""
        img = Image(file_name=image.file_name, location=Path("/some/path"))

        result = img.as_dict()

        assert isinstance(result["location"], str)

    @staticmethod
    @mark.basic
    def test_as_dict_excludes_annotation_ids_private_list(image: WhiteImage):
        """as_dict() excludes _annotation_ids internal list"""
        img = Image(file_name=image.file_name)
        ann = ImageAnnotation(
            category_name="TEST",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann)

        result = img.as_dict()

        assert "_annotation_ids" not in result

    @staticmethod
    @mark.basic
    def test_as_dict_includes_summary(image: WhiteImage):
        """as_dict() includes summary"""
        img = Image(file_name=image.file_name)
        _ = img.summary  # Access to create

        result = img.as_dict()

        assert "_summary" in result
        assert result["_summary"] is not None

    @staticmethod
    @mark.basic
    def test_as_dict_summary_none_when_not_accessed(image: WhiteImage):
        """as_dict() has None summary when not accessed"""
        img = Image(file_name=image.file_name)

        result = img.as_dict()

        assert "_summary" in result
        assert result["_summary"] is None

    @staticmethod
    @mark.basic
    def test_roundtrip_preserves_embeddings(image: WhiteImage):
        """Embeddings are preserved in roundtrip"""
        img1 = Image(file_name=image.file_name)
        img1.set_width_height(100, 100)
        bbox = BoundingBox(ulx=10, uly=20, width=30, height=40, absolute_coords=True)
        img1.set_embedding("parent", bbox)

        data = img1.as_dict()
        img2 = Image(**data)

        assert "parent" in img2.embeddings
        retrieved_bbox = img2.get_embedding("parent")
        assert retrieved_bbox.ulx == 10
        assert retrieved_bbox.uly == 20

    @staticmethod
    @mark.basic
    def test_roundtrip_preserves_page_number(image: WhiteImage):
        """page_number is preserved in roundtrip"""
        img1 = Image(file_name=image.file_name, page_number=5)

        data = img1.as_dict()
        img2 = Image(**data)

        assert img2.page_number == 5

    @staticmethod
    @mark.basic
    def test_roundtrip_preserves_document_id(image: WhiteImage):
        """document_id is preserved in roundtrip"""
        img1 = Image(file_name=image.file_name, document_id="doc_123")

        data = img1.as_dict()
        img2 = Image(**data)

        assert img2.document_id == "doc_123"

    @staticmethod
    @mark.basic
    def test_save_highest_hierarchy_only(image: WhiteImage, tmp_path: Path):
        """save(highest_hierarchy_only=True) removes nested images"""
        img = Image(file_name=image.file_name, location=str(tmp_path))
        img.image = image.get_image_as_np_array()

        ann = ImageAnnotation(
            category_name="TEST",
            bounding_box=BoundingBox(ulx=10, uly=10, width=20, height=20, absolute_coords=True)
        )
        img.dump(ann)
        img.image_ann_to_image(annotation_id=ann.annotation_id, crop_image=True)

        result = img.save(path=tmp_path, highest_hierarchy_only=True, dry=True)

        # Check that annotation image is removed
        assert result["annotations"][0]["image"] is None

    @staticmethod
    @mark.basic
    def test_as_dict_includes_bbox(image: WhiteImage):
        """as_dict() includes _bbox"""
        img = Image(file_name=image.file_name)
        img.set_width_height(100, 200)

        result = img.as_dict()

        assert "_bbox" in result
        assert isinstance(result["_bbox"], dict)

    @staticmethod
    @mark.basic
    def test_roundtrip_with_multiple_annotations(image: WhiteImage):
        """Multiple annotations are preserved in roundtrip"""
        img1 = Image(file_name=image.file_name)

        for i in range(3):
            ann = ImageAnnotation(
                category_name=f"TEST_{i}",
                bounding_box=BoundingBox(ulx=i*10, uly=i*10, width=20, height=20, absolute_coords=True)
            )
            img1.dump(ann)

        data = img1.as_dict()
        img2 = Image(**data)

        assert len(img2.annotations) == 3

    @staticmethod
    @mark.basic
    def test_json_contains_all_required_fields(image: WhiteImage):
        """as_json() contains all required fields"""
        img = Image(file_name=image.file_name, location=image.loc)
        img.image = image.get_image_as_np_array()

        json_str = img.as_json()
        parsed = json.loads(json_str)

        assert "file_name" in parsed
        assert "location" in parsed
        assert "embeddings" in parsed
        assert "annotations" in parsed
        assert "_bbox" in parsed
        assert "_image" in parsed

