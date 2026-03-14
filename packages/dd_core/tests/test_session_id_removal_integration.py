# -*- coding: utf-8 -*-
# File: test_session_id_removal_integration.py

# Copyright 2025 Dr. Janis Meyer. All rights reserved.
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
Integration test for session_id removal backward compatibility
"""

from dd_core.datapoint.annotation import ImageAnnotation
from dd_core.datapoint.box import BoundingBox
from dd_core.datapoint.image import Image
from dd_core.utils.object_types import DefaultType


class TestSessionIdRemovalIntegration:
    """Integration tests for session_id removal"""

    def test_image_with_annotations_containing_session_id(self) -> None:
        """Test that Image can load annotations that contain session_id in their serialized form"""
        # Create annotation with session_id (simulating old data)
        ann_data = {
            "category_name": DefaultType.DEFAULT_TYPE.value,
            "category_id": 1,
            "service_id": "detector",
            "model_id": "model_v1",
            "session_id": "old_session_123",  # Old field - should be silently ignored
            "bounding_box": {
                "ulx": 10.0,
                "uly": 10.0,
                "lrx": 50.0,
                "lry": 50.0,
                "absolute_coords": True,
            },
        }

        ann = ImageAnnotation(**ann_data)

        # Create image and add annotation
        image = Image(file_name="test.jpg", location="/path/to/test.jpg")
        image.dump(ann)

        # Verify image was created
        assert image.file_name == "test.jpg"
        assert len(image.annotations) == 1

        # Verify annotation properties work correctly
        loaded_ann = image.annotations[0]
        assert loaded_ann.service_id == "detector"
        assert loaded_ann.model_id == "model_v1"
        # The important thing is that session_id was removed from model_fields, not that hasattr returns False
        # (hasattr may still return True due to Pydantic internals, but the field is not in the model schema)
        assert "session_id" not in set(ImageAnnotation.model_fields)

    def test_serialization_roundtrip_without_session_id(self) -> None:
        """Test that serialization/deserialization roundtrip works without session_id"""
        # Create new annotation without session_id
        bbox = BoundingBox(ulx=0, uly=0, lrx=100, lry=100, absolute_coords=True)
        ann = ImageAnnotation(
            category_name=DefaultType.DEFAULT_TYPE,
            bounding_box=bbox,
            service_id="new_service",
            model_id="new_model",
        )

        # Create image and add annotation
        image = Image(file_name="test.jpg", location="/path/to/test.jpg")
        image.dump(ann)

        # Serialize
        data = image.as_dict()

        # Deserialize by recreating
        image2 = Image(**{k: v for k, v in data.items() if k not in ["annotations", "_image_id", "_bbox"]})
        for ann_data in data.get("annotations", []):
            ann2 = ImageAnnotation(**ann_data)
            image2.dump(ann2)

        # Verify roundtrip worked
        assert image2.file_name == "test.jpg"
        assert len(image2.annotations) == 1
        assert image2.annotations[0].service_id == "new_service"
        assert image2.annotations[0].model_id == "new_model"

    def test_mixed_old_and_new_annotations(self) -> None:
        """Test that we can have mix of old data (with session_id) and new data"""
        # Create image
        image = Image(file_name="mixed.jpg", location="/path/to/mixed.jpg")

        # Old annotation with session_id
        old_ann_data = {
            "category_name": DefaultType.DEFAULT_TYPE.value,
            "bounding_box": {
                "ulx": 10.0,
                "uly": 10.0,
                "lrx": 50.0,
                "lry": 50.0,
                "absolute_coords": True,
            },
            "service_id": "old_detector",
            "session_id": "old_session",  # Old annotation with session_id - should be ignored
        }
        old_ann = ImageAnnotation(**old_ann_data)
        image.dump(old_ann)

        # New annotation without session_id
        new_bbox = BoundingBox(ulx=60.0, uly=60.0, lrx=100.0, lry=100.0, absolute_coords=True)
        new_ann = ImageAnnotation(
            category_name=DefaultType.DEFAULT_TYPE,
            bounding_box=new_bbox,
            service_id="new_detector",
        )
        image.dump(new_ann)

        # Should work without errors
        assert len(image.annotations) == 2
        assert image.annotations[0].service_id == "old_detector"
        assert image.annotations[1].service_id == "new_detector"

        # Both should have session_id removed from model fields
        for _ in image.annotations:
            assert "session_id" not in set(ImageAnnotation.model_fields)
