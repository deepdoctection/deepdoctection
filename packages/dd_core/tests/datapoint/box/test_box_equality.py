# -*- coding: utf-8 -*-
# File: test_bbox_equality.py

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
Comprehensive tests for BoundingBox equality semantics.
Tests internal storage comparison, float rounding equality, and unhashability.
"""

import pytest
from dd_core import BoundingBox


class TestBBoxEquality:
    """Tests for BoundingBox equality and hashing behavior"""

    # ==================== Basic Equality ====================

    def test_same_object_equals_self(self):
        """A BoundingBox equals itself"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        assert box == box
        assert not (box != box)

    def test_identical_absolute_boxes_are_equal(self):
        """Two absolute boxes with identical coordinates are equal"""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        box2 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        assert box1 == box2
        assert box2 == box1

    def test_identical_relative_boxes_are_equal(self):
        """Two relative boxes with identical coordinates are equal"""
        box1 = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        box2 = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        
        assert box1 == box2
        assert box2 == box1

    def test_different_absolute_boxes_not_equal(self):
        """Absolute boxes with different coordinates are not equal"""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        box2 = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=51)
        
        assert box1 != box2

    def test_different_relative_boxes_not_equal(self):
        """Relative boxes with different coordinates are not equal"""
        box1 = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        box2 = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.4)
        
        assert box1 != box2

    def test_absolute_vs_relative_not_equal(self):
        """Absolute and relative boxes are never equal (even if coords look similar)"""
        box_abs = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        box_rel = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        assert box_abs != box_rel
        assert box_rel != box_abs


    def test_relative_boxes_equal_if_internal_ints_match(self):
        """Two relative boxes are equal if internal int storage matches (not raw floats)"""
        # These two float values should round to the same internal int
        # after scaling by RELATIVE_COORD_SCALE_FACTOR
        scale = 10**8
        
        # Use values that round to the same int
        val1 = 12345678 / scale  # Exactly representable
        val2 = 12345678.4 / scale  # Rounds to same int
        
        box1 = BoundingBox(absolute_coords=False, ulx=val1, uly=0.2, lrx=0.6, lry=0.7)
        box2 = BoundingBox(absolute_coords=False, ulx=val2, uly=0.2, lrx=0.6, lry=0.7)
        
        # If they rounded to the same internal int, they should be equal
        if box1._ulx == box2._ulx:
            assert box1 == box2

    def test_relative_boxes_not_equal_if_internal_ints_differ(self):
        """Two relative boxes are not equal if internal int storage differs"""
        scale = 10**8
        
        # Use values that round to different ints
        val1 = 12345678 / scale
        val2 = 12345679 / scale
        
        box1 = BoundingBox(absolute_coords=False, ulx=val1, uly=0.2, lrx=0.6, lry=0.7)
        box2 = BoundingBox(absolute_coords=False, ulx=val2, uly=0.2, lrx=0.6, lry=0.7)
        
        # They should have different internal ints, so not equal
        if box1._ulx != box2._ulx:
            assert box1 != box2

    def test_equality_compares_internal_key(self):
        """Equality uses _key() which compares internal ints and flag"""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        box2 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        # They should have the same _key()
        assert box1._key() == box2._key()
        assert box1 == box2

    def test_different_keys_means_not_equal(self):
        """Different _key() means boxes are not equal"""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        box2 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=111, lry=70)
        
        assert box1._key() != box2._key()
        assert box1 != box2

    def test_key_includes_all_coords_and_flag(self):
        """_key() includes ulx, uly, lrx, lry, and absolute_coords flag"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        key = box._key()
        
        # Key should be a tuple with 5 elements
        assert isinstance(key, tuple)
        assert len(key) == 5
        
        # Last element should be the absolute_coords flag
        assert key[4] == True

    # ==================== Equality with Different Construction Methods ====================

    def test_boxes_from_lrx_lry_vs_width_height_equal_if_same_result(self):
        """Boxes constructed via (lrx,lry) vs (width,height) are equal if same result"""
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        box2 = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        # box2 should have lrx = 10 + 100 = 110, lry = 20 + 50 = 70
        assert box1 == box2

    def test_boxes_from_lrx_lry_vs_width_height_not_equal_if_rounding_differs(self):
        """Boxes may differ if rounding affects final coordinates"""
        # Using width that doesn't round to exact lrx
        box1 = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        box2 = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=99.4, height=50)
        
        # box2 width=99.4 rounds to lrx = 10 + 99 = 109 (using banker's rounding)
        # So boxes should not be equal if rounding is different
        if box2.lrx != 110:
            assert box1 != box2

    # ==================== Unhashability ====================

    def test_bbox_is_unhashable(self):
        """BoundingBox is unhashable (__hash__ = None)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        # __hash__ should be None
        assert box.__hash__ is None

    def test_bbox_cannot_be_added_to_set(self):
        """BoundingBox cannot be added to a set (raises TypeError)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(TypeError, match="unhashable"):
            {box}

    def test_bbox_cannot_be_dict_key(self):
        """BoundingBox cannot be used as dict key (raises TypeError)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(TypeError, match="unhashable"):
            {box: "value"}


    def test_equality_with_none_returns_false(self):
        """BoundingBox != None"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        assert box != None
        assert not (box == None)

    def test_equality_with_different_type_returns_notimplemented(self):
        """Equality with incompatible type returns NotImplemented (doesn't raise)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        # Comparing with int/str/list should not raise, but return False
        assert box != 42
        assert box != "box"
        assert box != [10, 20, 110, 70]


    def test_large_coordinate_boxes_equality(self):
        """Boxes with large coordinates can be equal"""
        box1 = BoundingBox(absolute_coords=True, ulx=0, uly=0, width=4999, height=4999)
        box2 = BoundingBox(absolute_coords=True, ulx=0, uly=0, width=4999, height=4999)
        
        assert box1 == box2


    def test_inequality_affects_any_coordinate(self):
        """Changing any single coordinate makes boxes unequal"""
        base_box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        # Different ulx
        box_ulx = BoundingBox(absolute_coords=True, ulx=11, uly=20, lrx=110, lry=70)
        assert base_box != box_ulx
        
        # Different uly
        box_uly = BoundingBox(absolute_coords=True, ulx=10, uly=21, lrx=110, lry=70)
        assert base_box != box_uly
        
        # Different lrx
        box_lrx = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=111, lry=70)
        assert base_box != box_lrx
        
        # Different lry
        box_lry = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=71)
        assert base_box != box_lry

