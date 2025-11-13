# -*- coding: utf-8 -*-
# File: test_bbox_init_validate.py

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
Comprehensive tests for BoundingBox initialization and validation.
Tests constructor completeness, zero/negative extent rejection, relative coordinate bounds,
zero coordinate handling, and banker's rounding behavior.
"""

import pytest
from dd_datapoint.datapoint import BoundingBox
from dd_datapoint.utils.error import BoundingBoxError


class TestBBoxInitValidate:
    """Tests for BoundingBox initialization and validation invariants"""

    def test_constructor_requires_lrx_lry_or_width_height_absolute(self):
        """Absolute box must provide either (lrx,lry) or (width,height)"""
        # Neither provided -> error
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=True, ulx=10, uly=20)
    
    def test_constructor_requires_lrx_lry_or_width_height_relative(self):
        """Relative box must provide either (lrx,lry) or (width,height)"""
        # Neither provided -> error  
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.4)

    def test_constructor_zero_width_rejected_absolute(self):
        """Absolute box with zero width is rejected"""
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=True, ulx=10, uly=20, width=0, height=30)
    
    def test_constructor_zero_height_rejected_absolute(self):
        """Absolute box with zero height is rejected"""
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=True, ulx=10, uly=20, width=30, height=0)

    def test_constructor_negative_width_rejected_absolute(self):
        """Absolute box with negative width is rejected"""
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=True, ulx=10, uly=20, width=-5, height=30)
    
    def test_constructor_negative_height_rejected_absolute(self):
        """Absolute box with negative height is rejected"""
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=True, ulx=10, uly=20, width=30, height=-5)

    def test_constructor_zero_width_rejected_relative(self):
        """Relative box with zero width is rejected"""
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.0, height=0.3)
    
    def test_constructor_zero_height_rejected_relative(self):
        """Relative box with zero height is rejected"""
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.3, height=0.0)

    def test_constructor_negative_width_rejected_relative(self):
        """Relative box with negative width is rejected"""
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=False, ulx=0.5, uly=0.5, width=-0.1, height=0.3)
    
    def test_constructor_negative_height_rejected_relative(self):
        """Relative box with negative height is rejected"""
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=False, ulx=0.5, uly=0.5, width=0.3, height=-0.1)

    def test_constructor_lrx_less_than_ulx_rejected_absolute(self):
        """Absolute box where lrx < ulx is rejected"""
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=True, ulx=100, uly=50, lrx=50, lry=100)
    
    def test_constructor_lry_less_than_uly_rejected_absolute(self):
        """Absolute box where lry < uly is rejected"""
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=True, ulx=50, uly=100, lrx=100, lry=50)

    def test_constructor_lrx_equals_ulx_rejected_absolute(self):
        """Absolute box where lrx == ulx is rejected (zero width)"""
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            BoundingBox(absolute_coords=True, ulx=100, uly=50, lrx=100, lry=100)
    
    def test_constructor_lry_equals_uly_rejected_absolute(self):
        """Absolute box where lry == uly is rejected (zero height)"""
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            BoundingBox(absolute_coords=True, ulx=50, uly=100, lrx=100, lry=100)

    def test_relative_coords_within_0_1_boundaries_allowed(self):
        """Relative coordinates at exactly 0.0 and 1.0 are valid"""
        # Box from (0,0) to (1,1) should be valid
        box = BoundingBox(absolute_coords=False, ulx=0.0, uly=0.0, lrx=1.0, lry=1.0)
        assert box.ulx == 0.0
        assert box.uly == 0.0
        assert box.lrx == 1.0
        assert box.lry == 1.0

    
    def test_relative_coords_above_one_rejected(self):
        """Relative coordinates > 1 are rejected"""
        with pytest.raises(BoundingBoxError, match="must be between 0 and 1"):
            BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=1.1, lry=0.8)

    def test_zero_coordinate_is_valid_absolute(self):
        """Zero is a valid coordinate in absolute mode (no truthiness confusion)"""
        # ulx=0, uly=0 should work
        box = BoundingBox(absolute_coords=True, ulx=0, uly=0, width=100, height=50)
        assert box.ulx == 0
        assert box.uly == 0
        assert box.width == 100
        assert box.height == 50
    
    def test_zero_coordinate_is_valid_relative(self):
        """Zero is a valid coordinate in relative mode (no truthiness confusion)"""
        # ulx=0.0, uly=0.0 should work
        box = BoundingBox(absolute_coords=False, ulx=0.0, uly=0.0, width=0.5, height=0.5)
        assert box.ulx == 0.0
        assert box.uly == 0.0

    def test_negative_ul_rejected_absolute(self):
        """Absolute box with negative upper-left coordinates is rejected"""
        with pytest.raises(BoundingBoxError, match="ul must be >= \\(0,0\\)"):
            BoundingBox(absolute_coords=True, ulx=-10, uly=20, width=100, height=50)
        
        with pytest.raises(BoundingBoxError, match="ul must be >= \\(0,0\\)"):
            BoundingBox(absolute_coords=True, ulx=10, uly=-20, width=100, height=50)

    @pytest.mark.parametrize("value,expected_rounded", [
        (0.5, 0),
        (1.5, 1),
        (2.5, 2),
        (3.5, 3),
        (4.5, 4),
        (5.5, 5),
    ])
    def test_rounding_absolute_coords(self, value, expected_rounded):
        """Verify banker's rounding (round half to even) for absolute coordinates"""
        # Create box where ulx uses the test value
        box = BoundingBox(absolute_coords=True, ulx=value, uly=10, width=100, height=50)
        assert box.ulx == expected_rounded
    
    @pytest.mark.parametrize("width_val,expected_lrx", [
        (0.5, 1),
        (1.5, 2),
        (2.5, 3),
        (3.5, 4),
        (100.5, 101),
        (101.5, 102),
    ])
    def test_rounding_width_computation_absolute(self, width_val, expected_lrx):
        """Verify rounding when computing lrx from width"""
        box = BoundingBox(absolute_coords=True, ulx=0, uly=10, width=width_val, height=50)
        assert box._lrx == expected_lrx

    @pytest.mark.parametrize("rel_val,expected_int", [
        (0.5 / 10**8, 0),
        (1.5 / 10**8, 1),
        (2.5 / 10**8, 2),
        (3.5 / 10**8, 4),
    ])
    def test_rounding_relative_coords_internal_storage(self, rel_val, expected_int):
        """Verify rounding for relative coordinate internal storage"""
        box = BoundingBox(absolute_coords=False, ulx=rel_val, uly=0.1, width=0.5, height=0.3)
        assert box._ulx == expected_int

    def test_valid_minimal_absolute_box(self):
        """A minimal valid absolute box (1x1 pixel)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=1, height=1)
        assert box.ulx == 10
        assert box.uly == 20
        assert box.width == 1
        assert box.height == 1
        assert box.lrx == 11
        assert box.lry == 21

    def test_as_dict_roundtrip_absolute_width_height(self):
        """Round-trip via as_dict for absolute coords (constructed with width/height)"""
        box = BoundingBox(absolute_coords=True, ulx=10.5, uly=20.5, width=30.5, height=40.5)
        d = box.as_dict()
        rebuilt = BoundingBox(**d)
        assert rebuilt == box
        assert rebuilt.absolute_coords is True
        assert rebuilt.ulx == 10 and rebuilt.uly == 20
        assert rebuilt.lrx == 41 and rebuilt.lry == 61

    def test_as_dict_roundtrip_absolute_lrx_lry(self):
        """Round-trip via as_dict for absolute coords (constructed with lrx/lry)"""
        box = BoundingBox(absolute_coords=True, ulx=3.5, uly=2.5, lrx=7.5, lry=6.5)
        d = box.as_dict()
        rebuilt = BoundingBox(**d)
        assert rebuilt == box
        assert (rebuilt.ulx, rebuilt.uly, rebuilt.lrx, rebuilt.lry) == (3, 2, 8, 7)

    def test_as_dict_roundtrip_relative_width_height(self):
        """Round-trip via as_dict for relative coords (constructed with width/height)"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.3, height=0.4)
        d = box.as_dict()
        rebuilt = BoundingBox(**d)
        assert rebuilt == box
        assert rebuilt.absolute_coords is False
        # Check values (tolerate float imprecision)
        assert pytest.approx(rebuilt.ulx, rel=0, abs=1e-9) == 0.1
        assert pytest.approx(rebuilt.uly, rel=0, abs=1e-9) == 0.2
        assert pytest.approx(rebuilt.lrx, rel=0, abs=1e-9) == 0.4
        assert pytest.approx(rebuilt.lry, rel=0, abs=1e-9) == 0.6

    def test_as_dict_roundtrip_relative_lrx_lry(self):
        """Round-trip via as_dict for relative coords (constructed with lrx/lry)"""
        box = BoundingBox(absolute_coords=False, ulx=0.12, uly=0.34, lrx=0.56, lry=0.78)
        d = box.as_dict()
        rebuilt = BoundingBox(**d)
        assert rebuilt == box
        assert rebuilt.absolute_coords is False
        assert pytest.approx(rebuilt.ulx, rel=0, abs=1e-9) == 0.12
        assert pytest.approx(rebuilt.uly, rel=0, abs=1e-9) == 0.34
        assert pytest.approx(rebuilt.lrx, rel=0, abs=1e-9) == 0.56
        assert pytest.approx(rebuilt.lry, rel=0, abs=1e-9) == 0.78
