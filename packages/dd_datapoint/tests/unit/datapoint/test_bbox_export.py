# -*- coding: utf-8 -*-
# File: test_bbox_export.py

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
Comprehensive tests for BoundingBox export methods.
Tests to_np_array, to_list for all modes (xyxy, xywh, poly), area property,
and serialization via as_dict/model_dump.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from dd_datapoint.datapoint import BoundingBox


class TestBBoxExport:
    """Tests for BoundingBox export and serialization methods"""


    def test_to_np_array_xyxy_matches_to_list_absolute(self):
        """to_np_array('xyxy') matches to_list('xyxy') for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        np_arr = box.to_np_array(mode="xyxy")
        lst = box.to_list(mode="xyxy")
        
        assert_allclose(np_arr, lst, rtol=1e-6)

    def test_to_np_array_xyxy_matches_to_list_relative(self):
        """to_np_array('xyxy') matches to_list('xyxy') for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        
        np_arr = box.to_np_array(mode="xyxy")
        lst = box.to_list(mode="xyxy")
        
        assert_allclose(np_arr, lst, rtol=1e-6)

    def test_to_list_xyxy_absolute_returns_ints(self):
        """to_list('xyxy') returns rounded ints for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10.3, uly=20.7, lrx=110.5, lry=70.2)
        
        lst = box.to_list(mode="xyxy")
        
        # Absolute coords should be rounded
        assert isinstance(lst[0], int)
        assert isinstance(lst[1], int)
        assert isinstance(lst[2], int)
        assert isinstance(lst[3], int)

    def test_to_list_xyxy_relative_returns_floats(self):
        """to_list('xyxy') returns floats for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        
        lst = box.to_list(mode="xyxy")
        
        # Relative coords should be floats
        assert isinstance(lst[0], float)
        assert isinstance(lst[1], float)
        assert isinstance(lst[2], float)
        assert isinstance(lst[3], float)

    def test_to_list_xyxy_with_scaling_absolute(self):
        """to_list('xyxy') applies scale factors correctly for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=100, lry=80)
        
        lst = box.to_list(mode="xyxy", scale_x=2.0, scale_y=3.0)
        
        # Expected: [10*2, 20*3, 100*2, 80*3] = [20, 60, 200, 240]
        assert lst == [20, 60, 200, 240]

    def test_to_list_xyxy_raises_error_when_scaling_too_large(self):
        """to_list('xyxy') raises ValueError when scaling relative coords with scale > 1"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.5, lry=0.8)

        with pytest.raises(ValueError):
           box.to_list(mode="xyxy", scale_x=1000, scale_y=0.4)


    def test_to_np_array_xywh_matches_to_list_absolute(self):
        """to_np_array('xywh') matches to_list('xywh') for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        np_arr = box.to_np_array(mode="xywh")
        lst = box.to_list(mode="xywh")
        
        assert_allclose(np_arr, lst, rtol=1e-6)

    def test_to_np_array_xywh_matches_to_list_relative(self):
        """to_np_array('xywh') matches to_list('xywh') for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        np_arr = box.to_np_array(mode="xywh")
        lst = box.to_list(mode="xywh")
        
        assert_allclose(np_arr, lst, rtol=1e-6)

    def test_to_list_xywh_absolute_coords(self):
        """to_list('xywh') returns [ulx, uly, width, height] for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        lst = box.to_list(mode="xywh")
        
        assert lst == [10, 20, 100, 50]

    def test_to_list_xywh_relative_coords(self):
        """to_list('xywh') returns [ulx, uly, width, height] for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        lst = box.to_list(mode="xywh")
        
        assert abs(lst[0] - 0.1) < 1e-6
        assert abs(lst[1] - 0.2) < 1e-6
        assert abs(lst[2] - 0.5) < 1e-6
        assert abs(lst[3] - 0.3) < 1e-6

    def test_to_list_xywh_with_scaling_absolute(self):
        """to_list('xywh') applies scale factors correctly for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        lst = box.to_list(mode="xywh", scale_x=2.0, scale_y=3.0)
        
        # Expected: [10*2, 20*3, 100*2, 50*3] = [20, 60, 200, 150]
        assert lst == [20, 60, 200, 150]

    def test_to_np_array_poly_matches_to_list_absolute(self):
        """to_np_array('poly') matches to_list('poly') for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        np_arr = box.to_np_array(mode="poly")
        lst = box.to_list(mode="poly")
        
        assert_allclose(np_arr, lst, rtol=1e-6)

    def test_to_np_array_poly_matches_to_list_relative(self):
        """to_np_array('poly') matches to_list('poly') for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        
        np_arr = box.to_np_array(mode="poly")
        lst = box.to_list(mode="poly")
        
        assert_allclose(np_arr, lst, rtol=1e-6)

    def test_to_list_poly_absolute_coords(self):
        """to_list('poly') returns 8 coordinates (4 corners) for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        lst = box.to_list(mode="poly")
        
        # Expected: [ulx, uly, lrx, uly, lrx, lry, ulx, lry]
        assert lst == [10, 20, 110, 20, 110, 70, 10, 70]

    def test_to_list_poly_relative_coords(self):
        """to_list('poly') returns 8 coordinates (4 corners) for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        
        lst = box.to_list(mode="poly")
        
        # Should have 8 elements
        assert len(lst) == 8
        
        # Expected pattern: [ulx, uly, lrx, uly, lrx, lry, ulx, lry]
        assert abs(lst[0] - 0.1) < 1e-6  # ulx
        assert abs(lst[1] - 0.2) < 1e-6  # uly
        assert abs(lst[2] - 0.6) < 1e-6  # lrx
        assert abs(lst[3] - 0.2) < 1e-6  # uly
        assert abs(lst[4] - 0.6) < 1e-6  # lrx
        assert abs(lst[5] - 0.7) < 1e-6  # lry
        assert abs(lst[6] - 0.1) < 1e-6  # ulx
        assert abs(lst[7] - 0.7) < 1e-6  # lry

    def test_to_list_poly_with_scaling_absolute(self):
        """to_list('poly') applies scale factors correctly for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        lst = box.to_list(mode="poly", scale_x=2.0, scale_y=3.0)
        
        # Expected: [10*2, 20*3, 110*2, 20*3, 110*2, 70*3, 10*2, 70*3]
        # = [20, 60, 220, 60, 220, 210, 20, 210]
        assert lst == [20, 60, 220, 60, 220, 210, 20, 210]


    def test_to_np_array_invalid_mode_raises(self):
        """to_np_array with invalid mode raises AssertionError"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(AssertionError):
            box.to_np_array(mode="invalid")

    def test_to_list_invalid_mode_raises(self):
        """to_list with invalid mode raises AssertionError"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(AssertionError):
            box.to_list(mode="invalid")

    def test_area_absolute_correct_value(self):
        """area returns width * height for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        assert box.area == 100 * 50
        assert box.area == 5000

    def test_area_absolute_is_int(self):
        """area returns int for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        assert isinstance(box.area, int)

    def test_area_relative_raises_valueerror(self):
        """area raises ValueError for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        with pytest.raises(ValueError, match="Cannot calculate area.*relative"):
            _ = box.area


    def test_as_dict_contains_required_fields(self):
        """as_dict() returns dict with required fields"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        d = box.as_dict()
        
        assert "absolute_coords" in d
        assert "ulx" in d
        assert "uly" in d
        assert "lrx" in d
        assert "lry" in d

    def test_as_dict_correct_values_absolute(self):
        """as_dict() returns correct values for absolute coords"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        d = box.as_dict()
        
        assert d["absolute_coords"] is True
        assert d["ulx"] == 10
        assert d["uly"] == 20
        assert d["lrx"] == 110
        assert d["lry"] == 70

    def test_as_dict_correct_values_relative(self):
        """as_dict() returns correct values for relative coords"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.6, lry=0.7)
        
        d = box.as_dict()
        
        assert d["absolute_coords"] is False
        assert abs(d["ulx"] - 0.1) < 1e-6
        assert abs(d["uly"] - 0.2) < 1e-6
        assert abs(d["lrx"] - 0.6) < 1e-6
        assert abs(d["lry"] - 0.7) < 1e-6

    def test_as_dict_correct_types(self):
        """as_dict() returns correct types for fields"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, lrx=110, lry=70)
        
        d = box.as_dict()
        
        assert isinstance(d["absolute_coords"], bool)
        assert isinstance(d["ulx"], int)
        assert isinstance(d["uly"], int)
        assert isinstance(d["lrx"], int)
        assert isinstance(d["lry"], int)


    def test_to_np_array_dtype_float32(self):
        """to_np_array returns numpy array with float32 dtype"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        for mode in ["xyxy", "xywh", "poly"]:
            arr = box.to_np_array(mode=mode)
            assert arr.dtype == np.float32

    def test_to_np_array_shape_xyxy(self):
        """to_np_array('xyxy') returns shape (4,)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        arr = box.to_np_array(mode="xyxy")
        assert arr.shape == (4,)

    def test_to_np_array_shape_xywh(self):
        """to_np_array('xywh') returns shape (4,)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        arr = box.to_np_array(mode="xywh")
        assert arr.shape == (4,)

    def test_to_np_array_shape_poly(self):
        """to_np_array('poly') returns shape (8,)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        arr = box.to_np_array(mode="poly")
        assert arr.shape == (8,)


    def test_absolute_rounding_in_to_list(self):
        """Absolute coords are rounded in to_list"""
        # Create box with float coords that will be rounded internally
        box = BoundingBox(absolute_coords=True, ulx=10.6, uly=20.4, lrx=110.5, lry=70.3)
        
        lst = box.to_list(mode="xyxy")
        
        # All values should be ints (rounded)
        for val in lst:
            assert isinstance(val, int)

    def test_relative_raw_floats_in_to_list(self):
        """Relative coords preserve float precision in to_list"""
        box = BoundingBox(absolute_coords=False, ulx=0.123456, uly=0.234567, 
                         lrx=0.654321, lry=0.765432)
        
        lst = box.to_list(mode="xyxy")
        
        # All values should be floats
        for val in lst:
            assert isinstance(val, float)
