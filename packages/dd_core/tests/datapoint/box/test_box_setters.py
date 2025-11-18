# -*- coding: utf-8 -*-
# File: test_bbox_setters.py

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
Comprehensive tests for BoundingBox property setters.
Tests absolute/relative mode setter behavior, positive extent preservation,
and validation error messages.
"""

import pytest
from dd_core.datapoint import BoundingBox
from dd_core.utils.error import BoundingBoxError


class TestBBoxSetters:
    """Tests for BoundingBox property setters in both absolute and relative modes"""

    # ==================== Absolute Mode Setters ====================

    def test_set_width_updates_lrx_absolute(self):
        """Setting width in absolute mode sets _lrx = round(width) directly (NOT ulx+width)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        assert box.width == 100
        assert box.lrx == 110

        box.width = 150
        assert box.lrx == 160
        assert box.width == 150
        assert box.ulx == 10

    def test_set_height_updates_lry_absolute(self):
        """Setting height in absolute mode sets _lry = round(height) directly (NOT uly+height)"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        assert box.height == 50
        assert box.lry == 70

        box.height = 80
        assert box.lry == 100
        assert box.height == 80
        assert box.uly == 20

    def test_set_width_with_rounding_absolute(self):
        """Setting width with float value rounds using rounding"""
        box = BoundingBox(absolute_coords=True, ulx=0, uly=0, width=100, height=100)

        box.width = 50.5
        assert box.lrx == 0 + 51
        
        box.width = 51.5
        assert box.lrx == 0 + 52

    def test_set_height_with_rounding_absolute(self):
        """Setting height with float value rounds using rounding"""
        box = BoundingBox(absolute_coords=True, ulx=0, uly=0, width=100, height=100)

        box.height = 50.5
        assert box._lry == 0 + 51
        
        box.height = 51.5
        assert box._lry == 0 + 52

    def test_set_width_zero_rejected_absolute(self):
        """Setting width to zero is rejected in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.width = 0

    def test_set_height_zero_rejected_absolute(self):
        """Setting height to zero is rejected in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.height = 0

    def test_set_width_negative_rejected_absolute(self):
        """Setting width to negative is rejected in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.width = -10

    def test_set_height_negative_rejected_absolute(self):
        """Setting height to negative is rejected in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.height = -10

    def test_set_lrx_preserves_positive_width_absolute(self):
        """Setting lrx must maintain lrx > ulx in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=50, uly=20, width=100, height=50)

        box.lrx = 200
        assert box.lrx == 200
        assert box.width == 150
        
        # Invalid: lrx <= ulx
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.lrx = 50  # Would make width = 0


    def test_set_lry_preserves_positive_height_absolute(self):
        """Setting lry must maintain lry > uly in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=50, width=100, height=100)

        box.lry = 200
        assert box.lry == 200
        assert box.height == 150
        
        # Invalid: lry <= uly
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.lry = 50  # Would make height = 0


    def test_set_ulx_preserves_positive_width_absolute(self):
        """Setting ulx must maintain lrx > ulx in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=50, uly=20, lrx=150, lry=70)

        box.ulx = 100
        assert box.ulx == 100
        assert box.width == 50
        
        # Invalid: ulx >= lrx
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.ulx = 160  # Would make width < 0

    def test_set_uly_preserves_positive_height_absolute(self):
        """Setting uly must maintain lry > uly in absolute mode"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=50, lrx=110, lry=150)

        box.uly = 100
        assert box.uly == 100
        assert box.height == 50
        
        # Invalid: uly >= lry
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.uly = 160

    # ==================== Relative Mode Setters ====================

    def test_set_width_updates_lrx_relative(self):
        """Setting width in relative mode updates internal lrx using SCALE_FACTOR"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        initial_ulx_internal = box._ulx

        box.width = 0.6
        expected_lrx_internal = initial_ulx_internal + round(0.6 * 10**8)
        assert box._lrx == expected_lrx_internal
        assert abs(box.width - 0.6) < 1e-6

    def test_set_height_updates_lry_relative(self):
        """Setting height in relative mode updates internal lry using SCALE_FACTOR"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        initial_uly_internal = box._uly

        box.height = 0.4
        expected_lry_internal = initial_uly_internal + round(0.4 * 10**8)
        assert box._lry == expected_lry_internal
        assert abs(box.height - 0.4) < 1e-6

    def test_set_width_zero_rejected_relative(self):
        """Setting width to zero is rejected in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.width = 0.0

    def test_set_height_zero_rejected_relative(self):
        """Setting height to zero is rejected in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.height = 0.0

    def test_set_width_negative_rejected_relative(self):
        """Setting width to negative is rejected in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.width = -0.1

    def test_set_height_negative_rejected_relative(self):
        """Setting height to negative is rejected in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, width=0.5, height=0.3)
        
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.height = -0.1

    def test_set_lrx_preserves_positive_width_relative(self):
        """Setting lrx must maintain lrx > ulx in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.1, lrx=0.7, lry=0.4)

        box.lrx = 0.9
        assert abs(box.lrx - 0.9) < 1e-6
        assert box.width > 0
        assert abs(box.width - 0.7) < 1e-6
        
        # Invalid: lrx <= ulx
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.lrx = 0.2

    def test_set_lry_preserves_positive_height_relative(self):
        """Setting lry must maintain lry > uly in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.4, lry=0.7)

        box.lry = 0.9
        assert abs(box.lry - 0.9) < 1e-6
        assert box.height > 0
        assert abs(box.height - 0.7) < 1e-6
        
        # Invalid: lry <= uly
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.lry = 0.2

    def test_set_ulx_preserves_positive_width_relative(self):
        """Setting ulx must maintain lrx > ulx in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.2, uly=0.1, lrx=0.7, lry=0.4)
        
        # Valid update
        box.ulx = 0.1
        assert abs(box.ulx - 0.1) < 1e-6
        assert box.width > 0
        assert abs(box.width - 0.6) < 1e-6
        
        # Invalid: ulx >= lrx (would violate width > 0)
        with pytest.raises(BoundingBoxError, match="width must be >0"):
            box.ulx = 0.7

    def test_set_uly_preserves_positive_height_relative(self):
        """Setting uly must maintain lry > uly in relative mode"""
        box = BoundingBox(absolute_coords=False, ulx=0.1, uly=0.2, lrx=0.4, lry=0.7)
        
        # Valid update
        box.uly = 0.1
        assert abs(box.uly - 0.1) < 1e-6
        assert box.height > 0
        assert abs(box.height - 0.6) < 1e-6
        
        # Invalid: uly >= lry
        with pytest.raises(BoundingBoxError, match="height must be >0"):
            box.uly = 0.7  # Would make height = 0 (after rounding)

    # ==================== Cross-mode Setter Tests ====================

    def test_setter_error_messages_contain_coords(self):
        """Error messages should include coordinate values for debugging"""
        box = BoundingBox(absolute_coords=True, ulx=100, uly=200, width=50, height=50)
        
        with pytest.raises(BoundingBoxError, match="lrx.*ulx"):
            box.lrx = 50  # lrx < ulx

    def test_multiple_setter_operations_maintain_invariants(self):
        """Multiple setter operations should maintain box validity"""
        box = BoundingBox(absolute_coords=True, ulx=10, uly=20, width=100, height=50)
        
        # Chain of valid operations
        box.width = 200
        assert box.lrx == 210
        assert box.width == 200

        box.height = 100
        assert box.lry == 120
        assert box.height == 100

        box.ulx = 50
        assert box.ulx == 50
        assert box.width == 160

        box.uly = 50
        assert box.uly == 50
        assert box.height == 70

