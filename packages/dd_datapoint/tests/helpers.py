# -*- coding: utf-8 -*-
"""
Test utilities for dd_datapoint tests
Minimal version to support standalone package tests
"""

from pathlib import Path


def get_test_path() -> Path:
    """
    Get path to test objects
    Note: This is a placeholder. Actual test assets should be copied to
    packages/dd_datapoint/tests/assets/ or packages/tests/assets/
    """
    # For now, point to original test_objects location
    # TODO: Copy needed test assets to package-local location
    return Path(__file__).parent.parent.parent.parent.parent / "_tests" / "test_objects"

