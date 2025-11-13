# -*- coding: utf-8 -*-
"""
Test utilities for dd_datapoint tests
Minimal version to support standalone package tests
"""

from pathlib import Path

import pytest

from dd_datapoint.utils.object_types import ObjectTypes, object_types_registry, update_black_list
from dd_datapoint.utils.viz import viz_handler

# Register custom pytest plugins (relative to tests/ directory)
pytest_plugins = ["tests._pytest_plugins.dd_plugin"]


class ObjectTestType(ObjectTypes):
    """Object type members for testing purposes"""

    foo = "foo"
    FOO = "FOO"
    bak = "bak"
    BAK = "BAK"
    BAK_1 = "BAK_1"
    BAK_11 = "BAK_11"
    BAK_12 = "BAK_12"
    BAK_21 = "BAK_21"
    BAK_22 = "BAK_22"
    BLA = "BLA"
    BLU = "BLU"
    FOO_1 = "FOO_1"
    FOO_2 = "FOO_2"
    FOO_3 = "FOO_3"
    BLI = "BLI"
    FOOBAK = "FOOBAK"
    Test = "TEST"
    TEST_SUMMARY = "TEST_SUMMARY"
    baz = "baz"
    BAZ = "BAZ"
    b_foo = "B-FOO"
    i_foo = "I-FOO"
    sub = "sub"
    sub_2 = "sub_2"
    one = "1"
    two = "2"
    three = "3"
    four = "4"
    five = "5"
    report_date = "report_date"
    umbrella = "umbrella"
    report_type = "report_type"
    fund_name = "fund_name"


def get_test_path() -> Path:
    """
    Get path to test objects
    Note: This is a placeholder. Actual test assets should be copied to
    packages/dd_datapoint/tests/assets/ or packages/tests/assets/
    """
    # For now, point to original test_objects location
    # TODO: Copy needed test assets to package-local location
    return Path(__file__).parent.parent.parent.parent.parent / "_tests" / "test_objects"


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("ObjectTestType")(ObjectTestType)
    for item in ObjectTestType:
        update_black_list(item.value)
    viz_handler.refresh()
