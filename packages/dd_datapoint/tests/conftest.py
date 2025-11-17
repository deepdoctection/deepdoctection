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
#pytest_plugins = ["tests._shared.factory"]


class ObjectTestType(ObjectTypes):
    """Object type members for testing purposes"""

    TEST_CAT_1 = "test_cat_1"
    TEST_CAT_2 = "test_cat_2"
    TEST_CAT_3 = "test_cat_3"
    TEST_CAT_4 = "test_cat_4"
    SUB_CAT_1 = "sub_cat_1"
    SUB_CAT_2 = "sub_cat_2"
    SUB_CAT_3 = "sub_cat_3"
    RELATIONSHIP_1 = "relationship_1"
    RELATIONSHIP_2 = "relationship_2"
    NON_EXISTENT = "non_existent"



@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("ObjectTestType")(ObjectTestType)
    for item in ObjectTestType:
        update_black_list(item.value)
    viz_handler.refresh()
