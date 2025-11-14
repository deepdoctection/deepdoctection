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

    PERSON = "person"
    NUMBER = "number"
    FOO = "foo"
    BAR = "bar"
    TEST = "test"
    SUB_CATEGORY_NAME = "sub_category_name"
    SUB_CATEGORY_NAME_2 = "sub_category_name_2"
    SUB_CATEGORY_NAME_3 = "sub_category_name_3"
    RELATIONSHIP_NAME = "relationship_name"



@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart() -> None:
    """Pre configuration before testing starts"""
    object_types_registry.register("ObjectTestType")(ObjectTestType)
    for item in ObjectTestType:
        update_black_list(item.value)
    viz_handler.refresh()
