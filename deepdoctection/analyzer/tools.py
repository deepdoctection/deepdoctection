# -*- coding: utf-8 -*-
# File: tools.py

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
Module for tools around processing documents with Pipelines.
"""

import json
import os

from ..dataflow.base import DataFlow
from ..dataflow.common import MapData
from ..dataflow.custom_serialize import SerializerJsonlines
from ..datapoint.view import Page
from ..utils.detection_types import Pathlike
from ..utils.utils import is_file_extension


def load_page(path: Pathlike) -> Page:
    """
    Load a json file and generate a page object.

    :param path: Path to load from
    :return: A page object
    """

    assert os.path.isfile(path), path
    file_name = os.path.split(path)[1]
    assert is_file_extension(file_name, ".json"), file_name

    with open(path, "r", encoding="UTF-8") as file:
        page_dict = json.load(file)

    return Page.from_dict(**page_dict)


def load_document(path: Pathlike) -> DataFlow:
    """
    Load a parsed document from a .jsonl file and generate a DataFlow that can be streamed.

    :param path: Path to a .jsonl file
    :return: DataFlow
    """
    df: DataFlow
    df = SerializerJsonlines.load(path)
    df = MapData(df, Page.from_dict)
    return df
