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

from dataflow.dataflow import DataFlow, MapData  # type:ignore

from ..dataflow.custom_serialize import SerializerJsonlines
from ..datapoint.doc import Page
from ..mapper.pagestruct import page_dict_to_page
from ..utils.fs import is_file_extension


def load_page(path: str) -> Page:
    """
    Load a json file and generate a page object.

    :param path: Path to load from
    :return: A page object
    """

    assert os.path.isfile(path), path
    file = os.path.split(path)[1]
    assert is_file_extension(file, ".json"), file

    with open(path, "r", encoding="UTF-8") as file:  # type: ignore
        page_dict = json.load(file)  # type: ignore

    return page_dict_to_page(page_dict)


def load_document(path: str) -> DataFlow:
    """
    Load a parsed document from a .jsonl file and generate a DataFlow that can be streamed.

    :param path: Path to a .jsonl file
    :return: DataFlow
    """

    df = SerializerJsonlines.load(path)
    df = MapData(df, page_dict_to_page)
    return df
