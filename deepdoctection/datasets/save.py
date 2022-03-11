# -*- coding: utf-8 -*-
# File: save.py

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
Module for saving
"""

from typing import Optional

from ..dataflow import DataFlow, MapData, SerializerJsonlines  # type: ignore
from ..datapoint.image import Image
from ..utils.detection_types import JsonDict


def dataflow_to_jsonl(df: DataFlow, path: str, file_name: str, max_datapoints: Optional[int] = None) -> None:
    """
    Save a dataflow consisting of :class:`datapoint.Image` to a jsonl file. Each image will be dumped into a separate
    JSON object.

    :param df: Input dataflow
    :param path: Path to save the file to
    :param file_name: File name of the .jsonl file
    :param max_datapoints: Will stop saving after dumping max_datapoint images.
    """

    def image_to_json(dp: Image) -> JsonDict:
        return dp.get_export()

    df = MapData(df, image_to_json)
    SerializerJsonlines.save(df, path, file_name, max_datapoints)
