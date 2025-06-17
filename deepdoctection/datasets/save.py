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
Saving samples from a DataFlow
"""

import json
import os
from pathlib import Path
from typing import Optional

from ..dataflow import DataFlow, MapData, SerializerJsonlines
from ..datapoint.convert import convert_b64_to_np_array
from ..datapoint.image import Image
from ..utils.fs import mkdir_p
from ..utils.types import ImageDict, PathLikeOrStr
from ..utils.viz import viz_handler


def dataflow_to_json(
    df: DataFlow,
    path: PathLikeOrStr,
    single_files: bool = False,
    file_name: Optional[str] = None,
    max_datapoints: Optional[int] = None,
    save_image_in_json: bool = True,
    highest_hierarchy_only: bool = False,
) -> None:
    """
    Save a dataflow consisting of `datapoint.Image` to a `jsonl` file. Each image will be dumped into a separate
    `JSON` object.

    Args:
        df: Input dataflow
        path: Path to save the file(s) to
        single_files: Will save image results to a single `JSON` file. If False all images of the dataflow will be
                      dumped into a single `.jsonl` file.
        file_name: file name, only needed for `jsonl` files
        max_datapoints: Will stop saving after dumping max_datapoint images.
        save_image_in_json: Will save the image to the `JSON` object
        highest_hierarchy_only: If `True` it will remove all image attributes of `ImageAnnotation`s
    """
    path = Path(path)
    if single_files:
        mkdir_p(path)
    if not save_image_in_json:
        mkdir_p(path / "image")
    if highest_hierarchy_only:

        def _remove_hh(dp: Image) -> Image:
            dp.remove_image_from_lower_hierarchy()
            return dp

        df = MapData(df, _remove_hh)
    df = MapData(df, lambda dp: dp.as_dict())

    def _path_to_str(dp: ImageDict) -> ImageDict:
        dp["location"] = os.fspath(dp["location"])
        return dp

    df = MapData(df, _path_to_str)
    df.reset_state()
    if single_files:
        for idx, dp in enumerate(df):
            if idx == max_datapoints:
                break
            target_file = path / (dp["file_name"].split(".")[0] + ".json")
            if not save_image_in_json:
                target_file_png = path / "image" / (dp["file_name"].split(".")[0] + ".png")
                image = dp.pop("_image")
                image = convert_b64_to_np_array(image)

                viz_handler.write_image(str(target_file_png), image)

            with open(target_file, "w", encoding="UTF-8") as file:
                json.dump(dp, file, indent=2)

    else:
        if not file_name:
            raise ValueError("If single_files is set to False must pass a valid file name for .jsonl file")
        SerializerJsonlines.save(df, path, file_name, max_datapoints)
