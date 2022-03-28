# -*- coding: utf-8 -*-
# File: tessocr.py

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
Tesseract OCR engine for text extraction
"""
import shlex
import subprocess
import sys
from errno import ENOENT
from itertools import groupby
from os import environ
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..utils.context import save_tmp_file, timeout_manager
from ..utils.detection_types import ImageType, Requirement
from ..utils.file_utils import TesseractNotFound, get_tesseract_requirement
from ..utils.metacfg import config_to_cli_str, set_config_by_yaml
from ..utils.settings import names
from .base import DetectionResult, ObjectDetector, PredictorBase

# copy and paste with some light modifications from https://github.com/madmaze/pytesseract/tree/master/pytesseract


class TesseractError(RuntimeError):
    """
    Tesseract Error
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__()
        self.status = status
        self.message = message
        self.args = (status, message)


def _subprocess_args() -> Dict[str, Any]:
    # See https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
    # for reference and comments.

    kwargs = {
        "stdin": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "startupinfo": None,
        "env": environ,
        "stdout": subprocess.PIPE,
    }

    return kwargs


def _input_to_cli_str(lang: str, config: str, nice: int, input_file_name: str, output_file_name_base: str) -> List[str]:
    """
    Generates a tesseract cmd as list of string with given inputs
    """
    cmd_args: List[str] = []

    if not sys.platform.startswith("win32") and nice != 0:
        cmd_args += ("nice", "-n", str(nice))

    cmd_args += ("tesseract", input_file_name, output_file_name_base, "-l", lang)

    if config:
        cmd_args += shlex.split(config)

    cmd_args.append("tsv")

    return cmd_args


def _run_tesseract(tesseract_args: List[str]) -> None:
    try:
        proc = subprocess.Popen(tesseract_args, **_subprocess_args())  # pylint: disable=R1732
    except OSError as error:
        if error.errno != ENOENT:
            raise error from error
        raise TesseractNotFound("Tesseract not found. Please install or add to your PATH.") from error

    with timeout_manager(proc, 0) as error_string:
        if proc.returncode:
            raise TesseractError(
                proc.returncode,
                " ".join(line for line in error_string.decode("utf-8").splitlines()).strip(),  # type: ignore
            )


def image_to_dict(image: ImageType, lang: str, config: str) -> Dict[str, List[Union[str, int, float]]]:
    """
    This is more or less :func:pytesseract.image_to_data with a dict as returned value.
    What happens under the hood is:

    - saving an image file
    - defining tesseracts command line
    - saving a temp .tsv file with predicted results
    - reading the .tsv file and returning the results as dict.

    Requires Tesseract 3.05+
    :param image: Image in np.array.
    :param lang: String of language
    :param config: string of configs
    :return: Dictionary with keys 'left', 'top', 'width', 'height' (bounding box coords), 'conf' (confidence), 'text'
             (captured text), 'block_num' (block number) and 'lin_num' (line number).
    """

    with save_tmp_file(image, "tess_") as (tmp_name, input_file_name):
        _run_tesseract(_input_to_cli_str(lang, config, 0, input_file_name, tmp_name))
        with open(tmp_name + ".tsv", "rb") as output_file:
            output = output_file.read().decode("utf-8")
        result: Dict[str, List[Union[str, int, float]]] = {}
        rows = [row.split("\t") for row in output.strip().split("\n")]
        if len(rows) < 2:
            return result
        header = rows.pop(0)
        length = len(header)
        if len(rows[-1]) < length:
            # Fixes bug that occurs when last text string in TSV is null, and
            # last row is missing a final cell in TSV file
            rows[-1].append("")

        str_col_idx = -1
        str_col_idx += length

        for i, head in enumerate(header):
            result[head] = []
            for row in rows:
                if len(row) <= i:
                    continue

                val = row[i]
                if row[i].isdigit() and i != -1 and head != "text":
                    val = int(row[i])  # type: ignore
                elif head == "text":
                    val = str(row[i])
                result[head].append(val)

        return result


def tesseract_line_to_detectresult(detect_result_list: List[DetectionResult]) -> List[DetectionResult]:
    """
    Generating text line DetectionResult based on Tesseract word grouping. It generates line bounding boxes from
    word bounding boxes.
    :param detect_result_list: A list of detection result
    :return: An extended list of detection result
    """

    line_detect_result: List[DetectionResult] = []
    for _, block_group_iter in groupby(detect_result_list, key=lambda x: x.block):
        block_group = []
        for _, line_group_iter in groupby(list(block_group_iter), key=lambda x: x.line):
            block_group.extend(list(line_group_iter))
        assert all(isinstance(detect_result.box, list) for detect_result in block_group)
        ulx = min(detect_result.box[0] for detect_result in block_group)  # type: ignore
        uly = min(detect_result.box[1] for detect_result in block_group)  # type: ignore
        lrx = max(detect_result.box[2] for detect_result in block_group)  # type: ignore
        lry = max(detect_result.box[3] for detect_result in block_group)  # type: ignore
        if block_group:
            line_detect_result.append(
                DetectionResult(
                    box=[ulx, uly, lrx, lry],
                    class_id=2,
                    class_name=names.C.LINE,
                    text=" ".join([detect_result.text for detect_result in block_group if detect_result.text]),
                )
            )
    if line_detect_result:
        detect_result_list.extend(line_detect_result)
    return detect_result_list


def predict_text(np_img: ImageType, supported_languages: str, text_lines: bool, config: str) -> List[DetectionResult]:
    """
    Calls tesseract directly with some given configs. Requires Tesseract to be installed.

    :param np_img: Image in np.array.
    :param supported_languages: To improve ocr extraction quality it is helpful to pre-select the language of the
                                detected text, if this in known in advance. Combinations are possible, e.g. "deu",
                                "fr+eng".
    :param text_lines: If True, it will return DetectionResults of Text lines as well.
    :param config: The config parameter passing to Tesseract. Consult also https://guides.nyu.edu/tesseract/usage
    :return: A list of tesseract extractions wrapped in DetectionResult
    """

    np_img = np_img.astype(np.uint8)
    results = image_to_dict(np_img, supported_languages, config)
    all_results = []

    for caption in zip(
        results["left"],
        results["top"],
        results["width"],
        results["height"],
        results["conf"],
        results["text"],
        results["block_num"],
        results["line_num"],
    ):
        if int(caption[4]) != -1:
            word = DetectionResult(
                box=[caption[0], caption[1], caption[0] + caption[2], caption[1] + caption[3]],
                score=caption[4] / 100,
                text=caption[5],
                block=str(caption[6]),
                line=str(caption[7]),
                class_id=1,
                class_name=names.C.WORD,
            )
            all_results.append(word)
    if text_lines:
        all_results = tesseract_line_to_detectresult(all_results)
    return all_results


class TesseractOcrDetector(ObjectDetector):  # pylint: disable=R0903
    """
    Text object detector based on Tesseracts OCR engine. Note that tesseract has to be installed separately.

    The current Tesseract release is 4.1.1. A version 5.xx can be integrated via direct installation at
    https://github.com/tesseract-ocr/tesseract. Building from source is necessary here.

    Documentation can be found here: https://tesseract-ocr.github.io/

    All configuration options that are available via pytesseract can be given via the configuration. The best overview
    can be found at https://pypi.org/project/pytesseract/.
    """

    def __init__(
        self,
        path_yaml: str,
        config_overwrite: Optional[List[str]] = None,
    ):
        """
        Set up the configuration which is stored in a yaml-file, that need to be passed through.

        :param path_yaml: The path to the yaml config
        :param config_overwrite: Overwrite config parameters defined by the yaml file with new values.
                                 E.g. ["oem=14"]
        """

        if config_overwrite is None:
            config_overwrite = []

        hyper_param_config = set_config_by_yaml(path_yaml)
        if len(config_overwrite):
            hyper_param_config.update_args(config_overwrite)

        self.path_yaml = path_yaml
        self.config_overwrite = config_overwrite
        self.config = hyper_param_config

    def predict(self, np_img: ImageType) -> List[DetectionResult]:
        """
        Transfer of a numpy array and call of pytesseract. Return of the detection results.

        :param np_img: image as numpy array
        :return: A list of DetectionResult
        """
        detection_results = predict_text(
            np_img,
            supported_languages=self.config.LANGUAGES,
            text_lines=self.config.LINES,
            config=config_to_cli_str(self.config, "LANGUAGES", "LINES"),
        )
        return detection_results

    @classmethod
    def get_requirements(cls) -> List[Requirement]:
        return [get_tesseract_requirement()]

    def clone(self) -> PredictorBase:
        return self.__class__(self.path_yaml, self.config_overwrite)
