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
Tesseract OCR engine
"""
from __future__ import annotations

import shlex
import string
import subprocess
import sys
from errno import ENOENT
from itertools import groupby
from os import environ, fspath
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from packaging.version import InvalidVersion, Version, parse

from ..utils.context import save_tmp_file, timeout_manager
from ..utils.error import DependencyError, TesseractError
from ..utils.file_utils import _TESS_PATH, get_tesseract_requirement
from ..utils.metacfg import config_to_cli_str, set_config_by_yaml
from ..utils.settings import LayoutType, ObjectTypes, PageType
from ..utils.types import PathLikeOrStr, PixelValues, Requirement
from ..utils.viz import viz_handler
from .base import DetectionResult, ImageTransformer, ModelCategories, ObjectDetector

# copy and paste with some light modifications from https://github.com/madmaze/pytesseract/tree/master/pytesseract


_LANG_CODE_TO_TESS_LANG_CODE = {
    "fre": "fra",
    "dut": "nld",
    "chi": "chi_sim",
    "cze": "ces",
    "per": "fas",
    "gre": "ell",
    "mac": "mkd",
    "rum": "ron",
    "arm": "hye",
    "geo": "kat",
    "war": "ceb",
    "glg": "gla",
    "slv": "slk",
    "alb": "nor",
    "nn": "eng",
}


def _subprocess_args() -> dict[str, Any]:
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


def _input_to_cli_str(lang: str, config: str, nice: int, input_file_name: str, output_file_name_base: str) -> list[str]:
    """
    Generates a tesseract cmd as list of string with given inputs
    """
    cmd_args: list[str] = []

    if not sys.platform.startswith("win32") and nice != 0:
        cmd_args += ("nice", "-n", str(nice))

    cmd_args += (fspath(_TESS_PATH), input_file_name, output_file_name_base, "-l", lang)

    if config:
        cmd_args += shlex.split(config)

    cmd_args.append("tsv")

    return cmd_args


def _run_tesseract(tesseract_args: list[str]) -> None:
    try:
        proc = subprocess.Popen(tesseract_args, **_subprocess_args())  # pylint: disable=R1732
    except OSError as error:
        if error.errno != ENOENT:
            raise error from error
        raise DependencyError("Tesseract not found. Please install or add to your PATH.") from error

    with timeout_manager(proc, 0) as error_string:
        if proc.returncode:
            raise TesseractError(
                proc.returncode,
                " ".join(line for line in error_string.decode("utf-8").splitlines()).strip(),  # type: ignore
            )


def get_tesseract_version() -> Version:
    """
    Returns:
        Version of the installed tesseract engine.
    """
    try:
        output = subprocess.check_output(
            ["tesseract", "--version"],
            stderr=subprocess.STDOUT,
            env=environ,
            stdin=subprocess.DEVNULL,
        )
    except OSError as error:
        raise DependencyError("Tesseract not found. Please install or add to your PATH.") from error

    raw_version = output.decode("utf-8")
    str_version, *_ = raw_version.lstrip(string.printable[10:]).partition(" ")
    str_version, *_ = str_version.partition("-")

    try:
        version = parse(str_version)
        assert version >= Version("3.05")
    except (AssertionError, InvalidVersion) as error:
        raise SystemExit(f'Invalid tesseract version: "{raw_version}"') from error

    return version


def image_to_angle(image: PixelValues) -> Mapping[str, str]:
    """
    Generating a tmp file and running Tesseract to get the orientation of the image.

     Args:
        image: Image an `np.array`
    Returns:
        A dict with keys 'Orientation in degrees' and 'Orientation confidence'.
    """
    with save_tmp_file(image, "tess_") as (tmp_name, input_file_name):
        _run_tesseract(_input_to_cli_str("osd", "--psm 0", 0, input_file_name, tmp_name))
        with open(tmp_name + ".osd", "rb") as output_file:
            output = output_file.read().decode("utf-8")

    return {
        key_value[0]: key_value[1] for key_value in (line.split(": ") for line in output.split("\n") if len(line) >= 2)
    }


def image_to_dict(image: PixelValues, lang: str, config: str) -> dict[str, list[Union[str, int, float]]]:
    """
    This is more or less `pytesseract.image_to_data` with a dict as returned value.
    What happens under the hood is:

    - saving an image file
    - defining tesseracts command line
    - saving a temp .tsv file with predicted results
    - reading the .tsv file and returning the results as dict.

    Note:
        Requires Tesseract or 3.05 or higher

    Args:
        image: Image in np.array.
        lang: String of language
        config: string of configs

    Returns:
        Dictionary with keys `left`, `top`, `width`, `height` (bounding box coords), `conf` (confidence), `text`
        (captured text), `block_num` (block number) and `lin_num` (line number).
    """

    with save_tmp_file(image, "tess_") as (tmp_name, input_file_name):
        _run_tesseract(_input_to_cli_str(lang, config, 0, input_file_name, tmp_name))
        with open(tmp_name + ".tsv", "rb") as output_file:
            output = output_file.read().decode("utf-8")
        result: dict[str, list[Union[str, int, float]]] = {}
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

        val: Union[str, int]
        for i, head in enumerate(header):
            result[head] = []
            for row in rows:
                if len(row) <= i:
                    continue

                val = row[i]
                if row[i].isdigit() and i != -1 and head != "text":
                    val = int(row[i])
                elif head == "text":
                    val = str(row[i])
                result[head].append(val)

        return result


def tesseract_line_to_detectresult(detect_result_list: list[DetectionResult]) -> list[DetectionResult]:
    """
    Generating text line `DetectionResult`s based on Tesseract word grouping. It generates line bounding boxes from
    word bounding boxes.

    Args:
        detect_result_list: A list of `DetectionResult`s

    Returns:
        An extended list of `DetectionResult`s
    """

    line_detect_result: list[DetectionResult] = []
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
                    class_name=LayoutType.LINE,
                    text=" ".join(
                        [detect_result.text for detect_result in block_group if isinstance(detect_result.text, str)]
                    ),
                )
            )
    if line_detect_result:
        detect_result_list.extend(line_detect_result)
    return detect_result_list


def predict_text(np_img: PixelValues, supported_languages: str, text_lines: bool, config: str) -> list[DetectionResult]:
    """
    Calls Tesseract directly with some given configs. Requires Tesseract to be installed.

    Args:
        np_img: Image in `np.array`.
        supported_languages: To improve OCR extraction quality it is helpful to pre-select the language of the
                             detected text, if this in known in advance. Combinations are possible, e.g. `deu`,
                             `fr+eng`.
        text_lines: If `True`, it will return `DetectionResult`s of text lines as well.
        config: The config parameter passing to Tesseract. Consult also <https://guides.nyu.edu/tesseract/usage>

    Returns:
        A list of Tesseract extractions wrapped in `DetectionResult`
    """

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
        score = float(caption[4])
        if int(score) != -1:
            word = DetectionResult(
                box=[caption[0], caption[1], caption[0] + caption[2], caption[1] + caption[3]],
                score=score / 100,
                text=caption[5],
                class_id=1,
                class_name=LayoutType.WORD,
            )
            all_results.append(word)
    if text_lines:
        all_results = tesseract_line_to_detectresult(all_results)
    return all_results


def predict_rotation(np_img: PixelValues) -> Mapping[str, str]:
    """
    Predicts the rotation of an image using the Tesseract OCR engine.

    Args:
        np_img: numpy array of the image

    Returns:
        A dictionary with keys 'Orientation in degrees' and 'Orientation confidence'
    """
    return image_to_angle(np_img)


class TesseractOcrDetector(ObjectDetector):
    """
    Text object detector based on Tesseracts OCR engine.

    Note:
        Tesseract has to be installed separately. <https://tesseract-ocr.github.io/>

    All configuration options that are available via pytesseract can be added to the configuration file:
    <https://pypi.org/project/pytesseract/.>

    Example:
        ```python
        tesseract_config_path = ModelCatalog.get_full_path_configs("dd/conf_tesseract.yaml")
        ocr_detector = TesseractOcrDetector(tesseract_config_path)

        detection_result = ocr_detector.predict(bgr_image_as_np_array)
        ```

    To use it within a pipeline

    Example:
        ```python
        tesseract_config_path = ModelCatalog.get_full_path_configs("dd/conf_tesseract.yaml")
        ocr_detector = TesseractOcrDetector(tesseract_config_path)

        text_extract = TextExtractionService(ocr_detector)
        pipe = DoctectionPipe([text_extract])

        df = pipe.analyze(path="path/to/dir")

        for dp in df:
            ...
        ```
    """

    def __init__(
        self,
        path_yaml: PathLikeOrStr,
        config_overwrite: Optional[list[str]] = None,
    ):
        """
        Set up the configuration which is stored in a `.yaml` file, that need to be passed through.

        Args:
            path_yaml: The path to the yaml config
            config_overwrite: Overwrite config parameters defined by the yaml file with new values.
                              E.g. `["oem=14"]`
        """
        self.name = self.get_name()
        self.model_id = self.get_model_id()

        if config_overwrite is None:
            config_overwrite = []

        hyper_param_config = set_config_by_yaml(path_yaml)
        if len(config_overwrite):
            hyper_param_config.update_args(config_overwrite)

        self.path_yaml = Path(path_yaml)
        self.config_overwrite = config_overwrite
        self.config = hyper_param_config

        if self.config.LINES:
            self.categories = ModelCategories(init_categories={1: LayoutType.WORD, 2: LayoutType.LINE})
        else:
            self.categories = ModelCategories(init_categories={1: LayoutType.WORD})

    def predict(self, np_img: PixelValues) -> list[DetectionResult]:
        """
        Transfer of a numpy array and call of pytesseract. Return of the detection results.

        Args:
            np_img: image as `np.array`

        Returns:
            A list of `DetectionResult`
        """

        return predict_text(
            np_img,
            supported_languages=self.config.LANGUAGES,
            text_lines=self.config.LINES,
            config=config_to_cli_str(self.config, "LANGUAGES", "LINES"),
        )

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_tesseract_requirement()]

    def clone(self) -> TesseractOcrDetector:
        return self.__class__(self.path_yaml, self.config_overwrite)

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)

    def set_language(self, language: ObjectTypes) -> None:
        """
        Pass a language to change the model selection. For runtime language selection.

        Args:
            language: One of the following: `fre`,`dut`,`chi`,`cze`,`per`,`gre`,`mac`,`rum`,`arm`,
                      `geo`,`war`,`glg`,`slv`,`alb`,`nn`.
        """
        self.config.LANGUAGES = _LANG_CODE_TO_TESS_LANG_CODE.get(language, language.value)

    @staticmethod
    def get_name() -> str:
        """Returns the name of the model"""
        return f"Tesseract_{get_tesseract_version()}"


class TesseractRotationTransformer(ImageTransformer):
    """
    The `TesseractRotationTransformer` is designed to handle image rotations.. It inherits from the `ImageTransformer`
    base class and implements methods for predicting and applying rotation transformations.

    The `predict` method determines the angle of the rotated image. It can only handle angles that are multiples of 90
    degrees. This method uses the Tesseract OCR engine to predict the rotation angle of an image.

    The `transform` method applies the predicted rotation to the image, effectively rotating the image backwards.
    This method uses either the Pillow library or OpenCV for the rotation operation, depending on the configuration.

    This class can be particularly useful in OCR tasks where the orientation of the text in the image matters.
    The class also provides methods for cloning itself and for getting the requirements of the Tesseract OCR system.

    Example:
        ```python
        transformer = TesseractRotationTransformer()
        detection_result = transformer.predict(np_img)
        rotated_image = transformer.transform(np_img, detection_result)
        ```
    """

    def __init__(self) -> None:
        self.name = fspath(_TESS_PATH) + "-rotation"
        self.categories = ModelCategories(init_categories={1: PageType.ANGLE})
        self.model_id = self.get_model_id()

    def transform_image(self, np_img: PixelValues, specification: DetectionResult) -> PixelValues:
        """
        Applies the predicted rotation to the image, effectively rotating the image backwards.
        This method uses either the Pillow library or OpenCV for the rotation operation, depending on the configuration.

        Args:
            np_img: The input image as a numpy array.
            specification: A `DetectionResult` object containing the predicted rotation angle.

        Returns:
            The rotated image as a numpy array.
        """
        return viz_handler.rotate_image(np_img, specification.angle)  # type: ignore

    def predict(self, np_img: PixelValues) -> DetectionResult:
        """
        Determines the angle of the rotated image. It can only handle angles that are multiples of 90 degrees.
        This method uses the Tesseract OCR engine to predict the rotation angle of an image.

        Args:
            np_img: The input image as a numpy array.
        Returns:
            A `DetectionResult` object containing the predicted rotation angle and confidence.
        """
        output_dict = predict_rotation(np_img)
        return DetectionResult(
            angle=float(output_dict["Orientation in degrees"]), score=float(output_dict["Orientation confidence"])
        )

    @classmethod
    def get_requirements(cls) -> list[Requirement]:
        return [get_tesseract_requirement()]

    def clone(self) -> TesseractRotationTransformer:
        return self.__class__()

    def get_category_names(self) -> tuple[ObjectTypes, ...]:
        return self.categories.get_categories(as_dict=False)
