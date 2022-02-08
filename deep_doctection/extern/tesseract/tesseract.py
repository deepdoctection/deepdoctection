# -*- coding: utf-8 -*-
# File: tesseract.py

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
Module for calling  :func:`pytesseract.image_to_data`
"""
import subprocess
import sys
import shlex

from glob import iglob
from errno import ENOENT
from os import environ, remove
from os.path import normcase, normpath, realpath

from typing import List, Dict, Any
from tempfile import NamedTemporaryFile

import numpy as np

from contextlib import contextmanager
from cv2 import imwrite

from ...utils.detection_types import ImageType
from ...utils.file_utils import TesseractNotFound
from ..base import DetectionResult


__all__ = ["predict_text"]

# copy and paste with some light modifications from https://github.com/madmaze/pytesseract/tree/master/pytesseract


class TesseractError(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        self.message = message
        self.args = (status, message)


def _subprocess_args() -> Dict[str, Any]:
    # See https://github.com/pyinstaller/pyinstaller/wiki/Recipe-subprocess
    # for reference and comments.

    kwargs = {
        'stdin': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'startupinfo': None,
        'env': environ,
    }

    if hasattr(subprocess, 'STARTUPINFO'):
        kwargs['startupinfo'] = subprocess.STARTUPINFO()
        kwargs['startupinfo'].dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs['startupinfo'].wShowWindow = subprocess.SW_HIDE

    kwargs['stdout'] = subprocess.PIPE

    return kwargs


def _build_tesseract_args(lang: str, config: str, nice: int, input_file_name: str,
                          output_file_name_base: str):
    cmd_args: List[str] = []

    if not sys.platform.startswith('win32') and nice != 0:
        cmd_args += ('nice', '-n', str(nice))

    cmd_args += ('tesseract', input_file_name, output_file_name_base, "-l", lang)

    if config:
        cmd_args += shlex.split(config)

    cmd_args.append("tsv")

    return cmd_args


@contextmanager
def timeout_manager(proc, seconds=None):
    try:
        if not seconds:
            yield proc.communicate()[1]
            return

        try:
            _, error_string = proc.communicate(timeout=seconds)
            yield error_string
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.kill()
            proc.returncode = -1
            raise RuntimeError('Tesseract process timeout')
    finally:
        proc.stdin.close()
        proc.stdout.close()
        proc.stderr.close()


@contextmanager
def save(image):
    try:
        with NamedTemporaryFile(prefix='tess_', delete=False) as f:
            if isinstance(image, str):
                yield f.name, realpath(normpath(normcase(image)))
                return
            input_file_name = f.name + ".PNG"
            imwrite(input_file_name, image)
            yield f.name, input_file_name
    finally:
        for file_name in iglob(f.name + '*' if f.name else f.name):
            try:
                remove(file_name)
            except OSError as e:
                if e.errno != ENOENT:
                    raise e


def _run_tesseract(tesseract_args: List[str]):
    try:
        proc = subprocess.Popen(tesseract_args, **_subprocess_args())
    except OSError as e:
        if e.errno != ENOENT:
            raise e
        raise TesseractNotFound("Tesseract not found. Please install or add to your PATH.")

    with timeout_manager(proc, 0) as error_string:
        if proc.returncode:
            raise TesseractError(proc.returncode, ' '.join(
                line for line in error_string.decode("utf-8").splitlines()
            ).strip())


def image_to_dict(image: ImageType ,lang: str, config: str) -> Dict[str, List[str,int,float]]:
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

    with save(image) as (tmp_name, input_file_name):
        kwargs = {
            "lang": lang,
            "config": config,
            "nice": 0,
            "input_file_name": input_file_name,
            "output_file_name_base": tmp_name,
        }
        _run_tesseract(_build_tesseract_args(**kwargs))
        with open(kwargs["output_file_name_base"]+".tsv","rb") as output_file:
            output =  output_file.read().decode("utf-8")
        result = {}
        rows = [row.split("\t") for row in output.strip().split('\n')]
        if len(rows) < 2:
            return result
        header = rows.pop(0)
        length = len(header)
        if len(rows[-1]) < length:
            # Fixes bug that occurs when last text string in TSV is null, and
            # last row is missing a final cell in TSV file
            rows[-1].append('')

        str_col_idx = -1
        str_col_idx += length

        for i, head in enumerate(header):
            result[head] = list()
            for row in rows:
                if len(row) <= i:
                    continue

                val = row[i]
                if row[i].isdigit() and i != -1:
                    val = int(row[i])
                result[head].append(val)

        return result


def predict_text(np_img: ImageType, supported_languages: str, config: str) -> List[DetectionResult]:
    """
    Calls pytesseract :func:`pytesseract.image_to_data` with some given configs. It requires Tesseract to be installed.

    :param np_img: Image in np.array.
    :param supported_languages: To improve ocr extraction quality it is helpful to pre-select the language of the
                                detected text, if this in known in advance. Combinations are possible, e.g. "deu",
                                "fr+eng".
    :param config: The config parameter passing to Tesseract. Consult also https://guides.nyu.edu/tesseract/usage
    :return: A list of tesseract extractions wrapped in DetectionResult
    """

    np_img = np_img.astype(np.uint8)
    results = image_to_dict(np_img,lang=supported_languages,config=config)
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
            )
            all_results.append(word)

    return all_results
