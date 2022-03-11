# -*- coding: utf-8 -*-
# File: context.py

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
Some useful contextmanagers for various tasks
"""

import subprocess
from contextlib import contextmanager
from errno import ENOENT
from glob import iglob
from os import path, remove
from tempfile import NamedTemporaryFile
from typing import Iterator, Optional, Tuple, Union

import numpy as np
from cv2 import imwrite

from .detection_types import ImageType

__all__ = ["timeout_manager", "save_tmp_file"]


@contextmanager
def timeout_manager(proc, seconds: Optional[int] = None) -> Iterator[str]:  # type: ignore
    """
    Manager for time handling while Tesseract being called
    :param proc: process
    :param seconds: seconds to wait
    """
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
            raise RuntimeError("Tesseract process timeout")  # pylint: disable=W0707
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


@contextmanager
def save_tmp_file(image: Union[str, ImageType, bytes], prefix: str) -> Iterator[Tuple[str, str]]:
    """
    Save image temporarily and handle the clean-up once not necessary anymore
    :param image: image as string or numpy array
    :param prefix: prefix of the temp file name
    """
    try:
        with NamedTemporaryFile(prefix=prefix, delete=False) as file:
            if isinstance(image, str):
                yield file.name, path.realpath(path.normpath(path.normcase(image)))
                return
            if isinstance(image, (np.ndarray, np.generic)):
                input_file_name = file.name + ".PNG"
                imwrite(input_file_name, image)
                yield file.name, input_file_name
            if isinstance(image, bytes):
                input_file_name = file.name
                file.write(image)
                file.flush()
                yield file.name, input_file_name
    finally:
        for file_name in iglob(file.name + "*" if file.name else file.name):
            try:
                remove(file_name)
            except OSError as error:
                if error.errno != ENOENT:
                    raise error
