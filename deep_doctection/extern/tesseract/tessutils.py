# -*- coding: utf-8 -*-
# File: tessutils.py

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
Tesseract related utils
"""

from shutil import which
import importlib.util

from ...utils.detection_types import Requirement


_PYTESS_AVAILABLE = importlib.util.find_spec("pytesseract") is not None
_PYTESS_ERR_MSG = "Pytesseract must be installed: https://pypi.org/project/pytesseract/"

_TESS_AVAILABLE = which("tesseract") is not None
_TESS_ERR_MSG = "Tesseract >=4.0 must be installed: https://tesseract-ocr.github.io/tessdoc/Installation.html"


def py_tesseract_available() -> bool:
    """
    Returns True if Pytesseract is installed
    """
    return bool(_PYTESS_AVAILABLE)


def get_py_tesseract_requirement() -> Requirement:
    """
    Returns Pytesseract requirement
    """
    return "pytesseract", py_tesseract_available(), _PYTESS_ERR_MSG


def tesseract_available() -> bool:
    """
    Returns True if Tesseract is installed
    """
    return bool(_TESS_AVAILABLE)


def get_tesseract_requirement() -> Requirement:
    """
    Returns Tesseract requirement
    """
    return "tesseract", tesseract_available(), _TESS_ERR_MSG
