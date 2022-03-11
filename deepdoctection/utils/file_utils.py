# -*- coding: utf-8 -*-
# File: file_utils.py
# Copyright (c)  The HuggingFace Team, the AllenNLP library authors.
# Licensed under the Apache License, Version 2.0 (the "License")


"""
Utilities for maintaining dependencies and dealing with external library packages. Parts of this file is adapted from
https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
"""

import importlib.util
import multiprocessing as mp
import string
import subprocess
from os import environ
from shutil import which
from typing import Tuple, Union

import importlib_metadata
from packaging import version

from .detection_types import Requirement
from .metacfg import AttrDict

# Tensorflow and Tensorpack dependencies
_TF_AVAILABLE = False

try:
    _TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
except ValueError:
    pass

_TF_ERR_MSG = "Tensorflow >=2.4.1 must be installed: https://www.tensorflow.org/install/gpu"


def tf_available() -> bool:
    """
    Returns True if TF is installed
    """
    return bool(_TF_AVAILABLE)


def get_tensorflow_requirement() -> Requirement:
    """
    Returns Tensorflow requirement
    """

    tf_requirement_satisfied = False
    if tf_available():
        candidates: Tuple[str, ...] = (
            "tensorflow",
            "tensorflow-cpu",
            "tensorflow-gpu",
            "tf-nightly",
            "tf-nightly-cpu",
            "tf-nightly-gpu",
            "intel-tensorflow",
            "intel-tensorflow-avx512",
            "tensorflow-rocm",
            "tensorflow-macos",
        )
        tf_version = "0.0"
        for pkg in candidates:
            try:
                tf_version = importlib_metadata.version(pkg)  # type: ignore
                break
            except importlib_metadata.PackageNotFoundError:
                pass
        _tf_version_available = tf_version != "0.0"
        if _tf_version_available:
            if version.parse(tf_version) < version.parse("2.4.1"):
                pass
            else:
                tf_requirement_satisfied = True

    return "tensorflow", tf_requirement_satisfied, _TF_ERR_MSG


_TP_AVAILABLE = importlib.util.find_spec("tensorpack") is not None
_TP_ERR_MSG = (
    "Tensorflow models all use the Tensorpack modeling API. Therefore, Tensorpack must be installed: "
    ">>make install-dd-tf"
)


def tensorpack_available() -> bool:
    """
    Returns True if Tensorpack is installed
    """
    return bool(_TP_AVAILABLE)


def get_tensorpack_requirement() -> Requirement:
    """
    Returns Tensorpack requirement
    """
    return "tensorpack", tensorpack_available(), _TP_ERR_MSG


# Pytorch related dependencies
_PYTORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_PYTORCH_ERR_MSG = "Pytorch must be installed: https://pytorch.org/get-started/locally/#linux-pip"


def pytorch_available() -> bool:
    """
    Returns True if Pytorch is installed
    """
    return bool(_PYTORCH_AVAILABLE)


def get_pytorch_requirement() -> Requirement:
    """
    Returns HF Pytorch requirement
    """
    return "torch", pytorch_available(), _PYTORCH_ERR_MSG


# Transformers
_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
_TRANSFORMERS_ERR_MSG = "Transformers must be installed: >>install-dd-pt"


def transformers_available() -> bool:
    """
    Returns True if HF Transformers is installed
    """
    return bool(_TRANSFORMERS_AVAILABLE)


def get_transformers_requirement() -> Requirement:
    """
    Returns HF Transformers requirement
    """
    return "transformers", transformers_available(), _TRANSFORMERS_ERR_MSG


# Detectron2 related requirements
_DETECTRON2_AVAILABLE = importlib.util.find_spec("detectron2") is not None
_DETECTRON2_ERR_MSG = "Detectron2 must be installed: >>install-dd-pt"


def detectron2_available() -> bool:
    """
    Returns True if Detectron2 is installed
    """
    return bool(_DETECTRON2_AVAILABLE)


def get_detectron2_requirement() -> Requirement:
    """
    Returns Detectron2 requirement
    """
    return "detectron2", detectron2_available(), _DETECTRON2_ERR_MSG


# Tesseract related dependencies

_TESS_AVAILABLE = which("tesseract") is not None
_TESS_ERR_MSG = "Tesseract >=4.0 must be installed: https://tesseract-ocr.github.io/tessdoc/Installation.html"


def tesseract_available() -> bool:
    """
    Returns True if Tesseract is installed
    """
    return bool(_TESS_AVAILABLE)


# copy paste from https://github.com/madmaze/pytesseract/blob/master/pytesseract/pytesseract.py


class TesseractNotFound(BaseException):
    """
    Exception class for Tesseract being not found
    """


def get_tesseract_version() -> Union[int, version.Version, version.LegacyVersion]:
    """
    Returns Version object of the Tesseract version. We need at least Tesseract 3.05
    """
    try:
        output = subprocess.check_output(
            ["tesseract", "--version"],
            stderr=subprocess.STDOUT,
            env=environ,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        raise TesseractNotFound() from OSError

    raw_version = output.decode("utf-8")
    str_version, *_ = raw_version.lstrip(string.printable[10:]).partition(" ")
    str_version, *_ = str_version.partition("-")

    current_version = version.parse(str_version)

    if current_version >= version.Version("4.0"):
        return current_version
    return 0


def get_tesseract_requirement() -> Requirement:
    """
    Returns Tesseract requirement. The minimum version must be 3.05
    """
    if get_tesseract_version():
        return "tesseract", True, _TESS_ERR_MSG
    return "tesseract", False, _TESS_ERR_MSG


# Poppler utils or resp. pdftoppm and pdftocairo for Linux platforms
_PDF_TO_PPM_AVAILABLE = which("pdftoppm") is not None
_PDF_TO_CAIRO_AVAILABLE = which("pdftocairo") is not None
_POPPLER_ERR_MSG = "Poppler is not found. Please check that Poppler is installed and it is added to your path"


def pdf_to_ppm_available() -> bool:
    """
    Returns True if pdftoppm is installed
    """
    return bool(_PDF_TO_PPM_AVAILABLE)


def pdf_to_cairo_available() -> bool:
    """
    Returns True if pdftocairo is installed
    """
    return bool(_PDF_TO_CAIRO_AVAILABLE)


class PopplerNotFound(BaseException):
    """
    Exception class for Poppler being not found
    """


def get_poppler_version() -> Union[int, version.Version, version.LegacyVersion]:
    """
    Returns Version object of the Poppler version. We need at least Tesseract 3.05
    """

    if pdf_to_ppm_available():
        command = "pdftoppm"
    elif pdf_to_cairo_available():
        command = "pdftocairo"
    else:
        return 0

    try:
        output = subprocess.check_output(
            [command, "-v"], stderr=subprocess.STDOUT, env=environ, stdin=subprocess.DEVNULL
        )
    except OSError:
        raise PopplerNotFound() from OSError

    raw_version = output.decode("utf-8")
    list_version = raw_version.split("\n", maxsplit=1)[0].split(" ")[-1].split(".")

    current_version = version.parse(".".join(list_version[:2]))

    return current_version


def get_poppler_requirement() -> Requirement:
    """
    Returns Poppler requirement. The minimum version is not required in our setting
    """
    if get_poppler_version():
        return "poppler", True, _POPPLER_ERR_MSG
    return "poppler", False, _POPPLER_ERR_MSG


# Pdfplumber.six related dependencies
_PDFPLUMBER_AVAILABLE = importlib.util.find_spec("pdfplumber") is not None
_PDFPLUMBER_ERR_MSG = "pdfplumber must be installed. >> pip install pdfplumber"


def pdfplumber_available() -> bool:
    """
    Returns True if pdfplumber is installed
    """
    return bool(_PDFPLUMBER_AVAILABLE)


def get_pdfplumber_requirement() -> Requirement:
    """
    Returns pdfplumber requirement.
    """
    return "pdfplumber", pdfplumber_available(), _PDFPLUMBER_ERR_MSG


# Textract related dependencies
_BOTO3_AVAILABLE = importlib.util.find_spec("boto3") is not None
_BOTO3_ERR_MSG = "Boto3 must be installed: >> pip install boto3"

_AWS_CLI_AVAILABLE = which("aws") is not None
_AWS_ERR_MSG = "AWS CLI must be installed https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"


def boto3_available() -> bool:
    """
    Returns True if Boto3 is installed
    """

    return bool(_BOTO3_AVAILABLE)


def get_boto3_requirement() -> Requirement:
    """
    Return Boto3 requirement
    """
    return "boto3", boto3_available(), _BOTO3_ERR_MSG


def aws_available() -> bool:
    """
    Returns True if AWS CLI is installed
    """
    return bool(_AWS_CLI_AVAILABLE)


def get_aws_requirement() -> Requirement:
    """
    Return AWS CLI requirement
    """
    return "aws", aws_available(), _AWS_ERR_MSG


_S = AttrDict()
_S.mp_context_set = False
_S.freeze()


def set_mp_spawn() -> None:
    """
    Sets multiprocessing method to "spawn".

    from https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/train.py:

          "spawn/forkserver" is safer than the default "fork" method and
          produce more deterministic behavior & memory saving
          However its limitation is you cannot pass a lambda function to subprocesses.
    """

    if not _S.mp_context_set:
        _S.freeze(False)
        mp.set_start_method("spawn")
        _S.mp_context_set = True
        _S.freeze()
