# -*- coding: utf-8 -*-
# File: file_utils.py
# Copyright (c)  The HuggingFace Team, the AllenNLP library authors.
# Licensed under the Apache License, Version 2.0 (the "License")


"""
Utilities for maintaining dependencies and dealing with external library packages. Parts of this file is adapted from
<https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py>
"""
import importlib.util
import multiprocessing as mp
import string
import subprocess
import sys
from os import environ, path
from shutil import which
from types import ModuleType
from typing import Any, Union, no_type_check

import importlib_metadata
from packaging import version

from .error import DependencyError
from .logger import LoggingRecord, logger
from .metacfg import AttrDict
from .types import PathLikeOrStr, Requirement

_GENERIC_ERR_MSG = "Please check the required version either in the docs or in the setup file"

# Tensorflow and Tensorpack dependencies
_TF_AVAILABLE = False

try:
    _TF_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
except ValueError:
    pass

_TF_ERR_MSG = f"Tensorflow must be installed. {_GENERIC_ERR_MSG}"


def tf_available() -> bool:
    """
    Returns True if TF is installed
    """
    return bool(_TF_AVAILABLE)


def get_tf_version() -> str:
    """
    Determine the TF version which is installed
    """
    tf_version = "0.0"
    if tf_available():
        candidates: tuple[str, ...] = (
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

        for pkg in candidates:
            try:
                tf_version = importlib_metadata.version(pkg)
                break
            except importlib_metadata.PackageNotFoundError:
                pass
    return tf_version


def get_tensorflow_requirement() -> Requirement:
    """
    Returns Tensorflow requirement
    """

    tf_requirement_satisfied = False
    tf_version = get_tf_version()
    _tf_version_available = tf_version != "0.0"
    if _tf_version_available:
        if version.parse(tf_version) < version.parse("2.4.1"):
            pass
        else:
            tf_requirement_satisfied = True

    return "tensorflow", tf_requirement_satisfied, _TF_ERR_MSG


_TF_ADDONS_AVAILABLE = importlib.util.find_spec("tensorflow_addons") is not None
_TF_ADDONS_ERR_MSG = (
    "Tensorflow Addons must be installed. Please check the required version either in the docs or in the setup file."
    "Please note, that it has been announced, the this package will be deprecated in the near future."
)


def tf_addons_available() -> bool:
    """
    Returns True if tensorflow addons is installed
    """
    return bool(_TF_ADDONS_AVAILABLE)


def get_tf_addons_requirements() -> Requirement:
    """
    Returns Tensorflow Addons requirement
    """
    return "tensorflow-addons", tf_addons_available(), _TF_ADDONS_ERR_MSG


_TP_AVAILABLE = importlib.util.find_spec("tensorpack") is not None
_TP_ERR_MSG = f"Tensorpack must be installed. {_GENERIC_ERR_MSG}"


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
_PYTORCH_ERR_MSG = f"Pytorch must be installed. {_GENERIC_ERR_MSG}"


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


# lxml
_LXML_AVAILABLE = importlib.util.find_spec("lxml") is not None
_LXML_ERR_MSG = f"lxml must be installed. {_GENERIC_ERR_MSG}"


def lxml_available() -> bool:
    """
    Returns True if lxml is installed
    """
    return bool(_LXML_AVAILABLE)


def get_lxml_requirement() -> Requirement:
    """
    Returns lxml requirement
    """
    return "lxml", lxml_available(), _LXML_ERR_MSG


# apted
_APTED_AVAILABLE = importlib.util.find_spec("apted") is not None
_APTED_ERR_MSG = f"apted must be installed. {_GENERIC_ERR_MSG}"


def apted_available() -> bool:
    """
    Returns True if apted available
    """
    return bool(_APTED_AVAILABLE)


def get_apted_requirement() -> Requirement:
    """
    Returns APTED requirement
    """
    return "apted", apted_available(), _TRANSFORMERS_ERR_MSG


# distance
_DISTANCE_AVAILABLE = importlib.util.find_spec("distance") is not None
_DISTANCE_ERR_MSG = f"distance must be installed. {_GENERIC_ERR_MSG}"


def distance_available() -> bool:
    """
    Returns True if apted available
    """
    return bool(_DISTANCE_AVAILABLE)


def get_distance_requirement() -> Requirement:
    """
    Returns distance requirement
    """
    return "distance", distance_available(), _DISTANCE_ERR_MSG


# Transformers
_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
_TRANSFORMERS_ERR_MSG = f"transformers must be installed. {_GENERIC_ERR_MSG}"


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
_DETECTRON2_ERR_MSG = (
    "Detectron2 must be installed. Please follow the official installation instructions "
    "https://detectron2.readthedocs.io/en/latest/tutorials/install.html"
)


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
# Tesseract installation path
_TESS_PATH: PathLikeOrStr = "tesseract"
_TESS_ERR_MSG = (
    "Tesseract >=4.0 must be installed. Please follow the official installation instructions. "
    "https://tesseract-ocr.github.io/tessdoc/Installation.html"
)


def set_tesseract_path(tesseract_path: PathLikeOrStr) -> None:
    """Set the Tesseract path. If you have tesseract installed in Anaconda,
       you can use this function to set tesseract path.

    :param tesseract_path: Tesseract installation path.
    """

    global _TESS_AVAILABLE  # pylint: disable=W0603
    global _TESS_PATH  # pylint: disable=W0603

    tesseract_flag = which(tesseract_path)

    _TESS_AVAILABLE = False if tesseract_flag is not None else True  # pylint: disable=W0603,R1719

    _TESS_PATH = tesseract_path


def tesseract_available() -> bool:
    """
    Returns True if Tesseract is installed
    """
    return bool(_TESS_AVAILABLE)


# copy paste from https://github.com/madmaze/pytesseract/blob/master/pytesseract/pytesseract.py


def get_tesseract_version() -> Union[int, version.Version]:
    """
    Returns Version object of the Tesseract version. We need at least Tesseract 3.05
    """
    try:
        output = subprocess.check_output(
            [_TESS_PATH, "--version"],
            stderr=subprocess.STDOUT,
            env=environ,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        raise DependencyError(_TESS_ERR_MSG) from OSError

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
_POPPLER_ERR_MSG = "Poppler cannot be found. Please check that Poppler is installed and it is added to your path"


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


def get_poppler_version() -> Union[int, version.Version]:
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
        raise DependencyError(_POPPLER_ERR_MSG) from OSError

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
_PDFPLUMBER_ERR_MSG = f"pdfplumber must be installed. {_GENERIC_ERR_MSG}"


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


# pycocotools dependencies
_COCOTOOLS_AVAILABLE = importlib.util.find_spec("pycocotools") is not None
_COCOTOOLS_ERR_MSG = f"pycocotools must be installed. {_GENERIC_ERR_MSG}"


def cocotools_available() -> bool:
    """
    Returns True if pycocotools is installed
    """
    return bool(_COCOTOOLS_AVAILABLE)


def get_cocotools_requirement() -> Requirement:
    """
    Returns cocotools requirement.
    """
    return "pycocotools", cocotools_available(), _COCOTOOLS_ERR_MSG


# scipy dependency
_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


def scipy_available() -> bool:
    """
    Returns True if scipy is installed
    """
    return bool(_SCIPY_AVAILABLE)


# jdeskew dependency
_JDESKEW_AVAILABLE = importlib.util.find_spec("jdeskew") is not None
_JDESKEW_ERR_MSG = f"jdeskew must be installed. {_GENERIC_ERR_MSG}"


def jdeskew_available() -> bool:
    """
    Returns True if jdeskew is installed
    """
    return bool(_JDESKEW_AVAILABLE)


def get_jdeskew_requirement() -> Requirement:
    """
    Returns jdeskew requirement.
    """
    return "jdeskew", jdeskew_available(), _JDESKEW_ERR_MSG


# scikit-learn dependencies
_SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
_SKLEARN_ERR_MSG = f"scikit-learn must be installed. {_GENERIC_ERR_MSG}"


def sklearn_available() -> bool:
    """
    Returns True if sklearn is installed
    """
    return bool(_SKLEARN_AVAILABLE)


def get_sklearn_requirement() -> Requirement:
    """
    Returns sklearn requirement.
    """
    return "sklearn", sklearn_available(), _SKLEARN_ERR_MSG


# qpdf related dependencies
_QPDF_AVAILABLE = which("qpdf") is not None


def qpdf_available() -> bool:
    """
    Returns True if qpdf is installed
    """
    return bool(_QPDF_AVAILABLE)


# Textract related dependencies
_BOTO3_AVAILABLE = importlib.util.find_spec("boto3") is not None
_BOTO3_ERR_MSG = f"Boto3 must be installed. {_GENERIC_ERR_MSG}"

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


# DocTr related dependencies
_DOCTR_AVAILABLE = importlib.util.find_spec("doctr") is not None
_DOCTR_ERR_MSG = f"DocTr must be installed. {_GENERIC_ERR_MSG}"


def doctr_available() -> bool:
    """
    Returns True if doctr is installed
    """
    return bool(_DOCTR_AVAILABLE)


def get_doctr_requirement() -> Requirement:
    """
    Return Doctr requirement
    """
    if sys.platform == "darwin":
        if not get_poppler_version():
            return get_doctr_requirement()
        # don't know yet how to check whether pango gdk-pixbuf libffi are installed
        logger.info(
            LoggingRecord("package requires weasyprint. Check that poppler pango gdk-pixbuf libffi are installed")
        )
    return "doctr", doctr_available(), _DOCTR_ERR_MSG


# Fasttext related dependencies
_FASTTEXT_AVAILABLE = importlib.util.find_spec("fasttext") is not None
_FASTTEXT_ERR_MSG = f"fasttext must be installed. {_GENERIC_ERR_MSG}"


def fasttext_available() -> bool:
    """
    Returns True if fasttext is installed
    """
    return bool(_FASTTEXT_AVAILABLE)


def get_fasttext_requirement() -> Requirement:
    """
    Return Fasttext requirement
    """
    return "fasttext", fasttext_available(), _FASTTEXT_ERR_MSG


# Wandb related dependencies
_WANDB_AVAILABLE = importlib.util.find_spec("wandb") is not None
_WANDB_ERR_MSG = f"WandB must be installed. {_GENERIC_ERR_MSG}"


def wandb_available() -> bool:
    """
    Returns True if W&B package wandb is installed
    """
    return bool(_WANDB_AVAILABLE)


def get_wandb_requirement() -> Requirement:
    """
    Return WandB requirement
    """
    return "wandb", wandb_available(), _WANDB_ERR_MSG


_S = AttrDict()
_S.mp_context_set = False
_S.freeze()

# Image libraries: OpenCV and Pillow
# OpenCV
_CV2_AVAILABLE = importlib.util.find_spec("cv2") is not None
_CV2_ERR_MSG = f"OpenCV must be installed. {_GENERIC_ERR_MSG}"


def opencv_available() -> bool:
    """
    Returns True if OpenCV is installed
    """
    return bool(_CV2_AVAILABLE)


def get_opencv_requirement() -> Requirement:
    """
    Return OpenCV requirement
    """
    return "opencv", opencv_available(), _CV2_ERR_MSG


# Pillow
_PILLOW_AVAILABLE = importlib.util.find_spec("PIL") is not None
_PILLOW_ERR_MSG = f"pillow must be installed. {_GENERIC_ERR_MSG}"


def pillow_available() -> bool:
    """
    Returns True if Pillow is installed
    """
    return bool(_PILLOW_AVAILABLE)


def get_pillow_requirement() -> Requirement:
    """
    Return OpenCV requirement
    """
    return "pillow", pillow_available(), _PILLOW_ERR_MSG


# Pypdfium2
_PYPDFIUM2_AVAILABLE = importlib.util.find_spec("pypdfium2") is not None
_PYPDFIUM2_ERR_MSG = f"pypdfium2 must be installed. {_GENERIC_ERR_MSG}"


def pypdfium2_available() -> bool:
    """
    Returns True if pypdfium2 is installed
    """
    return bool(_PYPDFIUM2_AVAILABLE)


def get_pypdfium2_requirement() -> Requirement:
    """
    Return pypdfium2 requirement
    """
    return "pypdfium2", pypdfium2_available(), _PYPDFIUM2_ERR_MSG


# SpaCy
_SPACY_AVAILABLE = importlib.util.find_spec("spacy") is not None
_SPACY_ERR_MSG = f"SpaCy must be installed. {_GENERIC_ERR_MSG}"


def spacy_available() -> bool:
    """
    Returns True if SpaCy is installed
    """

    return bool(_SPACY_AVAILABLE)


def get_spacy_requirement() -> Requirement:
    """
    Return SpaCy requirement
    """
    return "spacy", spacy_available(), _SPACY_ERR_MSG


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


# Copy and paste from https://github.com/Layout-Parser/layout-parser/blob/main/src/layoutparser/file_utils.py


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    @no_type_check
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + sum(import_structure.values(), [])
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

        # Following [PEP 366](https://www.python.org/dev/peps/pep-0366/)
        # The __package__ variable should be set
        # https://docs.python.org/3/reference/import.html#__package__
        self.__package__ = self.__name__

    # Needed for autocompletion in an IDE
    @no_type_check
    def __dir__(self):
        return super().__dir__() + self.__all__

    @no_type_check
    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():  # pylint: disable=C0201
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    @no_type_check
    def _get_module(self, module_name: str):
        return importlib.import_module("." + module_name, self.__name__)

    @no_type_check
    def __reduce__(self):
        return self.__class__, (self._name, self.__file__, self._import_structure)
