# -*- coding: utf-8 -*-
# File: env_info.py

# Copyright 2023 Dr. Janis Meyer. All rights reserved.
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
Some useful function for collecting environment information.

This is also the place where we give an overview of the important environment variables.

`USE_TENSORFLOW
USE_PYTORCH
USE_CUDA
USE_MPS`

are responsible for selecting the predictors based on the installed DL framework and available devices.
It is not recommended to touch them.

`USE_DD_PILLOW
USE_DD_OPENCV`

decide what image processing library the `viz_handler` should use. The default library is PIL and OpenCV need
to be installed separately. However, if both libraries have been detected `viz_handler` will opt for OpenCV.
Use the variables to let choose `viz_handler` according to your preferences.

`HF_CREDENTIALS`

will be used by the `ModelDownloadManager` to pass your credentials if you have a model registered that resides in a
private repo.

`MODEL_CATALOG`

can store an (absolute) path to a `.jsonl` file.

"""

import ast
import importlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import List, Literal, Optional, Tuple

import numpy as np
from tabulate import tabulate

from .file_utils import (
    apted_available,
    aws_available,
    boto3_available,
    cocotools_available,
    distance_available,
    doctr_available,
    fasttext_available,
    get_poppler_version,
    get_tesseract_version,
    jdeskew_available,
    lxml_available,
    opencv_available,
    pdf_to_cairo_available,
    pdf_to_ppm_available,
    pdfplumber_available,
    pytorch_available,
    qpdf_available,
    scipy_available,
    sklearn_available,
    tensorpack_available,
    tesseract_available,
    tf_available,
    transformers_available,
    wandb_available,
)
from .logger import LoggingRecord, logger

__all__ = [
    "collect_torch_env",
    "collect_env_info",
    "get_device",
    "auto_select_lib_and_device",
    "auto_select_viz_library",
]

# pylint: disable=import-outside-toplevel


def collect_torch_env() -> str:
    """Wrapper for torch.utils.collect_env.get_pretty_env_info"""
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_installed_dependencies(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Collect installed dependencies for all third party libraries.

    :param data: A list of tuples to dump all collected package information such as the name and the version
    :return: A list of tuples containing the name of the library and the version (if available)
    """

    if tensorpack_available():
        import tensorpack  # pylint: disable=E0401

        data.append(("Tensorpack", tensorpack.__version__))
    else:
        data.append(("Tensorpack", "None"))

    if opencv_available():
        import cv2

        data.append(("OpenCV", cv2.__version__))
    else:
        data.append(("OpenCV", "None"))

    if lxml_available():
        import lxml

        data.append(("Lxml", lxml.__version__))  # type: ignore
    else:
        data.append(("Lxml", "None"))

    if apted_available():
        data.append(("Apted", "available"))
    else:
        data.append(("Apted", "None"))

    if distance_available():
        data.append(("Distance", "available"))
    else:
        data.append(("Distance", "None"))

    if transformers_available():
        import transformers

        data.append(("Transformers", transformers.__version__))
    else:
        data.append(("Transformers", "None"))

    if tesseract_available():
        data.append(("Tesseract", str(get_tesseract_version())))
    else:
        data.append(("Tesseract", "None"))

    if pdf_to_ppm_available() or pdf_to_cairo_available():
        data.append(("Poppler", str(get_poppler_version())))
    else:
        data.append(("Poppler", "None"))

    if pdfplumber_available():
        import pdfplumber

        data.append(("Pdfplumber", pdfplumber.__version__))
    else:
        data.append(("Pdfplumber", "None"))

    if cocotools_available():
        data.append(("Pycocotools", "available"))
    else:
        data.append(("Pycocotools", "None"))

    if scipy_available():
        import scipy  # type: ignore

        data.append(("Scipy", scipy.__version__))
    else:
        data.append(("Scipy", "None"))

    if jdeskew_available():
        data.append(("Jdeskew", "available"))
    else:
        data.append(("Jdeskew", "None"))

    if sklearn_available():
        import sklearn  # type: ignore # pylint: disable=E0401

        data.append(("Sklearn", sklearn.__version__))
    else:
        data.append(("Sklearn", "None"))

    if qpdf_available():
        data.append(("Qpdf", "available"))
    else:
        data.append(("Qpdf", "None"))

    if boto3_available():
        import boto3  # type: ignore

        data.append(("Boto3", boto3.__version__))
    else:
        data.append(("Boto3", "None"))

    if aws_available():
        data.append(("Awscli", "available"))
    else:
        data.append(("Awscli", "None"))

    if doctr_available():
        import doctr

        data.append(("Doctr", doctr.__version__))
    else:
        data.append(("Doctr", "None"))

    if fasttext_available():
        data.append(("Fasttext", "available"))
    else:
        data.append(("Fasttext", "None"))

    if wandb_available():
        import wandb

        data.append(("Wandb", wandb.__version__))
    else:
        data.append(("Wandb", "None"))

    return data


def detect_compute_compatibility(cuda_home: Optional[str], so_file: Optional[str]) -> str:
    """
    Detect the compute compatibility of a CUDA library.

    :param cuda_home: The path to the CUDA installation
    :param so_file: The path to the shared object file
    :return: The compute compatibility of the CUDA library
    """
    try:
        cuobjdump = os.path.join(cuda_home, "bin", "cuobjdump")  # type: ignore
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(f"'{cuobjdump}' --list-elf '{so_file}'", shell=True)
            output = output.decode("utf-8").strip().split("\n")  # type: ignore
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]  # type: ignore
                arch.append(".".join(line))  # type: ignore
            arch = sorted(set(arch))
            return ", ".join(arch)
        return str(so_file) + "; cannot find cuobjdump"
    except Exception:  # pylint: disable=W0718
        # unhandled failure
        return str(so_file)


# Copied from https://github.com/tensorpack/tensorpack/blob/master/tensorpack/tfutils/collect_env.py
def tf_info(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Returns a list of (key, value) pairs containing tensorflow information.

    :param data: A list of tuples to dump all collected package information such as the name and the version
    :return: A list of tuples containing all the collected information
    """
    if tf_available():
        import tensorflow as tf  # type: ignore # pylint: disable=E0401

        data.append(("Tensorflow", tf.__version__))
    else:
        data.append(("Tensorflow", "None"))
        return data

    from tensorflow.python.platform import build_info  # type: ignore # pylint: disable=E0401

    try:
        for key, value in list(build_info.build_info.items()):
            if key == "cuda_version":
                data.append(("TF built with CUDA", value))
            elif key == "cudnn_version":
                data.append(("TF built with CUDNN", value))
            elif key == "cuda_compute_capabilities":
                data.append(("TF compute capabilities", ",".join([k.replace("compute_", "") for k in value])))
        return data
    except AttributeError:
        pass
    try:
        data.append(("TF built with CUDA", build_info.cuda_version_number))
        data.append(("TF built with CUDNN", build_info.cudnn_version_number))
    except AttributeError:
        pass
    return data


# Heavily inspired by https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/collect_env.py
def pt_info(data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Returns a list of (key, value) pairs containing Pytorch information.

    :param data: A list of tuples to dump all collected package information such as the name and the version
    :return: A list of tuples containing all the collected information
    """

    if pytorch_available():
        import torch
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    has_mps = torch.backends.mps.is_available()

    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    if has_cuda and CUDA_HOME is not None:
        try:
            nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
            nvcc = subprocess.check_output(f"'{nvcc}' -V", shell=True)  # type: ignore
            nvcc = nvcc.decode("utf-8").strip().rsplit("\n", maxsplit=1)[-1]  # type: ignore
        except subprocess.SubprocessError:
            nvcc = "Not found"
        data.append(("CUDA compiler", nvcc))

    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", str(torch.version.debug)))

    if not has_gpu:
        has_gpu_text = "No: torch.cuda.is_available() == False"
    else:
        has_gpu_text = "Yes"
    data.append(("GPU available", has_gpu_text))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            msg = " - invalid!" if not (ROCM_HOME and os.path.isdir(ROCM_HOME)) else ""
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))
        else:
            try:
                from torch.utils.collect_env import get_nvidia_driver_version
                from torch.utils.collect_env import run as _run

                data.append(("Driver version", get_nvidia_driver_version(_run)))
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            msg = " - invalid!" if not (CUDA_HOME and os.path.isdir(CUDA_HOME)) else ""
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))

    mps_build = "No: torch.backends.mps.is_built() == False"
    if not has_mps:
        has_mps_text = "No: torch.backends.mps.is_available() == False"
    else:
        has_mps_text = "Yes"
        mps_build = str(torch.backends.mps.is_built())

    data.append(("MPS available", has_mps_text))
    data.append(("MPS available", mps_build))

    try:
        import torchvision  # type: ignore

        data.append(
            (
                "torchvision",
                str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_c = importlib.util.find_spec("torchvision._C").origin  # type: ignore
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_c)
                data.append(("torchvision arch flags", msg))
            except (ImportError, AttributeError):
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    return data


def collect_env_info() -> str:
    """

    :return:
    """
    data = []
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import deepdoctection  # noqa

        data.append(
            ("deepdoctection", str(deepdoctection.__version__) + " @" + os.path.dirname(deepdoctection.__file__))
        )
    except ImportError:
        data.append(("deepdoctection", "failed to import"))
    except AttributeError:
        data.append(("deepdoctection", "imported a wrong installation"))

    has_prctl = True
    try:
        import prctl  # type: ignore

        _ = prctl.set_pdeathsig  # pylint: disable=E1101
    except ModuleNotFoundError:
        has_prctl = False
    data.append(("python-prctl", str(has_prctl)))

    # print system compilers when extension fails to build
    if sys.platform != "win32":  # don't know what to do for windows
        data.append(("Plattform", sys.platform))
        try:
            # this is how torch/utils/cpp_extensions.py choose compiler
            cxx = os.environ.get("CXX", "c++")
            cxx = subprocess.check_output(f"'{cxx}' --version", shell=True)  # type: ignore
            cxx = cxx.decode("utf-8").strip().split("\n", maxsplit=1)[0]  # type: ignore
        except subprocess.SubprocessError:
            cxx = "Not found"
        data.append(("Compiler ($CXX)", cxx))
    else:
        data.append(("Plattform", sys.platform + " Plattform not supported."))

    data = pt_info(data)
    data = tf_info(data)

    data = collect_installed_dependencies(data)

    env_str = tabulate(data) + "\n"

    if pytorch_available():
        env_str += collect_torch_env()

    return env_str


def auto_select_lib_and_device() -> None:
    """
    Select the DL library and subsequently the device.
    This will set environment variable `USE_TENSORFLOW`, `USE_PYTORCH` and `USE_CUDA`

    If TF is available, use TF unless a GPU is not available, in which case choose PT. If CUDA is not available and PT
    is not installed raise ImportError.
    """

    if tf_available() and tensorpack_available():
        from tensorpack.utils.gpu import get_num_gpu  # pylint: disable=E0401

        if get_num_gpu() >= 1:
            os.environ["USE_TENSORFLOW"] = "True"
            os.environ["USE_PYTORCH"] = "False"
            os.environ["USE_CUDA"] = "True"
            os.environ["USE_MPS"] = "False"
            return
        if pytorch_available():
            os.environ["USE_TENSORFLOW"] = "False"
            os.environ["USE_PYTORCH"] = "True"
            os.environ["USE_CUDA"] = "False"
            return
        logger.warning(
            LoggingRecord("You have Tensorflow installed but no GPU is available. All Tensorflow models require a GPU.")
        )
        os.environ["USE_TENSORFLOW"] = "False"
        os.environ["USE_PYTORCH"] = "False"
        os.environ["USE_CUDA"] = "False"
    if pytorch_available():
        import torch

        if torch.cuda.is_available():
            os.environ["USE_TENSORFLOW"] = "False"
            os.environ["USE_PYTORCH"] = "True"
            os.environ["USE_CUDA"] = "True"
            return
        if torch.backends.mps.is_available():
            os.environ["USE_TENSORFLOW"] = "False"
            os.environ["USE_PYTORCH"] = "True"
            os.environ["USE_CUDA"] = "False"
            os.environ["USE_MPS"] = "True"
            return
        os.environ["USE_TENSORFLOW"] = "False"
        os.environ["USE_PYTORCH"] = "True"
        os.environ["USE_CUDA"] = "False"
        os.environ["USE_MPS"] = "False"
        return
    logger.warning(
        LoggingRecord(
            "Neither Tensorflow or Pytorch are available. You will not be able to use any Deep Learning "
            "model from the library."
        )
    )


def get_device(ignore_cpu: bool = True) -> Literal["cuda", "mps", "cpu"]:
    """
    Device checks for running PyTorch with CUDA, MPS or optionall CPU.
    If nothing can be found and if `disable_cpu` is deactivated it will raise a `ValueError`

    :param ignore_cpu: Will not consider `cpu` as valid return value
    :return: Either cuda or mps
    """

    if ast.literal_eval(os.environ.get("USE_CUDA", "True")):
        return "cuda"
    if ast.literal_eval(os.environ.get("USE_MPS", "True")):
        return "mps"
    if not ignore_cpu:
        return "cpu"
    raise RuntimeWarning("Could not find either GPU nor MPS")


def auto_select_viz_library() -> None:
    """Setting PIL as default image library if cv2 is not installed"""

    # if env variables are already set, don't change them
    if os.environ.get("USE_DD_PILLOW") or os.environ.get("USE_DD_OPENCV"):
        return
    if opencv_available():
        os.environ["USE_DD_PILLOW"] = "False"
        os.environ["USE_DD_OPENCV"] = "True"
    else:
        os.environ["USE_DD_PILLOW"] = "True"
        os.environ["USE_DD_OPENCV"] = "False"


# pylint: enable=import-outside-toplevel
