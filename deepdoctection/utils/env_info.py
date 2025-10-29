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
Function for collecting environment information.

This is also the place where we give an overview of some environment variables.

For env variables with boolean character, use one of the following values:

```python
{"1", "True", "TRUE", "true", "yes"}
```

```python
USE_PYTORCH
USE_CUDA
USE_MPS
```

are responsible for selecting the predictors based on the installed DL framework and available devices.
It is not recommended to touch them.

```python
USE_DD_PILLOW
USE_DD_OPENCV
```

decide what image processing library the `viz_handler` should use. The default library is PIL and OpenCV need
to be installed separately. However, if both libraries have been detected `viz_handler` will opt for OpenCV.
Use the variables to let choose `viz_handler` according to your preferences.

```python
USE_DD_POPPLER
USE_DD_PDFIUM
```

For PDF rendering we use PyPDFium2 as default but for legacy reasons, we also support Poppler. If you want to enforce
Poppler set one to `USE_DD_POPPLER=True` and `USE_DD_PDFIUM=False` the other to `False`.

```python
HF_CREDENTIALS
```

will be used by the `ModelDownloadManager` to pass your credentials if you have a model registered that resides in a
private repo.

```python
MODEL_CATALOG
```

can store an (absolute) path to a `.jsonl` file.

"""
from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from typing import Optional
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


import numpy as np

from tabulate import tabulate

from .file_utils import (
    apted_available,
    aws_available,
    boto3_available,
    cocotools_available,
    distance_available,
    doctr_available,
    get_poppler_version,
    get_tesseract_version,
    jdeskew_available,
    lxml_available,
    opencv_available,
    pdf_to_cairo_available,
    pdf_to_ppm_available,
    pdfplumber_available,
    pypdfium2_available,
    pytorch_available,
    qpdf_available,
    scipy_available,
    sklearn_available,
    tesseract_available,
    transformers_available,
    wandb_available,
    mkdir_p,
    copy_file_to_target
)

from .types import KeyValEnvInfos, PathLikeOrStr

ENV_VARS_TRUE: set[str] = {"1", "True", "TRUE", "true", "yes"}

__all__ = ["collect_env_info",
           "EnvSettings",
           "SETTINGS",
           "ENV_VARS_TRUE"]

# pylint: disable=import-outside-toplevel



class EnvSettings(BaseSettings):
    """
    Central settings manager for deepdoctection.

    Responsibilities:
    - Load `.env` and process OS environment.
    - Apply rule-based overrides (viz backend, pdf rendering, DL framework).
    - Prepare cache dirs and copy default config files.
    - Export effective values back into `os.environ` for legacy modules.
    """

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_PROPAGATE: bool = False
    FILTER_THIRD_PARTY_LIB: bool = False
    STD_OUT_VERBOSE: bool = False

    # HF / Model catalog
    HF_CREDENTIALS: Optional[str] = None
    MODEL_CATALOG: Optional[str] = None

    # DL framework toggles
    DD_USE_TORCH: bool = False
    USE_TORCH: bool = False
    PYTORCH_AVAILABLE: bool = False
    USE_CUDA: bool = False
    USE_MPS: bool = False

    # Rendering and viz
    USE_DD_PILLOW: bool = False
    USE_DD_OPENCV: bool = False
    USE_DD_PDFIUM: bool = False
    USE_DD_POPPLER: bool = False

    # Optional dependency
    QPDF_AVAILABLE: bool = False

    # Paths (defaults inside user cache)
    DEEPDOCTECTION_CACHE: Path = Field(default_factory=lambda: Path.home() / ".cache" / "deepdoctection")
    MODEL_DIR: Path = Field(default_factory=lambda: Path.home() / ".cache" / "deepdoctection" / "weights")
    CONFIGS_DIR: Path = Field(default_factory=lambda: Path.home() / ".cache" / "deepdoctection" / "configs")
    DATASET_DIR: Path = Field(default_factory=lambda: Path.home() / ".cache" / "deepdoctection" / "datasets")

    # Package root (read-only)
    PACKAGE_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1], frozen=True)

    # Default bundled config sources (inside the package)
    PROFILES_SRC: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "configs" / "profiles.jsonl")
    CONF_DD_ONE_SRC: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "configs" / "conf_dd_one.yaml")
    CONF_TESSERACT_SRC: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "configs" / "conf_tesseract.yaml")

    # Target filenames in CONFIGS_DIR
    PROFILES_TARGET_NAME: str = "profiles.jsonl"
    CONF_DD_ONE_TARGET_NAME: str = "conf_dd_one.yaml"
    CONF_TESSERACT_TARGET_NAME: str = "conf_tesseract.yaml"


    # Pydantic Settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Post-load reconciliation and rule application ----
    @model_validator(mode="after")
    def _apply_runtime_rules(self) -> EnvSettings:
        """
        Apply availability and rule-based overrides, but only when user did not explicitly set the variable.
        """
        fields_set = getattr(self, "__fields_set__", set())

        # 0) Derive dirs from DEEPDOCTECTION_CACHE unless explicitly set by user/.env
        if "MODEL_DIR" not in fields_set:
            self.MODEL_DIR = self.DEEPDOCTECTION_CACHE / "weights"
        if "CONFIGS_DIR" not in fields_set:
            self.CONFIGS_DIR = self.DEEPDOCTECTION_CACHE / "configs"
        if "DATASET_DIR" not in fields_set:
            self.DATASET_DIR = self.DEEPDOCTECTION_CACHE / "datasets"


        # 1) DL framework (PyTorch, CUDA, MPS)
        if "PYTORCH_AVAILABLE" not in fields_set:
            if pytorch_available():
                self.PYTORCH_AVAILABLE = True
            else:
                self.PYTORCH_AVAILABLE = False

        if self.PYTORCH_AVAILABLE:
            # Prefer user choices if set; otherwise enable torch by default
            if "DD_USE_TORCH" not in fields_set:
                self.DD_USE_TORCH = True
            if "USE_TORCH" not in fields_set:
                self.USE_TORCH = True

            import torch  # noqa
            if "USE_CUDA" not in fields_set:
                self.USE_CUDA = bool(torch.cuda.is_available())
            if "USE_MPS" not in fields_set:
                self.USE_MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

        # 2) Viz backend auto-selection (PIL vs OpenCV)
        if ("USE_DD_PILLOW" not in fields_set) and ("USE_DD_OPENCV" not in fields_set):
            if opencv_available():
                self.USE_DD_OPENCV = True
                self.USE_DD_PILLOW = False
            else:
                self.USE_DD_OPENCV = False
                self.USE_DD_PILLOW = True

        # 3) PDF rendering framework (pdfium vs poppler)
        if ("USE_DD_PDFIUM" not in fields_set) and ("USE_DD_POPPLER" not in fields_set):
            if pypdfium2_available():
                self.USE_DD_PDFIUM = True
                self.USE_DD_POPPLER = False
            elif pdf_to_cairo_available() or pdf_to_ppm_available():
                self.USE_DD_PDFIUM = False
                self.USE_DD_POPPLER = True
            else:
                # Defer raising to runtime usage sites; here just leave both False
                self.USE_DD_PDFIUM = False
                self.USE_DD_POPPLER = False

        # 4) Optional: Track qpdf presence as a convenience flag
        if "QPDF_AVAILABLE" not in fields_set:
            self.QPDF_AVAILABLE = bool(qpdf_available())


        return self

    def export_to_environ(self) -> None:
        """
        Export effective settings into `os.environ`, so legacy modules keep working without refactors.
        """
        def _set(k: str, v) -> None:
            if isinstance(v, bool):
                os.environ[k] = "True" if v else "False"
            elif v is None:
                # do not export None
                return
            else:
                os.environ[k] = str(v)

        export_vars = {
            # logging
            "LOG_LEVEL": self.LOG_LEVEL,
            "LOG_PROPAGATE": self.LOG_PROPAGATE,
            "FILTER_THIRD_PARTY_LIB": self.FILTER_THIRD_PARTY_LIB,
            "STD_OUT_VERBOSE": self.STD_OUT_VERBOSE,
            # hf / catalog
            "HF_CREDENTIALS": self.HF_CREDENTIALS,
            "MODEL_CATALOG": self.MODEL_CATALOG,
            # dl flags
            "DD_USE_TORCH": self.DD_USE_TORCH,
            "USE_TORCH": self.USE_TORCH,
            "PYTORCH_AVAILABLE": self.PYTORCH_AVAILABLE,
            "USE_CUDA": self.USE_CUDA,
            "USE_MPS": self.USE_MPS,
            # viz/pdf
            "USE_DD_PILLOW": self.USE_DD_PILLOW,
            "USE_DD_OPENCV": self.USE_DD_OPENCV,
            "USE_DD_PDFIUM": self.USE_DD_PDFIUM,
            "USE_DD_POPPLER": self.USE_DD_POPPLER,
            "MODEL_CATALOG_BASE": self.CONFIGS_DIR / self.PROFILES_TARGET_NAME,
            "DD_ONE_CONFIG": self.CONFIGS_DIR / "dd" / self.CONF_DD_ONE_TARGET_NAME,
        }
        for k, v in export_vars.items():
            _set(k, v)

        # Export paths
        os.environ["DEEPDOCTECTION_CACHE"] = str(self.DEEPDOCTECTION_CACHE)
        os.environ["MODEL_DIR"] = str(self.MODEL_DIR)
        os.environ["CONFIGS_DIR"] = str(self.CONFIGS_DIR)
        os.environ["DATASET_DIR"] = str(self.DATASET_DIR)
        os.environ["PATH_DD_PACKAGE"] = str(self.PACKAGE_PATH)

    def ensure_cache_layout_and_copy_defaults(self, force_copy: bool = False) -> None:
        """
        Ensure cache dirs exist and copy default config files to CONFIGS_DIR.
        """


        mkdir_p(self.DEEPDOCTECTION_CACHE)
        mkdir_p(self.MODEL_DIR)
        mkdir_p(self.CONFIGS_DIR)
        mkdir_p(self.DATASET_DIR)
        mkdir_p(self.CONFIGS_DIR / "dd")

        # Compute targets
        profiles_target = self.CONFIGS_DIR / self.PROFILES_TARGET_NAME
        conf_dd_one_target = self.CONFIGS_DIR / "dd" / self.CONF_DD_ONE_TARGET_NAME
        conf_tesseract_target = self.CONFIGS_DIR / "dd" / self.CONF_TESSERACT_TARGET_NAME

        # Copy (idempotent unless force_copy)
        copy_file_to_target(self.PROFILES_SRC, profiles_target, force_copy=force_copy)
        copy_file_to_target(self.CONF_DD_ONE_SRC, conf_dd_one_target, force_copy=force_copy)
        copy_file_to_target(self.CONF_TESSERACT_SRC, conf_tesseract_target, force_copy=force_copy)

# Load .env and OS env, apply rules, then export to environ for legacy modules
SETTINGS = EnvSettings()
SETTINGS.export_to_environ()
SETTINGS.ensure_cache_layout_and_copy_defaults()

def collect_torch_env() -> str:
    """
    Wrapper for `torch.utils.collect_env.get_pretty_env_info`.

    Returns:
        The environment information as a string.
    """
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def collect_installed_dependencies(data: KeyValEnvInfos) -> KeyValEnvInfos:
    """
    Collect installed dependencies for all third party libraries.

    Args:
        data: A list of tuples to dump all collected package information such as the name and the version.

    Returns:
        A list of tuples containing the name of the library and the version (if available).
    """


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
        import scipy

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

    if wandb_available():
        import wandb

        data.append(("Wandb", wandb.__version__))
    else:
        data.append(("Wandb", "None"))

    return data


def detect_compute_compatibility(cuda_home: Optional[PathLikeOrStr], so_file: Optional[PathLikeOrStr]) -> str:
    """
    Detect the compute compatibility of a CUDA library.

    Args:
        cuda_home: The path to the CUDA installation.
        so_file: The path to the shared object file.

    Returns:
        The compute compatibility of the CUDA library.
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


# Heavily inspired by https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/collect_env.py
def pt_info(data: KeyValEnvInfos) -> KeyValEnvInfos:
    """
    Returns a list of (key, value) pairs containing PyTorch information.

    Args:
        data: A list of tuples to dump all collected package information such as the name and the version.

    Returns:
        A list of tuples containing all the collected information.
    """

    if not SETTINGS.PYTORCH_AVAILABLE:
        data.append(("PyTorch", "None"))
        return []
    else:
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

    if has_gpu:
        has_gpu_text = "Yes"
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
    else:
        has_gpu_text = "No: torch.cuda.is_available() == False"

    data.append(("GPU available", has_gpu_text))

    mps_build = "No: torch.backends.mps.is_built() == False"
    if not has_mps:
        has_mps_text = "No: torch.backends.mps.is_available() == False"
    else:
        has_mps_text = "Yes"
        mps_build = str(torch.backends.mps.is_built())

    data.append(("MPS available", has_mps_text))
    data.append(("MPS built", mps_build))

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
    except (AttributeError, ModuleNotFoundError):
        data.append(("torchvision", "unknown"))

    return data


def collect_env_info() -> str:
    """
    Collects and returns environment information.

    Returns:
        A string containing the collected environment information.
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

    data = collect_installed_dependencies(data)

    env_str = tabulate(data) + "\n"

    if pytorch_available():
        env_str += collect_torch_env()

    return env_str


# pylint: enable=import-outside-toplevel
