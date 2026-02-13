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
Utilities for collecting runtime and environment information and for centralizing
deepdoctection settings.

Environment variable usage
--------------------------
Settings are loaded from the environment (and from a `.env` file by default).
Boolean-like environment values are interpreted using the allowed true values:

    {"1", "True", "TRUE", "true", "yes"}

After `EnvSettings` is created, `SETTINGS.export_to_environ()` writes the
effective values back to `os.environ` using the mapping below. The mapping
controls what legacy modules see; booleans are exported as the strings
`"True"`/`"False"`, and `None` values are not exported.

Exported environment variables (names and meaning)
- `LOG_LEVEL` (str)
    Global logging level (default: `INFO`). Use `DEBUG` for a verbose logging.

- `LOG_PROPAGATE` (bool)
    Whether log records should propagate to parent loggers (default: `False`).

- `FILTER_THIRD_PARTY_LIB` (bool)
    Whether to filter third-party library output in logs (default: `False`).

- `STD_OUT_VERBOSE` (bool)
    Increase verbosity for stdout output (default: `False`). Especially useful to
    understand, if the `DatapointManager` has issues to add `Annotation`s to an `Image`.

- `HF_CREDENTIALS` (Secret / sensitive)
    Hugging Face credentials used by the `ModelDownloadManager`. Treated as a secret.

- `MODEL_CATALOG` (str / path)
    Optional path to a `.jsonl` model catalog (default: `None`). This will add your (custom)
    models to the `ModelCatalog`. Do not confuse with `MODEL_CATALOG_BASE`.

- `DD_USE_TORCH` (bool)
    Internal toggle to prefer deepdoctection's Torch-based predictors (default: auto).
    Mainly there, for historical reasons.

- `USE_TORCH` (bool)
    Whether Torch should be used (default: enabled when PyTorch is available). Also
    used in other Libs, so do not touch.

- `PYTORCH_AVAILABLE` (bool)
    Read-only detection of whether PyTorch is installed.

- `USE_CUDA` (bool)
    Whether to prefer CUDA devices (default: auto-detected from PyTorch). This will be
    set to `True`, if a GPU can be found.

- `USE_MPS` (bool)
    Whether to prefer Apple's MPS backend (default: auto-detected from PyTorch).
    This will be set to `True`, if available.

- `USE_DD_PILLOW` (bool)
    Prefer Pillow for viz handling (default: `True` unless OpenCV detected). Pillow will be installed by default


- `USE_DD_OPENCV` (bool)
    Prefer OpenCV for viz handling (default: `True` when OpenCV is available).
    Note, that OpenCV will have to be installed independently.

- `USE_DD_PDFIUM` (bool)
    Prefer PyPDFium2 for PDF rendering (default: auto-detected when available). This is also the default choice.


- `USE_DD_POPPLER` (bool)
    Prefer Poppler-based rendering (legacy option; used if pdfium not available). Note, that Poppler wheels cannot
    by installed by any Python package and have to be installed separately.

- `DPI` (int)
    Default DPI used for rendering (default: `300`). The default setting is very high and for historical
    reasons we will be keeping this value, even though we recommend 200

- `IMAGE_WIDTH` (int)
    Optional default image width (default: `0`). Only relevant when rendering with Poppler.

- `IMAGE_HEIGHT` (int)
    Optional default image height (default: `0`). Only relevant when rendering with Poppler.

- `MODEL_CATALOG_BASE` (path)
    Path computed as `CONFIGS_DIR / profiles target` (exported for compatibility). There is no need to adjust
    this path.

- `DD_ONE_CONFIG` (path)
    Path to the `dd_analyzer` configuration in the configs directory (exported for compatibility).

- `DEEPDOCTECTION_CACHE` (path)
    Root cache directory for deepdoctection (default: `~/.cache/deepdoctection`).

- `MODEL_DIR` (path)
    Directory for downloaded model weights (default: `DEEPDOCTECTION_CACHE/weights`).

- `CONFIGS_DIR` (path)
    Directory for config files (default: `DEEPDOCTECTION_CACHE/configs`).

- `DATASET_DIR` (path)
    Directory for datasets (default: `DEEPDOCTECTION_CACHE/datasets`).

- `PATH_DD_PACKAGE` (path)
    Filesystem path to the installed deepdoctection package root.

Notes and behavior
- The pydantic settings loader reads `.env` by default (see `EnvSettings.model_config`).
- If a user explicitly sets a setting via environment or `.env`, that value is respected;
  otherwise runtime detection (e.g., presence of PyTorch, OpenCV, pdf backends) will
  determine defaults.
- When exporting, boolean values are written as the strings `True`/`False`. `None` is not exported.
- Secret values (e.g., `HF_CREDENTIALS`) are treated as sensitive and are redacted by helper
  text-dumping utilities.
- To customize behavior you may set any of the above variables in your shell, for example:

    export LOG_LEVEL=DEBUG
    export USE_DD_OPENCV=true
    export DEEPDOCTECTION_CACHE=/path/to/cache

"""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tabulate import tabulate

from .file_utils import (
    apted_available,
    aws_available,
    boto3_available,
    cocotools_available,
    copy_file_to_target,
    distance_available,
    doctr_available,
    get_poppler_version,
    get_tesseract_version,
    jdeskew_available,
    lxml_available,
    mkdir_p,
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
)
from .types import KeyValEnvInfos, PathLikeOrStr

ENV_VARS_TRUE: set[str] = {"1", "True", "TRUE", "true", "yes"}

__all__ = ["collect_env_info", "EnvSettings", "SETTINGS", "ENV_VARS_TRUE", "collect_torch_env"]

# pylint: disable=import-outside-toplevel


def _import_custom_object_types_module(mod: str) -> None:
    """
    Import a custom ObjectTypes module from an absolute ``*.py`` file path.

    This loader does not accept dotted module names. It requires an absolute
    filesystem path to a Python file and imports it under a temporary module
    name ``dd_custom_<stem>`` so its top-level code executes (e.g., enum
    registration decorators).

    Args:
        mod: Absolute path to a ``*.py`` file to import. Empty or relative values
             are not supported.

    Raises:
        ImportError: If ``mod`` is empty, not an absolute path, does not point to
            an existing file, an import spec cannot be created, or the module
            cannot be executed.
    """
    if not mod:
        return

    # Filesystem path to a Python file
    path = Path(mod).expanduser()
    if not path.is_file():
        raise ImportError(f"Custom ObjectTypes module path not found: {path}")

    spec = importlib.util.spec_from_file_location(f"dd_custom_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def resolve_config_source(
    filename: str,
    env_keys: tuple[str, ...],
    pkg_subdirs: tuple[tuple[str, str], ...],
) -> Path:
    """
    Resolve a configuration file path.

    The lookup order is:
    1. For each environment variable in `env_keys`, if set and points to an existing file,
       that path is returned.
    2. Walk up the parents of the current module and for each `(pkg, subdir)` in `pkg_subdirs`
       check `parent / pkg / "src" / pkg / subdir / filename` and return the first existing file.
       Otherwise check `parent / pkg / subdir / filename` and return the first existing file.
    3. Check the legacy location `here.parents[1] / "configs" / filename` and return it if it exists.
    4. If nothing is found, return the legacy location as a fallback (it may not exist).

    Args:
        filename: The basename of the configuration file to locate.
        env_keys: Tuple of environment variable names to consult for explicit paths.
        pkg_subdirs: Tuple of `(package_name, subdirectory)` pairs to search under repository parents.

    Returns:
        A `Path` pointing to the resolved configuration file or a fallback legacy path.
    """
    for key in env_keys:
        val = os.environ.get(key)
        if val:
            p = Path(val).expanduser()
            if p.is_file():
                return p

    here = Path(__file__).resolve()
    for parent in here.parents:
        for pkg, subdir in pkg_subdirs:
            candidate = parent / pkg / "src" / pkg / subdir / filename
            if candidate.is_file():
                return candidate
            candidate = parent / pkg / subdir / filename
            if candidate.is_file():
                return candidate

    legacy = here.parents[1] / "configs" / filename
    if legacy.is_file():
        return legacy
    return legacy


def find_env_file() -> Path | None:
    """Injecting a custom env file to EnvSettings."""

    value = os.environ.get("DD_ENV_FILE")
    if value is None:
        return

    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"DD_ENV_FILE must be an absolute path, got: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"DD_ENV_FILE does not point to an existing file: {path}")

    return path


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
    HF_CREDENTIALS: Optional[SecretStr] = None
    MODEL_CATALOG: Optional[str] = None

    # Custom object types autoload (comma‑separated string or JSON list in env is accepted)
    CUSTOM_OBJECT_TYPES_MODULES: list[str] = Field(default_factory=list)

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
    DPI: int = 300
    IMAGE_WIDTH: int = 0
    IMAGE_HEIGHT: int = 0

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
    PROFILES_SRC: Path = Field(
        default_factory=lambda: resolve_config_source(
            filename="profiles.jsonl",
            env_keys=("PROFILES_SRC", "DD_PROFILES_SRC"),
            pkg_subdirs=(("dd_core", "configs"), ("deepdoctection", "configs")),
        )
    )
    CONF_DD_ONE_SRC: Path = Field(
        default_factory=lambda: resolve_config_source(
            filename="conf_dd_one.yaml",
            env_keys=("CONF_DD_ONE_SRC", "DD_CONF_DD_ONE_SRC"),
            pkg_subdirs=(("dd_core", "configs"), ("deepdoctection", "configs")),
        )
    )
    CONF_TESSERACT_SRC: Path = Field(
        default_factory=lambda: resolve_config_source(
            filename="conf_tesseract.yaml",
            env_keys=("CONF_TESSERACT_SRC", "DD_CONF_TESSERACT_SRC"),
            pkg_subdirs=(("deepdoctection", "configs"), ("dd_core", "configs")),
        )
    )

    # Target filenames in CONFIGS_DIR
    PROFILES_TARGET_NAME: str = "profiles.jsonl"
    CONF_DD_ONE_TARGET_NAME: str = "conf_dd_one.yaml"
    CONF_TESSERACT_TARGET_NAME: str = "conf_tesseract.yaml"

    # AWS settings
    AWS_ACCESS_KEY_ID: Optional[SecretStr] = None
    AWS_SECRET_ACCESS_KEY: Optional[SecretStr] = None
    AWS_REGION: Optional[str] = None

    # Pydantic Settings config
    model_config = SettingsConfigDict(
        env_file= find_env_file() or ".env",
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
        fields_set: set[str] = getattr(self, "__fields_set__", set())

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

        # 5) Autoload custom ObjectTypes modules so their @register decorators run
        if self.CUSTOM_OBJECT_TYPES_MODULES:
            # Ensure `settings` is imported so the wrapped register is active
            try:
                from . import object_types as _dd_object_types  # noqa: F401,WPS433
            except ImportError:
                pass

            for mod in self.CUSTOM_OBJECT_TYPES_MODULES:  # pylint: disable=E1133
                if not mod:
                    continue
                try:
                    _import_custom_object_types_module(mod)
                except ImportError:
                    pass

        return self

    def export_to_environ(self) -> None:
        """
        Export effective settings into `os.environ`, so legacy modules keep working without refactors.
        """

        def _set(k: str, v: Any) -> None:
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
            "DPI": self.DPI,
            "IMAGE_WIDTH": self.IMAGE_WIDTH,
            "IMAGE_HEIGHT": self.IMAGE_HEIGHT,
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

    def ensure_cache_layout_and_copy_defaults(self, force_copy: bool = True) -> None:
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

        # Copy (idempotent unless force_copy) - only if source exists (for dd_datapoint package)
        if self.PROFILES_SRC.exists():  # pylint: disable=E1101
            copy_file_to_target(self.PROFILES_SRC, profiles_target, force_copy=force_copy)
        if self.CONF_DD_ONE_SRC.exists():  # pylint: disable=E1101
            copy_file_to_target(self.CONF_DD_ONE_SRC, conf_dd_one_target, force_copy=force_copy)
        if self.CONF_TESSERACT_SRC.exists():  # pylint: disable=E1101
            copy_file_to_target(self.CONF_TESSERACT_SRC, conf_tesseract_target, force_copy=force_copy)


# Load .env and OS env, apply rules, then export to environ for legacy modules
SETTINGS = EnvSettings()
SETTINGS.export_to_environ()
SETTINGS.ensure_cache_layout_and_copy_defaults()


def append_settings_to_env_data(data: KeyValEnvInfos) -> KeyValEnvInfos:
    """
    Append all EnvSettings attributes to the env info list as (key, value) tuples.
    - SecretStr values are fully redacted.
    - Known secret keys are redacted even if not typed as SecretStr.
    """

    def _stringify(v: Any) -> str:
        if isinstance(v, SecretStr):
            return "***"
        if v is None:
            return "None"
        if isinstance(v, (str, int, float, bool)):
            return str(v)
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (list, tuple, set)):
            return ", ".join(_stringify(x) for x in v)
        if isinstance(v, dict):
            return ", ".join(f"{_stringify(k)}={_stringify(v2)}" for k, v2 in v.items())
        return repr(v)

    items: list[tuple[str, Any]] = []
    for name in SETTINGS.model_fields.keys():
        items.append((name, getattr(SETTINGS, name)))

    for key, val in sorted(items, key=lambda kv: kv[0]):
        data.append((f"{key}", _stringify(val)))
    return data


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
            output_b = subprocess.check_output(f"'{cuobjdump}' --list-elf '{so_file}'", shell=True)
            output = output_b.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line_o = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line_o))
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

        _ = prctl.set_pdeathsig
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
    data = append_settings_to_env_data(data)

    env_str = tabulate(data) + "\n"

    if pytorch_available():
        env_str += collect_torch_env()

    return env_str


# pylint: enable=import-outside-toplevel
