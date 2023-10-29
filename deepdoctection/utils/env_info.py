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

import os
import sys
import subprocess
import re
from collections import defaultdict
from tabulate import tabulate

import importlib
import numpy as np

from .file_utils import pytorch_available, tensorpack_available, tf_available

if pytorch_available():
    import torch

if tf_available() and tensorpack_available():
    from tensorpack.utils.gpu import get_num_gpu  # pylint: disable=E0401

__all__=["collect_torch_env", "collect_env_info", "get_device", "auto_select_lib_and_device"]


def collect_torch_env():
    """Wrapper for torch.utils.collect_env.get_pretty_env_info"""
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()

def collect_installed_dependencies():
    pass

def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line))
            arch = sorted(set(arch))
            return ", ".join(arch)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info():
    """

    :return:
    """
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    has_mps = torch.backends.mps.is_available()

    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

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

    # print system compilers when extension fails to build
    if sys.platform != "win32":  # don't know what to do for windows
        data.append(("Plattform", sys.platform))
        try:
            # this is how torch/utils/cpp_extensions.py choose compiler
            cxx = os.environ.get("CXX", "c++")
            cxx = subprocess.check_output("'{}' --version".format(cxx), shell=True)
            cxx = cxx.decode("utf-8").strip().split("\n")[0]
        except subprocess.SubprocessError:
            cxx = "Not found"
        data.append(("Compiler ($CXX)", cxx))

        if has_cuda and CUDA_HOME is not None:
            try:
                nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                nvcc = subprocess.check_output("'{}' -V".format(nvcc), shell=True)
                nvcc = nvcc.decode("utf-8").strip().split("\n")[-1]
            except subprocess.SubprocessError:
                nvcc = "Not found"
            data.append(("CUDA compiler", nvcc))

    else:
        data.append(("Plattform", sys.platform + " Plattform not supported."))

    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

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
                from torch.utils.collect_env import get_nvidia_driver_version, run as _run

                data.append(("Driver version", get_nvidia_driver_version(_run)))
            except Exception:
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
        mps_build = torch.backends.mps.is_built()

    data.append(("MPS available", has_mps_text))
    data.append(("MPS available", mps_build))

    try:
        import torchvision

        data.append(
            (
                "torchvision",
                str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except (ImportError, AttributeError):
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


def get_device(ignore_cpu: bool = True) -> str:
    """
    Device checks for running PyTorch with CUDA, MPS or optionall CPU.
    If nothing can be found and if `disable_cpu` is deactivated it will raise a `ValueError`

    :param ignore_cpu: Will not consider `cpu` as valid return value
    :return: Either cuda or mps
    """

    if os.environ.get("USE_CUDA"):
        return "cuda"
    if os.environ.get("USE_MPS"):
        return "mps"
    if not ignore_cpu:
        return "cpu"
    raise ValueError(f'Could not find either GPU nor MPS')


def auto_select_lib_and_device() -> None:
    """
    Select the DL library and subsequently the device.
    This will set environment variable `USE_TENSORFLOW`, `USE_PYTORCH` and `USE_CUDA`

    If TF is available, use TF unless a GPU is not available, in which case choose PT. If CUDA is not available and PT
    is not installed raise ImportError.
    """

    if tf_available() and tensorpack_available():
        if get_num_gpu() >= 1:
            os.environ["USE_TENSORFLOW"]="True"
            os.environ["USE_PYTORCH"] = "False"
            os.environ["USE_CUDA"] = "True"
            os.environ["USE_MPS"] = "False"
        if pytorch_available():
            os.environ["USE_TENSORFLOW"]="False"
            os.environ["USE_PYTORCH"] = "True"
            os.environ["USE_CUDA"] = "False"
        raise ModuleNotFoundError("Install Pytorch and Torchvision to run with a CPU")
    if pytorch_available():
        if torch.cuda.is_available():
            os.environ["USE_TENSORFLOW"]="False"
            os.environ["USE_PYTORCH"] = "True"
            os.environ["USE_CUDA"] = "True"
            return
        if torch.backends.mps.is_available():
            os.environ["USE_TENSORFLOW"]="False"
            os.environ["USE_PYTORCH"] = "True"
            os.environ["USE_CUDA"] = "True"
            os.environ["USE_MPS"] = "True"
            return
        os.environ["USE_TENSORFLOW"] = "False"
        os.environ["USE_PYTORCH"] = "True"
        os.environ["USE_CUDA"] = "False"
        os.environ["USE_MPS"] = "False"
    raise ModuleNotFoundError("Install Tensorflow or Pytorch before building analyzer")