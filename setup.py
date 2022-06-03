# -*- coding: utf-8 -*-
# File: setup.py

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

import os
import re
import sys

from setuptools import find_packages, setup

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__)))


def get_version():
    init_path = os.path.join(ROOT, "deepdoctection", "__init__.py")
    init_py = open(init_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


sys.path.insert(0, ROOT)

# Taken from https://github.com/huggingface/transformers/blob/master/setup.py. Will list all dependencies, even those
# that nee to be installed separately
_DEPS = [
    # the minimum requirements to run pipelines without considering DL models specific dependencies
    "catalogue==2.0.7",
    "importlib-metadata>=4.11.2",
    "huggingface_hub>=0.4.0",
    "jsonlines==3.0.0",
    "mock==4.0.3",
    "networkx>=2.7.1",
    "numpy>=1.21",
    "opencv-python",
    "packaging>=20.0",
    "pypdf2>=1.27.5",
    "python-prctl",
    "pyyaml==6.0",
    "types-PyYAML",
    "types-termcolor==1.1.3",
    "types-tabulate",
    "dataflow @ git+https://github.com/tensorpack/dataflow.git",
    # additional requirements to run eval and datasets (again without considering DL models)
    "lxml",
    "pycocotools>=2.0.2",
    "scikit-learn",
    # Tensorflow related dependencies
    "protobuf==3.20.1",
    "tensorpack",
    # PyTorch related dependencies
    "transformers",
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git",
    # other third party related dependencies (services or DL libraries). Must be installed by users
    "boto3",
    "pdfplumber",
    "tensorflow-addons>=0.13.0",
    "python-doctr",
    "fasttext",
    # dev dependencies
    "black==22.3.0",
    "isort",
    "pylint",
    "mypy",
    # docs
    "sphinx",
    "sphinx_rtd_theme",
    "recommonmark",
    # test
    "pytest",
    "pytest-cov",
]

# lookup table with items like:
#
# pycocotools: "pycocotools>=2.0.2"
# tensorpack: "tensorpack"
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _DEPS)}


def deps_list(*pkgs: str):
    return [deps[pkg] for pkg in pkgs]


# pyp-pi dependencies without considering DL models specific dependencies
dist_deps = deps_list(
    "catalogue",
    "importlib-metadata",
    "huggingface_hub",
    "jsonlines",
    "mock",
    "networkx",
    "numpy",
    "opencv-python",
    "packaging",
    "pypdf2",
    "pyyaml",
    "types-termcolor",
)

if sys.platform == "linux":
    dist_deps.extend(deps_list("python-prctl"))

# source dependencies with dataflow
source_deps = dist_deps + deps_list("dataflow @ git+https://github.com/tensorpack/dataflow.git")

# full dependencies for using evaluations and all datasets
additional_deps = deps_list("lxml", "pycocotools", "scikit-learn")

# remaining depencies to use all models
remaining_deps = deps_list("boto3", "pdfplumber", "tensorflow-addons", "python-doctr", "fasttext")

full_deps = dist_deps + additional_deps
source_full_deps = source_deps + additional_deps
source_all_deps = source_deps + additional_deps + remaining_deps

# Tensorflow dependencies
additional_tf_deps = deps_list("tensorpack", "protobuf")

source_tf_deps = source_deps + additional_tf_deps
full_tf_deps = full_deps + additional_tf_deps
source_full_tf_deps = source_full_deps + additional_tf_deps
source_all_tf_deps = source_all_deps + additional_tf_deps

# PyTorch dependencies
additional_pt_deps = deps_list("lxml", "transformers")
source_additional_pt_deps = additional_pt_deps + deps_list(
    "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
)
# it does not make sense to define a non-full pt dependency, because everything is already available
full_pt_deps = full_deps + additional_pt_deps
source_full_pt_deps = source_full_deps + source_additional_pt_deps
source_all_pt_deps = source_all_deps + source_additional_pt_deps

# dependencies for rtd. Only needed to create requirements.txt
docs_deps = deps_list(
    "dataflow @ git+https://github.com/tensorpack/dataflow.git",
    "tensorpack",
    "boto3",
    "transformers",
    "pdfplumber",
    "lxml",
    "pycocotools",
    "scikit-learn",
)
if "python-prctl" in docs_deps:
    docs_deps.remove("python-prctl")

# test dependencies
test_deps = deps_list("pytest", "pytest-cov")

# dev dependencies
dev_deps = deps_list("black", "isort", "pylint", "mypy")

# TODO: add function that lists correct not pre-installed third party libs in package, such that requirement errors
#  can be printed with correct version dependencies

EXTRA_DEPS = {
    "tf": additional_tf_deps,
    "source-tf": source_tf_deps,
    "full-tf": full_tf_deps,
    "source-full-tf": source_full_tf_deps,
    "source-all-tf": source_all_tf_deps,
    "pt": full_pt_deps,
    "source-pt": source_full_pt_deps,
    "source-all-pt": source_all_pt_deps,
    "docs": docs_deps,
    "dev": dev_deps,
    "test": test_deps,
}

setup(
    name="deepdoctection",
    version=get_version(),
    author="Dr. Janis Meyer",
    url="https://github.com/deepdoctection/deepdoctection",
    license="Apache License 2.0",
    description="Repository for Document AI",
    install_requires=dist_deps,
    extras_require=EXTRA_DEPS,
    packages=find_packages(),
    package_data={
        "deepdoctection.configs": ["*.yaml"],
        "deepdoctection.datasets.instances.xsl": ["*.xsl"],
        "deepdoctection": ["py.typed"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
