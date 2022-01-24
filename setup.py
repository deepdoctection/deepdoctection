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
import sys

from setuptools import find_packages, setup

ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__)))

sys.path.insert(0, ROOT)

about = {}
with open(os.path.join(ROOT, "__about__.py")) as about_file:
    exec(about_file.read(), about)

DIST_DEPS = [
    "importlib-metadata",
    "jsonlines",
    "types-termcolor",
    "opencv-python",
    "pycocotools",
    "pypdf2",
    "numpy",
    "huggingface_hub",
    "packaging>=20.0",
    "pdf2image",
    "pyyaml",
    "pytesseract",
    "scikit-learn",
    "scipy",
    "networkx",
    "mock",
    "dataflow @ git+https://github.com/tensorpack/dataflow.git",
]

# when building requirements.txt for rtd uncomment the following lines
# DIST_DEPS.append("tensorpack")
# DIST_DEPS.append("boto3")
# DIST_DEPS.append("transformers")
# TF_DEPS, AWS_DEPS,HF_DEPS = [], [], []

# when building requirements.txt for rtd comment the following lines
TF_DEPS = ["tensorpack"]
AWS_DEPS = ["boto3"]

# even though transformers works for certain models in Tensorflow, we currently support only models
# in Pytorch
PT_DEPS = ["transformers", "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"]

# recommonmark add .md files to rst easily
DEV_DEPS = ["types-PyYAML", "types-tabulate", "sphinx", "sphinx_rtd_theme", "recommonmark"]

# when building requirements.txt for rtd comment the following two lines
if sys.platform == "linux":
    DEV_DEPS.append("python-prctl")

TEST_DEPS = ["black", "isort", "pylint", "mypy", "pytest", "pytest-cov"]

EXTRA_DEPS = {"tf": TF_DEPS, "dev": DEV_DEPS, "test": TEST_DEPS, "aws": AWS_DEPS, "pt": PT_DEPS}

setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    url=about["__uri__"],
    license=about["__license__"],
    description=about["__summary__"],
    install_requires=DIST_DEPS,
    extras_require=EXTRA_DEPS,
    packages=find_packages(),
    package_data={"deep_doctection": ["py.typed"]},
    classifiers=[
        "Development Status :: Inception",
        "License :: Apache License 2.0",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
