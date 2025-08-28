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

with open(os.path.join(ROOT, "README.md"), "rb") as f:
    long_description = f.read().decode("utf-8")


def get_version():
    init_path = os.path.join(ROOT, "deepdoctection", "__init__.py")
    init_py = open(init_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


sys.path.insert(0, ROOT)

# Taken from https://github.com/huggingface/transformers/blob/master/setup.py. Will list all dependencies, even those
# that need to be installed separately
_DEPS = [
    # the minimum requirements to run pipelines without considering DL models specific dependencies
    "apted==1.0.3",
    "catalogue==2.0.10",
    "distance==0.1.3",
    "huggingface_hub>=0.26.0",
    "importlib-metadata>=5.0.0",
    "jsonlines==3.1.0",
    "lazy-imports==0.3.1",
    "lxml>=4.9.1",
    "mock==4.0.3",
    "networkx>=2.7.1",
    "numpy>2.0", # When using fasttext-wheel, downgrading to numpy<1.x is required
    "opencv-python==4.8.0.76",  # this is not required anymore, but we keep its version as a reference
    "packaging>=20.0",
    "Pillow>=10.0.0",
    "pycocotools>=2.0.2",
    "pypdf>=6.0.0",
    "pypdfium2>=4.30.0",
    "python-prctl",
    "pyyaml>=6.0.1",
    "pyzmq>=16",
    "scipy>=1.13.1",
    "termcolor>=1.1",
    "tabulate>=0.7.7",
    "tqdm>=4.64.0",
    # type-stubs
    "types-PyYAML>=6.0.12.12",
    "types-termcolor>=1.1.3",
    "types-tabulate>=0.9.0.3",
    "types-tqdm>=4.66.0.5",
    "types-Pillow>=10.2.0.20240406",
    "types-urllib3>=1.26.25.14",
    "lxml-stubs>=0.5.1",
    # Tensorflow related dependencies
    "protobuf==3.20.1",
    "tensorpack==0.11",
    # PyTorch related dependencies
    "timm>=0.9.16",
    "transformers>=4.48.0",
    "accelerate>=0.29.1",
    # As maintenance of Detectron2 decreases, we will now use our own Fork the keep updating after rigorous testing.
    # This will hopefully prevent from issues like 233
    "detectron2 @ git+https://github.com/deepdoctection/detectron2.git",
    # other third party related dependencies (services or DL libraries). Must be installed by users
    "jdeskew>=0.2.2",
    "boto3==1.34.102",
    "pdfplumber>=0.11.0",
    "tensorflow-addons>=0.17.1",
    "tf2onnx>=1.9.2",
    "python-doctr==0.10.0",
    #"fasttext-wheel",
    # dev dependencies
    "python-dotenv==1.0.0",
    "click",  # version will not break black
    "black==23.7.0",
    "isort==5.13.2",
    "pylint==2.17.4",
    "mypy==1.4.1",
    # docs
    "jinja2",
    "mkdocs-material",
    "mkdocstrings-python",
    "griffe==0.25.0",
    # test
    "pytest==8.0.2",
    "pytest-cov",
    # other
    "wandb",
]

# lookup table with items like:
# pycocotools: "pycocotools>=2.0.2"
# tensorpack: "tensorpack"
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _DEPS)}


def deps_list(*pkgs: str):
    return [deps[pkg] for pkg in pkgs]


# pypi dependencies without considering DL models specific dependencies
dist_deps = deps_list(
    "catalogue",
    "huggingface_hub",
    "importlib-metadata",
    "jsonlines",
    "lazy-imports",
    "mock",
    "networkx",
    "numpy",
    "packaging",
    "Pillow",
    "pypdf",
    "pypdfium2",
    "pyyaml",
    "pyzmq",
    "scipy",
    "termcolor",
    "tabulate",
    "tqdm",
)


# remaining dependencies to use models that neither require TF nor PyTorch
additional_deps = deps_list(
    "boto3",
    "pdfplumber",
    #"fasttext-wheel",
    "jdeskew",
    "apted",
    "distance",
    "lxml",
)

tf_deps = deps_list("tensorpack", "protobuf", "tensorflow-addons", "tf2onnx", "python-doctr", "pycocotools")

# PyTorch dependencies
pt_deps = deps_list("timm", "transformers", "accelerate", "python-doctr", "pycocotools")

# Putting all together
tf_deps = dist_deps + tf_deps + additional_deps
pt_deps = dist_deps + pt_deps + additional_deps


# dependencies for rtd. Only needed to create requirements.txt
docs_deps = deps_list(
    "tensorpack",
    "boto3",
    "transformers",
    "accelerate",
    "pdfplumber",
    "lxml",
    "lxml-stubs",
    "jdeskew",
    "jinja2",
    "mkdocs-material",
    "mkdocstrings-python",
    "griffe",
)


# test dependencies
test_deps = deps_list("pytest", "pytest-cov")

# dev dependencies
dev_deps = deps_list(
    "python-dotenv",
    "click",
    "black",
    "isort",
    "pylint",
    "mypy",
    "wandb",
    "types-PyYAML",
    "types-termcolor",
    "types-tabulate",
    "types-tqdm",
    "lxml-stubs",
    "types-Pillow",
    "types-urllib3",
)

# TODO: add function that lists correct not pre-installed third party libs in package, such that requirement errors
#  can be printed with correct version dependencies.
# when uploading to pypi first comment all source extra dependencies so that there are no dependencies to dataflow

EXTRA_DEPS = {
    "tf": tf_deps,
    "pt": pt_deps,
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=dist_deps,
    extras_require=EXTRA_DEPS,
    packages=find_packages(exclude=["tests", "tests.*", "tests_d2"]),
    package_data={
        "deepdoctection.configs": ["*.yaml", "*.jsonl"],
        "deepdoctection.datasets.instances.xsl": ["*.xsl"],
        "deepdoctection": ["py.typed"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
)
