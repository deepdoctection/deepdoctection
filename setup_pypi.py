# -*- coding: utf-8 -*-
# File: setup_pypi.py

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

with open(os.path.join(ROOT, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')


def get_version():
    init_path = os.path.join(ROOT, "deepdoctection", "__init__.py")
    init_py = open(init_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


sys.path.insert(0, ROOT)

DIST_DEPS = [
    "importlib-metadata",
    "jsonlines",
    "types-termcolor",
    "opencv-python",
    "pycocotools",
    "pypdf2",
    "numpy>=1.21",
    "lxml",
    "huggingface_hub",
    "packaging>=20.0",
    "pyyaml",
    "scikit-learn",
    "scipy",
    "networkx",
    "mock",
]

TF_DEPS = ["tensorpack"]

PT_DEPS = ["transformers"]

# recommonmark add .md files to rst easily
DEV_DEPS = ["types-PyYAML", "types-tabulate", "sphinx", "sphinx_rtd_theme", "recommonmark"]

# when building requirements.txt for rtd comment the following two lines
if sys.platform == "linux":
    DEV_DEPS.append("python-prctl")

TEST_DEPS = ["black==22.3.0", "isort", "pylint", "mypy", "pytest", "pytest-cov"]

EXTRA_DEPS = {"tf": TF_DEPS, "dev": DEV_DEPS, "test": TEST_DEPS, "pt": PT_DEPS}

setup(
    name="deepdoctection",
    version=get_version(),
    author="Dr. Janis Meyer",
    url="https://github.com/deepdoctection/deepdoctection",
    license="Apache License 2.0",
    description="Repository for Document AI",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=DIST_DEPS,
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
