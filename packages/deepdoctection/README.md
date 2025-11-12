# deepdoctection

**deepdoctection** is the core inference/server package for Document AI. It provides the pipeline framework, model integration (via extern), and analysis tools needed for document processing tasks.

## Overview

This package contains:

- **analyzer**: Configuration and factory functions for creating document analysis pipelines
- **configs**: YAML configuration files and model profiles
- **extern**: External model integrations (Detectron2, DocTr, HuggingFace Transformers, Tesseract, etc.)
- **pipe**: Pipeline components and services for document processing
- **eval**: Evaluation metrics (only available with `[full]` installation)
- **train**: Training utilities (only available with `[full]` installation)

## Installation

### Basic Installation (Inference Only)

For server/inference use cases, install the base package:

```bash
pip install deepdoctection
```

**Important**: Heavy ML dependencies must be installed separately:
- **PyTorch**: Follow instructions at https://pytorch.org/get-started/locally/
- **Transformers**: `pip install transformers>=4.48.0` (if using HF models)
- **DocTr**: `pip install python-doctr==0.10.0` (if using DocTr models)
- **Detectron2**: Follow instructions at https://detectron2.readthedocs.io/en/latest/tutorials/install.html

### Full Installation (Training & Evaluation)

For development, training, and evaluation:

```bash
pip install deepdoctection[full]
```

This includes additional dependencies for training and evaluation, but PyTorch must still be installed separately.

### Development Installation

From the repository root:

```bash
uv pip install -e packages/dd_deepdoctection[full,dev,test]
```

## Dependencies

This package depends on:
- **dd-datapoint**: Core data structures and utilities
- **dd-datasets**: Dataset building and processing tools

## Python Version

Requires Python >= 3.10

## License

Apache License 2.0

## Author

Dr. Janis Meyer
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "deepdoctection"
version = "1.0"
authors = [
    {name = "Dr. Janis Meyer"}
]
description = "Repository for Document AI - server/inference core package"
readme = "README.md"
license = {text = "Apache License 2.0"}
requires-python = ">=3.10"
classifiers = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

# Core dependencies for deepdoctection (server/inference)
# Note: Heavy ML dependencies (PyTorch, DocTr, Transformers, Detectron2) should be installed separately by users
dependencies = [
    "dd-datapoint>=1.0",
    "dd-datasets>=1.0",
    "huggingface_hub>=0.26.0",
    "opencv-python==4.8.0.76",
    "pycocotools>=2.0.2",
    "pyzmq>=16",
    # External services/tools
    "boto3==1.34.102",
    "pdfplumber>=0.11.0",
    "jdeskew>=0.2.2",
    "apted==1.0.3",
    "distance==0.1.3",
    "networkx>=2.7.1",
]

[project.optional-dependencies]
# Full installation including all training and evaluation dependencies
# Note: PyTorch should still be installed separately by users
full = [
    "timm>=0.9.16",
    "transformers>=4.48.0",
    "accelerate>=0.29.1",
    "python-doctr==0.10.0",
    "detectron2 @ git+https://github.com/deepdoctection/detectron2.git",
]

dev = [
    "black==23.7.0",
    "isort==5.13.2",
    "pylint==2.17.4",
    "mypy==1.4.1",
    "types-PyYAML>=6.0.12.12",
    "types-termcolor>=1.1.3",
    "types-tabulate>=0.9.0.3",
    "types-tqdm>=4.66.0.5",
    "types-Pillow>=10.2.0.20240406",
    "types-urllib3>=1.26.25.14",
    "lxml-stubs>=0.5.1",
]

test = [
    "pytest==8.0.2",
    "pytest-cov",
]

docs = [
    "jinja2",
    "mkdocs-material",
    "mkdocstrings-python",
    "griffe==0.25.0",
    "transformers>=4.48.0",
    "accelerate>=0.29.1",
]

[project.urls]
Homepage = "https://github.com/deepdoctection/deepdoctection"
Documentation = "https://deepdoctection.readthedocs.io"
Repository = "https://github.com/deepdoctection/deepdoctection"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
deepdoctection = ["py.typed", "configs/*.yaml", "configs/*.jsonl"]

[tool.black]
line-length = 120
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 120

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

