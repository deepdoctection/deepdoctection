# Installation Guide - deepdoctection

This guide covers the installation of the `deepdoctection` package.

## Prerequisites

- Python >= 3.10
- pip or uv package manager

## Installation Options

### 1. Basic Installation (Server/Inference)

For server/inference use cases without training capabilities:

```bash
pip install deepdoctection
```

This installs the core deepdoctection package with its dependencies (dd-datapoint and dd-datasets), but **does NOT include heavy ML frameworks**.

### 2. Install Heavy Dependencies Separately

After installing deepdoctection, you must install heavy ML dependencies based on your needs:

#### PyTorch (Required for most models)

Follow the official PyTorch installation instructions for your platform:
https://pytorch.org/get-started/locally/

Example for CPU:
```bash
pip install torch torchvision
```

Example for CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Transformers (For HuggingFace models)

```bash
pip install transformers>=4.48.0 accelerate>=0.29.1
```

#### DocTr (For DocTr text recognition)

```bash
pip install python-doctr==0.10.0
```

#### Detectron2 (For Detectron2 models)

Follow the official Detectron2 installation instructions:
https://detectron2.readthedocs.io/en/latest/tutorials/install.html

Or use the deepdoctection fork:
```bash
pip install git+https://github.com/deepdoctection/detectron2.git
```

### 3. Full Installation (Training & Evaluation)

For training and evaluation capabilities:

```bash
pip install deepdoctection[full]
```

This includes additional dependencies (timm, transformers, accelerate, python-doctr, detectron2), but **PyTorch must still be installed separately** as described above.

### 4. Development Installation

For development work on the package:

```bash
# From the repository root
uv pip install -e packages/dd_deepdoctection[full,dev,test]
```

This installs the package in editable mode with development and testing dependencies.

## Verifying Installation

```python
import deepdoctection as dd
print(dd.__version__)

# Check what's available
from deepdoctection.extern import ModelCatalog
print_model_infos()
```

## Dependencies

The deepdoctection package depends on:

- **dd-datapoint** (>=1.0): Core data structures and utilities
- **dd-datasets** (>=1.0): Dataset building and processing tools
- Additional external dependencies for model integrations

## Troubleshooting

### Import Errors

If you get import errors related to PyTorch, Transformers, or Detectron2, make sure you've installed these separately as described above.

### Version Conflicts

If you encounter version conflicts, try creating a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install deepdoctection
# Then install PyTorch and other heavy dependencies
```

## Platform Support

- **Linux**: Full support
- **macOS**: Supported (some models may have limited support)
- **Windows**: Limited support (check individual model requirements)

## License

Apache License 2.0

