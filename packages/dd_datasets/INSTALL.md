# dd-datasets Installation Guide

## Overview

The `dd-datasets` package provides dataset building and processing tools for deepdoctection. It depends on `dd-datapoint` for core data structures.

## Package Structure

```
dd_datasets/
├── dataflow/          # Data loading and processing pipelines
├── mapper/            # Dataset format transformations
└── datasets/          # Built-in dataset definitions
    └── instances/     # Specific dataset implementations
```

## Installation

### From PyPI (when published)

```bash
pip install dd-datasets
```

### For Development (Editable Install)

From the monorepo root:

```bash
# Install dd-datapoint first (dependency)
pip install -e packages/dd_datapoint

# Install dd-datasets
pip install -e packages/dd_datasets
```

### With PyTorch Features

For Detectron2 and HuggingFace transformer support:

```bash
pip install -e packages/dd_datasets[pt]
```

## Dependencies

### Core Dependencies
- `dd-datapoint>=1.0` - Core data structures and utilities
- `apted==1.0.3` - Tree edit distance
- `distance==0.1.3` - Distance metrics
- `jsonlines==3.1.0` - JSON Lines format support
- `lxml>=4.9.1` - XML processing
- `networkx>=2.7.1` - Graph algorithms
- `pycocotools>=2.0.2` - COCO format support
- `scipy>=1.13.1` - Scientific computing

### Optional Dependencies (pt extra)
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - PyTorch vision
- `transformers>=4.48.0` - HuggingFace transformers
- `accelerate>=0.29.1` - Training acceleration
- `detectron2` - Object detection framework

## Key Changes from Original deepdoctection

1. **Namespace Change**: All imports now use `dd_datapoint` instead of relative imports to `..utils` and `..datapoint`

2. **Independent Module**: `batched_nms` function is now included directly in `d2struct.py` (imported from torchvision)

3. **No External Dependencies**: Removed dependency on deepdoctection's `extern` module

## Verification

After installation, verify the package:

```python
import dd_datasets
print(dd_datasets.__version__)
```

## Python Version

Requires Python 3.10 or higher.

