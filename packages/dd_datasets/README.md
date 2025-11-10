# dd-datasets

Dataset building and processing tools for deepdoctection.

## Overview

`dd-datasets` is a package that provides comprehensive dataset management capabilities for document AI tasks. It includes:

- **Dataflow**: Efficient data loading and processing pipelines inspired by Tensorpack
- **Mapper**: Transformation functions for various dataset formats (COCO, Pascal VOC, PubTables, XFUND, etc.)
- **Datasets**: Built-in dataset definitions and builders for popular document understanding datasets

## Installation

```bash
pip install dd-datasets
```

For PyTorch-based features (Detectron2, HuggingFace transformers integration):

```bash
pip install dd-datasets[pt]
```

## Dependencies

This package depends on `dd-datapoint` for core data structures and utilities.

## Features

### Dataflow

Efficient data loading and processing with support for:
- Parallel processing
- Custom serialization (JSON, PDF, etc.)
- Data caching and statistics

### Mapper

Transformation functions for converting between various dataset formats:
- COCO format
- Pascal VOC format
- Prodigy format
- PubTables format
- XFUND format
- Detectron2 format
- HuggingFace format
- LayoutLM format

### Datasets

Built-in support for popular datasets:
- PublayNet
- PubTabNet
- FinTabNet
- PubTables-1M
- FUNSD
- XFUND
- RVL-CDIP
- DocLayNet
- IIT-CDIP (IIITAR-13K)

## License

Apache License 2.0

## Author

Dr. Janis Meyer

