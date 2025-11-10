# dd_datapoint

Core data structures and utilities for the deepdoctection ecosystem.

## Overview

`dd_datapoint` is the foundational package of the deepdoctection suite, providing:

- **utils**: Core utility functions including logging, file operations, type definitions, and more
- **datapoint**: Data models for annotations, bounding boxes, images, and views

This package is designed to be lightweight and minimal, containing only the essential components needed for client-side processing of deepdoctection data structures without requiring any deep learning frameworks.

## Installation

```bash
pip install dd-datapoint
```

## Usage

```python
from dd_datapoint import BoundingBox, Image, Annotation
from dd_datapoint.utils import logger

# Use the core data structures
bbox = BoundingBox(ulx=10, uly=10, lrx=100, lry=100)
```

## Package Structure

- `dd_datapoint.utils`: Utility functions and helpers
- `dd_datapoint.datapoint`: Core data models for document analysis

## License

Apache License 2.0

## Author

Dr. Janis Meyer

