<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
</p>

# deepdoctection-datasets

Categories and Datasets as well as some dataset instances for training models supported by deepdoctection.

## Overview

`dd-datasets` is a package that provides comprehensive dataset management capabilities for Document AI tasks. 

It includes:

- **datasets**: Built-in dataset definitions and dataflow builders for popular document understanding datasets. 
- **instances**: Pre-defined dataset instances for common document understanding tasks such as object detection, text 
                 classifications and named entity recognition.

## Installation

```bash
uv pip install dd-datasets
```

For using all datasets including those that require the xml-parsing tool lxml:

```bash
uv pip install dd-datasets[full]
```

## License

Apache License 2.0

## Author

Dr. Janis Meyer

