# Installation Guide for dd_core

## Quick Start

### Option 1: Editable Installation (Development)
For active development on the package:

```bash
cd packages/dd_core
pip install -e .
```

For installation with the more dependencies, i.e. to run with deepdoctection:

```bash
cd packages/dd_core
pip install -e ".[full]"
```

### Option 2: Editable Installation with Development Tools
Include linting, type checking, and formatting tools:

```bash
cd packages/dd_core
pip install -e ".[dev]"
```

### Option 3: Editable Installation with All Extras
Include both dev and test dependencies:

```bash
cd packages/dd_core
pip install -e ".[dev,test]"
```

## Notes

- Package name on PyPI will be `dd-docre` (with hyphen)
- Import name in Python is `dd_core` (with underscore)
- Editable install means changes to source files are immediately reflected
- No need to reinstall after making changes to the code

