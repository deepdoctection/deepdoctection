# Installation Guide for dd_datapoint

## Quick Start

### Option 1: Editable Installation (Development)
For active development on the package:

```bash
cd packages/dd_datapoint
pip install -e .
```

### Option 2: Editable Installation with Development Tools
Include linting, type checking, and formatting tools:

```bash
cd packages/dd_datapoint
pip install -e ".[dev]"
```

### Option 3: Editable Installation with All Extras
Include both dev and test dependencies:

```bash
cd packages/dd_datapoint
pip install -e ".[dev,test]"
```

## Verify Installation

After installation, verify everything works:

```bash
python verify_setup.py
```

Or manually test:

```python
python -c "import dd_datapoint; print(f'✓ dd_datapoint v{dd_datapoint.__version__}')"
```


### Verification Failed
Check the error message from `verify_setup.py` and ensure all dependencies are installed.

## Notes

- Package name on PyPI will be `dd-datapoint` (with hyphen)
- Import name in Python is `dd_datapoint` (with underscore)
- Editable install means changes to source files are immediately reflected
- No need to reinstall after making changes to the code

