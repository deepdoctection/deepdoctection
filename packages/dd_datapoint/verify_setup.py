#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification script for dd_datapoint package setup
"""

import sys
from pathlib import Path

# Add src to path for testing
package_root = Path(__file__).parent / "src"
sys.path.insert(0, str(package_root))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import dd_datapoint
        print(f"✓ dd_datapoint imported successfully (version: {dd_datapoint.__version__})")
    except ImportError as e:
        print(f"✗ Failed to import dd_datapoint: {e}")
        return False
    
    try:
        from dd_datapoint import utils
        print("✓ dd_datapoint.utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dd_datapoint.utils: {e}")
        return False
    
    try:
        from dd_datapoint import datapoint
        print("✓ dd_datapoint.datapoint imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dd_datapoint.datapoint: {e}")
        return False
    
    # Test specific exports
    try:
        from dd_datapoint.utils import logger, get_uuid, ObjectTypes
        print("✓ Sample utils exports accessible")
    except ImportError as e:
        print(f"✗ Failed to import utils exports: {e}")
        return False
    
    try:
        from dd_datapoint.datapoint import BoundingBox, Image, Annotation
        print("✓ Sample datapoint exports accessible")
    except ImportError as e:
        print(f"✗ Failed to import datapoint exports: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that key dependencies are available"""
    print("\nTesting dependencies...")
    
    deps = [
        "pydantic",
        "numpy",
        "catalogue",
        "yaml",
        "PIL",
        "pypdf",
        "termcolor",
        "tabulate",
        "tqdm",
    ]
    
    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - MISSING")
            missing.append(dep)
    
    if missing:
        print(f"\nWarning: Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -e .")
        return False
    
    return True


def verify_structure():
    """Verify package structure"""
    print("\nVerifying package structure...")
    
    base = Path(__file__).parent
    required_files = [
        "pyproject.toml",
        "README.md",
        "src/dd_datapoint/__init__.py",
        "src/dd_datapoint/py.typed",
        "src/dd_datapoint/utils/__init__.py",
        "src/dd_datapoint/datapoint/__init__.py",
    ]
    
    all_present = True
    for file_path in required_files:
        full_path = base / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            all_present = False
    
    return all_present


if __name__ == "__main__":
    print("=" * 60)
    print("DD_DATAPOINT PACKAGE VERIFICATION")
    print("=" * 60)
    
    structure_ok = verify_structure()
    deps_ok = test_dependencies()
    imports_ok = test_imports()
    
    print("\n" + "=" * 60)
    if structure_ok and imports_ok:
        print("✓ VERIFICATION PASSED")
        print("\nPackage is ready for installation!")
        print("Run: pip install -e .")
        sys.exit(0)
    else:
        print("✗ VERIFICATION FAILED")
        print("\nPlease fix the issues above.")
        sys.exit(1)

