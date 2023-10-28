# -*- coding: utf-8 -*-
# File: __init__.py

"""
Init file for utils package
"""
from typing import Optional, Tuple, Union, no_type_check

from .context import *
from .file_utils import *
from .fs import *
from .identifier import *
from .logger import *
from .metacfg import *
from .pdf_utils import *
from .settings import *
from .systools import *
from .tqdm import *
from .transform import *
from .utils import *
from .viz import *

__all__ = []


@no_type_check
def _global_import(
    name, prefix: Optional[Union[str, Tuple[str, ...]]] = None, suffix: Optional[Union[str, Tuple[str, ...]]] = None
):
    prefix_default = prefix is None
    suffix_default = suffix is None
    p = __import__(name, globals(), None, level=1)  # pylint: disable=C0103
    lst = p.__all__ if "__all__" in dir(p) else dir(p)
    for k in lst:
        if not k.startswith("__"):
            if prefix_default and suffix_default:
                globals()[k] = p.__dict__[k]
                __all__.append(k)
            elif not prefix_default:
                if k.startswith(prefix):
                    globals()[k] = p.__dict__[k]
                    __all__.append(k)
            elif not suffix_default:
                if k.endswith(suffix):
                    globals()[k] = p.__dict__[k]
                    __all__.append(k)


_global_import("file_utils", suffix=("_available", "_requirement"))
_global_import("metacfg", prefix=("set_config_by_yaml", "save_config_to_yaml", "config_to_cli_str"))
_global_import("utils", prefix=("delete_keys_from_dict", "split_string", "string_to_dict"))
_global_import(
    "settings", suffix=("Type", "TokenClasses", "BioTag", "TokenClassWithTag", "Relationships", "Languages", "get_type")
)

