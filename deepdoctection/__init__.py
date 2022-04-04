# -*- coding: utf-8 -*-
# File: __init__.py

"""
Init file for deepdoctection package
"""

from .analyzer import *
from .dataflow import *  # pylint: disable = W0622
from .datapoint import *
from .datasets import *
from .eval import *
from .extern import *
from .mapper import *  # pylint: disable = W0622
from .pipe import *
from .train import *
from .utils import *
from .utils.file_utils import pytorch_available, tf_available
from .utils.logger import logger

__version__ = 0.12

if not tf_available() and not pytorch_available():
    logger.info(
        "Neither Tensorflow or Pytorch are available. You will not be able to use any Deep Learning model from"
        "the library."
    )
