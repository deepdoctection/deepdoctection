# -*- coding: utf-8 -*-
# File: __init__.py

"""
# Dataflows

Info:
    Dataflow is a package  for loading and processing data in both training and prediction environments. Dataflow
    is essentially pure Python and, with a simple API, it contains a variety of methods for parallelling complex
    transformations. We have integrated the most important DataFlow classes into deepdoctection in order to avoid
    installing the package separately from source.

    Further information (including several tutorials about performance) can be found in the excellent documentation:

    <https://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html>
"""


from .base import *
from .common import *
from .custom import *
from .custom_serialize import *
from .parallel_map import *
from .serialize import *
from .stats import *
