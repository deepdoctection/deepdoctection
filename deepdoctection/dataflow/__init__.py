# -*- coding: utf-8 -*-
# File: __init__.py

"""
Dataflow is the package of choice for loading and processing data in both training and prediction environments. Dataflow
is essentially pure Python and, with a simple API, contains a variety of methods for parallelling complex
transformations. Further information can be found in the excellent documentation:

https://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html

To make it easier to use, we re-import dataflow into the package of the same name.
"""

from dataflow.dataflow import *  # type: ignore # pylint: disable=W0622

from .common import *
from .custom import *
from .custom_serialize import *
from .stats import *
