# -*- coding: utf-8 -*-
# File: __init__.py

"""
Dataflow is the package of choice for loading and processing data in both training and prediction environments. Dataflow
is essentially pure Python and, with a simple API, contains a variety of methods for parallelling complex
transformations. Further information can be found in the excellent documentation:

https://tensorpack.readthedocs.io/en/latest/tutorial/dataflow.html

To make it easier to use, we re-import dataflow into the package of the same name.
"""

from dataflow import *  # type: ignore # pylint: disable=W0622
from .custom import *
from .stats import *
from .common import *
from .custom_serialize import *

__all__ = [
    "DataFlow",
    "ProxyDataFlow",
    "RNGDataFlow",
    "DataFlowTerminated",
    "TestDataSpeed",
    "PrintData",
    "BatchData",
    "BatchDataByShape",
    "FixedSizeData",
    "MapData",
    "MapDataComponent",
    "RepeatedData",
    "RepeatedDataPoint",
    "RandomChooseData",
    "RandomMixData",
    "JoinData",
    "ConcatData",
    "SelectComponent",
    "LocallyShuffleData",
    "HDF5Data",
    "LMDBData",
    "LMDBDataDecoder",
    "CaffeLMDB",
    "SVMLightData",
    "ImageFromFile",
    "AugmentImageComponent",
    "AugmentImageCoordinates",
    "AugmentImageComponents",
    "MultiProcessRunner",
    "MultiProcessRunnerZMQ",
    "MultiThreadRunner",
    "MultiThreadMapData",
    "MultiProcessMapData",
    "MultiProcessMapDataZMQ",
    "MultiProcessMapAndBatchDataZMQ",
    "MultiProcessMapAndBatchData",
    "FakeData",
    "DataFromQueue",
    "DataFromList",
    "DataFromGenerator",
    "DataFromIterable",
    "send_dataflow_zmq",
    "RemoteDataZMQ",
    "LMDBSerializer",
    "NumpySerializer",
    "TFRecordSerializer",
    "HDF5Serializer",
    "CacheData",
    "CustomDataFromList",
    "CustomDataFromIterable",
    "MeanFromDataFlow",
    "StdFromDataFlow",
    "FlattenData",
    "SerializerJsonlines",
    "SerializerFiles",
    "SerializerCoco",
    "SerializerPdfDoc",
]
