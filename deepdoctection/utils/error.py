# -*- coding: utf-8 -*-
# File: error.py

# Copyright 2024 Dr. Janis Meyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom exceptions
"""


class BoundingBoxError(BaseException):
    """Special exception only for `datapoint.box.BoundingBox`"""


class AnnotationError(BaseException):
    """Special exception only for `datapoint.annotation.Annotation`"""


class ImageError(BaseException):
    """Special exception only for `datapoint.image.Image`"""


class UUIDError(BaseException):
    """Special exception only for `utils.identifier`"""


class DependencyError(BaseException):
    """Special exception only for missing dependencies. We do not use the internals `ImportError` or
    `ModuleNotFoundError`."""


class DataFlowTerminatedError(BaseException):
    """
    An exception indicating that the `DataFlow` is unable to produce any more data.

    This exception is raised when something wrong happens so that calling `__iter__` cannot give a valid iterator
    anymore. In most `DataFlow` this will never be raised.
    """


class DataFlowResetStateNotCalledError(BaseException):
    """
    An exception indicating that `reset_state()` has not been called before starting iteration.

    Example:
        ```python
        raise DataFlowResetStateNotCalledError()
        ```
    """

    def __init__(self) -> None:
        super().__init__("Iterating a dataflow requires .reset_state() to be called first")


class MalformedData(BaseException):
    """
    Exception class for malformed data.

    Use this class if something does not look right with the data.
    """


class FileExtensionError(BaseException):
    """
    Exception class for wrong file extensions.
    """


class TesseractError(RuntimeError):
    """
    Tesseract Error
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__()
        self.status = status
        self.message = message
        self.args = (status, message)
