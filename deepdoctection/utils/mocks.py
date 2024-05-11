# -*- coding: utf-8 -*-
# File: mocks.py

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
Some classes with the purpose to mock the original classes from the Tensorpack library, if Tensorpack is not installed
"""

from deepdoctection.utils.error import DependencyError


def layer_register(log_shape):  # pylint: disable=W0613
    """Mock layer_register function from tensorpack."""

    def inner(inputs):  # pylint: disable=W0613
        pass

    return inner


def under_name_scope():
    """Mock under_name_scope function from tensorpack."""

    def inner(inputs):  # pylint: disable=W0613
        pass

    return inner


def memoized(func):
    """Mock memoized function from tensorpack."""
    return func


def memoized_method(func):
    """Mock memoized_method function from tensorpack."""
    return func


def auto_reuse_variable_scope(inputs):  # pylint: disable=W0613
    """Mock auto_reuse_variable_scope function from tensorpack."""


class ModelDesc:  # pylint: disable=R0903
    """Mock ModelDesc class from tensorpack."""

    def __init__(self) -> None:
        raise DependencyError("Tensorpack not found.")


class ImageAugmentor:  # pylint: disable=R0903
    """Mock ImageAugmentor class from tensorpack."""

    def __init__(self) -> None:
        raise DependencyError("Tensorpack not found.")


class Callback:  # pylint: disable=R0903
    """Mock Callback class from tensor"""

    def __init__(self) -> None:
        raise DependencyError("Tensorpack not found.")


class Config:  # pylint: disable=R0903
    """Mock class for Config"""

    pass  # pylint: disable=W0107


class Tree:  # pylint: disable=R0903
    """Mock class for Tree"""

    pass  # pylint: disable=W0107


class IterableDataset:  # pylint: disable=R0903
    """Mock class for IterableDataset"""

    pass  # pylint: disable=W0107
