# -*- coding: utf-8 -*-
# File: registry.py

# Copyright 2021 Dr. Janis Meyer. All rights reserved.
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
Module for PipeRegistry
"""

import inspect
import sys
from typing import Dict, List, Type

from .base import PipelineComponent
from .cell import *
from .common import *
from .layout import *
from .refine import *
from .segment import *
from .text import *

__all__ = ["PipelineComponentRegistry"]


_PIPELINE_COMPONENT: Dict[str, Type[PipelineComponent]] = dict(
    m
    for m in [m for m in inspect.getmembers(sys.modules[__name__], inspect.isclass) if m[0] not in ["ImageType"]]
    if issubclass(m[1], PipelineComponent) and m[0] != "PipelineComponent"
)


class PipelineComponentRegistry:
    """
    The PipelineComponentRegistry is the object for receiving pipeline component and registering new ones.
    """

    @staticmethod
    def print_pipeline_component_names() -> None:
        """
        Print a list of registered pipeline component names
        """
        print(list(_PIPELINE_COMPONENT.keys()))

    @staticmethod
    def get_pipeline_component(name: str) -> Type[PipelineComponent]:
        """
        Returns an instance of a pipeline component with a given name

        :param name: A pipeline component name
        :return: An instance of a pipeline component
        """
        return _PIPELINE_COMPONENT[name]

    @staticmethod
    def register_pipeline_component(name: str, pipeline_component: Type[PipelineComponent]) -> None:
        """
        Register a new pipeline component.

        :param name: A pipeline component name
        :param pipeline_component: A new pipeline component to add to the registry.
        """
        _PIPELINE_COMPONENT[name] = pipeline_component

    @staticmethod
    def get_pipeline_component_names() -> List[str]:
        """
        Get a list of available pipeline component names

        :return: A list of names
        """
        return list(_PIPELINE_COMPONENT.keys())
