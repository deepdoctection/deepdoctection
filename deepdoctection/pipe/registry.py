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
Pipeline component registry
"""

import catalogue  # type: ignore

__all__ = ["pipeline_component_registry"]


pipeline_component_registry = catalogue.create("deepdoctection", "pipeline_components", entry_points=True)

# todo: add func get_pipe_component
