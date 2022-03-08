# -*- coding: utf-8 -*-
# File: test_registry.py

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
Testing module pipe.registry
"""

from deepdoctection.datapoint.image import Image
from deepdoctection.pipe.base import PipelineComponent
from deepdoctection.pipe.registry import PipelineComponentRegistry


def test_pipe_registry_has_all_build_in_pipe_component_registered() -> None:
    """
    test pipe registry has all pipeline components registered
    """
    assert len(PipelineComponentRegistry.get_pipeline_component_names()) == 8


def test_pipe_registry_registered_new_pipeline_component() -> None:
    """
    test, that the new generated pipe component "TestPipeComponent" can be registered and retrieved from registry
    """

    class TestPipelineComponent(PipelineComponent):
        """
        TestPipelineComponent
        """

        def serve(self, dp: Image) -> None:
            """
            Processing an image through the whole pipeline component.
            """

    # Act
    PipelineComponentRegistry.register_pipeline_component("testpipe", TestPipelineComponent)
    test = PipelineComponentRegistry.get_pipeline_component("testpipe")

    # Assert
    assert test == TestPipelineComponent
