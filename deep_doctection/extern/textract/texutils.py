# -*- coding: utf-8 -*-
# File: texutils.py

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
Textract related utils
"""
from shutil import which
import importlib.util

from ...utils.detection_types import Requirement

_BOTO3_AVAILABLE = importlib.util.find_spec("boto3") is not None
_BOTO3_ERR_MSG = "Boto3 must be installed: >>make install-aws-dependencies"

_AWS_CLI_AVAILABLE = which("aws") is not None
_AWS_ERR_MSG = "AWS CLI must be installed https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"


def boto3_available() -> bool:
    """
    Returns True if Boto3 is installed
    """

    return bool(_BOTO3_AVAILABLE)


def get_boto3_requirement() -> Requirement:
    """
    Return Boto3 requirement
    """
    return "boto3", boto3_available(), _BOTO3_ERR_MSG


def aws_available() -> bool:
    """
    Returns True if AWS CLI is installed
    """
    return bool(_AWS_CLI_AVAILABLE)


def get_aws_requirement() -> Requirement:
    """
    Return AWS CLI requirement
    """
    return "aws", aws_available(), _AWS_ERR_MSG
