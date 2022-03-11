# -*- coding: utf-8 -*-
# File: tqdm.py

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
tqdm related functions. (Settings, options, etc.)
"""

from typing import Optional, Union

from tqdm import tqdm  # type: ignore

__all__ = ["get_tqdm"]


def get_tqdm(total: Optional[Union[int, float]] = None, **kwargs: Union[str, int]) -> tqdm:
    """
    Get tqdm progress bar with some default options to have consistent style.

    :param total:  The number of expected iterations.
    :return: A tqdm instance
    """

    default_tqdm_setting = dict(
        total=total,
        leave=True,
        smoothing=0.5,
        dynamic_ncols=True,
        ascii=True,
        bar_format="{l_bar}{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_noinv_fmt}]",
        mininterval=5,
    )
    default_tqdm_setting.update(kwargs)

    return tqdm(**default_tqdm_setting)
