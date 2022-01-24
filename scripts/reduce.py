# -*- coding: utf-8 -*-
# File: reduce.py

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
A script to reduce the model size by removing variables that are only relevant for
training. Note, that once reduced the model can ony be used for inference not for fine
tuning.
"""
import os.path

from tensorpack.tfutils.varmanip import load_checkpoint_vars, save_checkpoint_vars


def reduce_model(weights):
    all_keys = list(weights.keys())
    for t in all_keys:
        if t.endswith("/AccumGrad"):
            weights.pop(t)
        if t.endswith("/Momentum"):
            weights.pop(t)
    if "global_step" in weights:
        weights.pop("global_step")
    if "learning_rate" in weights:
        weights.pop("learning_rate")
    if "apply_gradients/AccumGrad/counter" in weights:
        weights.pop("apply_gradients/AccumGrad/counter")


if __name__ == '__main__':
    path_model = "/home/janis/Public/deepdoctection/weights/layout/model-2026500.data-00000-of-00001"
    path, file_name = os.path.split(path_model)
    file_name,_ = file_name.split(".")
    tp_dict = load_checkpoint_vars(path_model)
    reduce_model(tp_dict)
    path_target = path + "/" + file_name + "_reduced"
    save_checkpoint_vars(tp_dict, path_target)