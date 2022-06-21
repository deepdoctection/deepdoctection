# -*- coding: utf-8 -*-
# File: reduce_d2.py

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
A script to reduce the detectron2 model size by removing variables that are only relevant for
training. You will need the full checkpoint if you want to resume training from given artefacts.
"""

import os

import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def get_state_dict(path_yaml, path_weights):
    d2_conf_list = ["MODEL.WEIGHTS", path_weights]
    cfg = get_cfg()
    cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.1
    cfg.merge_from_file(path_yaml)
    cfg.merge_from_list(d2_conf_list)
    cfg.freeze()
    model = build_model(cfg).eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    return model.state_dict()


if __name__ == '__main__':

    path_model_weights = "/path/to/model.pth"
    path_config_yaml = "/path/to/config.yaml"
    path, file_name = os.path.split(path_model_weights)
    file_name,_ = file_name.split(".")
    path_target = path + "/" + file_name + "_inf_only.pt"
    state_dict = get_state_dict(path_config_yaml,path_model_weights)

    torch.save(state_dict,path_target)