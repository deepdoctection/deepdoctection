# -*- coding: utf-8 -*-
# File: tp2d2.py

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
A script to convert TP checkpoints to D2 checkpoints
"""

import pickle
import numpy as np

from collections import OrderedDict
from copy import copy

from tensorpack.tfutils.varmanip import load_checkpoint_vars, save_checkpoint_vars
from deep_doctection.utils import set_config_by_yaml


def reduce_tp_model_size(weights):
    all_keys = copy(list(weights.keys()))
    for t in all_keys:
        if t.endswith("/AccumGrad"):
            weights.pop(t)
        if t.endswith("/Momentum"):
            weights.pop(t)
    weights.pop("global_step")
    weights.pop("learning_rate")
    weights.pop("apply_gradients/AccumGrad/counter")
    return weights


def convert_weights_tp_to_d2(weights, cfg):
    d2_weights = OrderedDict()
    all_keys = copy(list(weights.keys()))
    for t in all_keys:
        if t.endswith("/AccumGrad"):
            weights.pop(t)
        if t.endswith("/Momentum"):
            weights.pop(t)

    def _convert_conv(src, dst):
        if src + "/W" in weights:
            src_w = weights.pop(src + "/W")
            d2_weights[dst + ".weight"] = src_w.transpose(3,2,0,1)

        if src + "/b" in weights:
            d2_weights[dst + ".bias"] = weights.pop(src + "/b")

        if src + "/gn/gamma" in weights:
            d2_weights[dst + ".norm.weight"] = weights.pop(src + "/gn/gamma")
            d2_weights[dst + ".norm.bias"] = weights.pop(src + "/gn/beta")

        if src + "/gamma" in weights:
            d2_weights[dst + ".norm.weight"] = weights.pop(src + "/gamma")
            d2_weights[dst + ".norm.bias"] = weights.pop(src + "/beta")

    def _convert_fc(src,dst):
        d2_weights[dst + ".weight"] = weights.pop(src + "/W").transpose()
        d2_weights[dst + ".bias"] = weights.pop(src + "/b").transpose()

    # the convertion
    d2_backbone_prefix = "backbone.bottom_up."

    # first conv
    _convert_conv("conv0",d2_backbone_prefix +"stem.conv1")

    # four backbone groups
    for grpid in range(4):
        # numb blocks in third group
        num_resnet_blocks = 6 if cfg.BACKBONE.RESNET_NUM_BLOCKS[2]==6 else 23
        for blkid in range([3, 4, num_resnet_blocks ,3 ][grpid]):
            _convert_conv(f"group{grpid}/block{blkid}/conv1",d2_backbone_prefix + f"res{grpid + 2}.{blkid}.conv1")
            _convert_conv(f"group{grpid}/block{blkid}/conv2",d2_backbone_prefix + f"res{grpid + 2}.{blkid}.conv2")
            _convert_conv(f"group{grpid}/block{blkid}/conv3",d2_backbone_prefix + f"res{grpid + 2}.{blkid}.conv3")

            # skip connection
            if blkid == 0:
                _convert_conv(f"group{grpid}/block{blkid}/convshortcut",d2_backbone_prefix + f"res{grpid + 2}.{blkid}.shortcut")

    # FPN lateral and posthoc
    for lvl in range(2, 6):
        _convert_conv(f"fpn/lateral_1x1_c{lvl}", f"backbone.fpn_lateral{lvl}")
        _convert_conv(f"fpn/gn_c{lvl}", f"backbone.fpn_lateral{lvl}")
        _convert_conv(f"fpn/posthoc_3x3_p{lvl}", f"backbone.fpn_output{lvl}")
        _convert_conv(f"fpn/gn_p{lvl}", f"backbone.fpn_output{lvl}")

    # RPN
    _convert_conv("rpn/conv0", "proposal_generator.rpn_head.conv" )
    _convert_conv("rpn/class","proposal_generator.rpn_head.objectness_logits")
    _convert_conv("rpn/box", "proposal_generator.rpn_head.anchor_deltas")

    def _convert_box_predictor(src, dst):
        assert cfg.FPN.CASCADE
        _convert_fc(src + "/box", dst + ".bbox_pred")
        _convert_fc(src + "/class", dst + ".cls_score")

        num_class = d2_weights[dst + ".cls_score.weight"].shape[0]
        idx = np.roll(np.arange(num_class),shift=-1)
        d2_weights[dst + ".cls_score.weight"] = d2_weights[dst + ".cls_score.weight"][idx, :]
        d2_weights[dst + ".cls_score.bias"] = d2_weights[dst + ".cls_score.bias"][idx]

    if cfg.FPN.CASCADE:
        for k in range(3):
            for i in range(2 if cfg.FPN.FRCNN_HEAD_FUNC=="fastrcnn_2fc_head" else 1):
                _convert_fc(f"cascade_rcnn_stage{k+1}/head/fc{i+6}",f"roi_heads.box_head.{k}.fc{i+1}")
            _convert_box_predictor(f"cascade_rcnn_stage{k+1}/outputs",f"roi_heads.box_predictor.{k}")

    if cfg.MODE_MASK:
        raise NotImplementedError

    weights.pop("global_step")
    weights.pop("learning_rate")
    weights.pop("apply_gradients/AccumGrad/counter")
    return {"__author__": "deepdoctection", "model": d2_weights}


if __name__ == '__main__':

    #path_config = "path/to/yaml_config"
    #path_model = "path/to/tp_checkpoint"
    path_config = "/home/janis/Public/deepdoctection/configs/tp/rows/conf_frcnn_rows.yaml"
    path_model = "/home/janis/Documents/train/rows_2/model-1370000.data-00000-of-00001"
    path_output_model = "/home/janis/Documents/d2/layout/d2_layout.pkl"
    path_output_tp_model = "/home/janis/Documents/tp_inference/rows/model-1370000_inf_only"
    """
    cfg = set_config_by_yaml(path_config)
    tp_dict = load_checkpoint_vars(path_model)
    d2_dict =  convert_weights_tp_to_d2(tp_dict,cfg)
    with open(path_output_model, 'wb') as handle:
        pickle.dump(d2_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    tp_dict = load_checkpoint_vars(path_model)
    tp_dict = reduce_tp_model_size(tp_dict)
    save_checkpoint_vars(tp_dict, path_output_tp_model)




