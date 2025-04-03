#!/usr/bin/env python
# adapted from https://github.com/facebookresearch/detectron2/blob/main/tools/deploy/export_model.py
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import torch

import cv2

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.export import (
    TracingAdapter,
    dump_torchscript_IR,
)
from detectron2.modeling import GeneralizedRCNN, build_model
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager


def setup_cfg(path_config_yaml, device):
    cfg = get_cfg()
    cfg.defrost()
    cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.01
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.DEVICE = device
    cfg.merge_from_file(path_config_yaml)
    cfg.freeze()
    return cfg

def export_tracing(torch_model, image, path_output):
    assert TORCH_VERSION >= (1, 8)
    inputs = [{"image": image}]

    assert isinstance(torch_model, GeneralizedRCNN)

    def inference(model, inputs):
        # use do_postprocess=False so it returns ROI mask
        inst = model.inference(inputs, do_postprocess=False)[0]
        return [{"instances": inst}]

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    ts_model = torch.jit.trace(traceable_model, (image,))
    with PathManager.open(os.path.join(path_output, "model.ts"), "wb") as f:
        torch.jit.save(ts_model, f)
    dump_torchscript_IR(ts_model, path_output)


if __name__ == "__main__":

    path_model_weights="/path/to/model.pt"
    path_config_yaml = "/path/to/config.yaml"
    path_output_dir = "/path/to/output_dir"
    path_sample_image = "/path/to/sample_image.png"
    device = "cpu"

    PathManager.mkdirs(path_output_dir)
    # Disable re-specialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(path_config_yaml, device)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(path_model_weights)
    torch_model.eval()

    np_image = cv2.imread(path_sample_image)
    image = torch.as_tensor(np_image.astype("float32").transpose(2, 0, 1))
    export_tracing(torch_model, image, path_output_dir)
