# -*- coding: utf-8 -*-
# File: config.py

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
Config functions for setting up Tensorpack Faster-RCNN models and training schemes.


Description of the config file.

Most of the descriptions are taken from

<https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/config.py>

Backbone settings


**BACKBONE**

```python
.BOTTLENECK: Resnet oder resnext_32xd4

.FREEZE_AFFINE: Do not train affine parameters inside norm layers

.FREEZE_AT: Options: 0, 1, 2. How many stages in backbone to freeze (not training)

.NORM: options: FreezeBN, SyncBN, GN, None

.RESNET_NUM_BLOCKS: For resnet50: [3,4,6,3], for resnet101: [3,4,23,3]

.TF_PAD_MODE: Use a base model with TF-preferred padding mode, which may pad more pixels on right or bottom
than top/left. See https://github.com/tensorflow/tensorflow/issues/18213. Using either one should probably give the same
performance.
```

**CASCADE**

```python
.BBOX_REG_WEIGHTS: Bounding box regression weights

.IOUS: Iou levels
```

**DATA**

```python
.TRAIN_NUM_WORKERS: Number of threads to use when parallelizing the pre-processing (e.g. augmenting, adding anchors,
RPN gt-labelling,...)
```

**FPN**

```
.ANCHOR_STRIDES: Strides for each FPN level. Must be the same length as ANCHOR_SIZES

.CASCADE: Use Cascade RCNN

.FRCNN_CONV_HEAD_DIM: Dimension of conv head in FRCNN head

.FRCNN_FC_HEAD_DIM: Dimension(s) of fc layer in FRCNN head

.FRCNN_HEAD_FUNC: Stack of FRCNN head. Choices: fastrcnn_2fc_head, fastrcnn_4conv1fc_{,gn_}head

.MRCNN_HEAD_FUNC: Stack of MRCNN head. Choices: maskrcnn_up4conv_{,gn_}head

.NORM: Choices: 'None', 'GN'

.NUM_CHANNEL: Number of channels

.PROPOSAL_MODE: Choices: 'Level', 'Joint'
```

**FRCNN**

```python
.BATCH_PER_IM: Number of total proposals selected. Will divide into fg and bg by given ratio

.BBOX_REG_WEIGHTS: Bounding box regression weights

.FG_RATIO: Fg ratio for proposals selection

.FG_THRESH: Threshold how to divide fg and bg selection

.MODE_MASK: Whether to train mask head
```

**MRCNN**

```python
.ACCURATE_PASTE: Slightly more aligned results, but very slow on numpy

.HEAD_DIM: Head dimension
```

**PREPROC**

```python
.MAX_SIZE: Maximum edge size

.PIXEL_MEAN: Pixel mean (on the training data set)

.PIXEL_STD: Pixel std (on the training data set)

.SHORT_EDGE_SIZE: Size to resize the image to (inference), while not exceeding max size

.TRAIN_SHORT_EDGE_SIZE: The size to resize the image to (training), while not exceeding max size. [min, max] to sample
```

**RPN**

```python
.ANCHOR_RATIOS: Anchor ratios

.ANCHOR_SIZES: Anchor sizes

.ANCHOR_STRIDE: Anchor stride

.BATCH_PER_IM: Total (across FPN levels) number of anchors that are marked valid

.CROWD_OVERLAP_THRESH: Anchors which overlap with a crowd box (IOA larger than threshold) will be ignored. Setting this
to a value larger than 1.0 will disable the feature. It is disabled by default because Detectron does not do this.

.FG_RATIO: Fg ratio among selected RPN anchors

.HEAD_DIM: Deprecated

.MIN_SIZE: Minimal size length for proposals

.NEGATIVE_ANCHOR_THRESH: Negative anchor threshold

.POSITIVE_ANCHOR_THRESH: Positive anchor threshold

.PER_LEVEL_NMS_TOPK: Number of top k proposals after carrying out nms (inference). selection per level proposals

.TRAIN_PER_LEVEL_NMS_TOPK: Number of proposals after carrying out nms (training)

.TRAIN_PRE_NMS_TOPK: Number of top k proposals before carrying out nms (training)

.PRE_NMS_TOPK: Number of top k proposals before carrying out nms (inference)

.TRAIN_POST_NMS_TOPK: Number of proposals after carrying out nms (training)

.POST_NMS_TOPK: Number of proposals after carrying out nms (inference)
```

**OUTPUT**

```python
.FRCNN_NMS_THRESH: Nms threshold for output. nms being performed per class prediction

.RESULTS_PER_IM: Number of output detection results

.RESULT_SCORE_THRESH: Threshold for detection result

.NMS_THRESH_CLASS_AGNOSTIC: Nms threshold for output. nms being performed over all class predictions
```

TRAINER: options: 'horovod', 'replicated'. Note that Horovod trainer is not available when TF2 is installed

**TRAIN**

```python
.LR_SCHEDULE: "1x" schedule in detectron.  LR_SCHEDULE means equivalent steps when the total batch size is 8.
               It can be either a string like "3x" that refers to standard convention, or a list of int.
               LR_SCHEDULE=3x is the same as LR_SCHEDULE=[420000, 500000, 540000], which
               means to decrease LR at steps 420k and 500k and stop training at 540k.
               When the total bs!=8, the actual iterations to decrease learning rate, and
               the base learning rate are computed from BASE_LR and LR_SCHEDULE.
               Therefore, there is *no need* to modify the config if you only change the number of GPUs.

.EVAL_PERIOD: Will call eval callback every eval period

.CHECKPOINT_PERIOD: Will save model weights every checkpoint period

.WEIGHT_DECAY: Regularization weight decay

.BASE_LR: Base learning rate

.WARMUP: In terms of iterations. This is not affected by #GPUs

.WARMUP_INIT_LR: Defined for total batch size=8. Otherwise it will be adjusted automatically

.STEPS_PER_EPOCH: Steps per epoch. One steps is equivalent to the forward/backward path of one image.

.STARTING_EPOCH: Starting epoch. Useful when restarting training.

.LOG_DIR: Log dir
```

"""

import os
from typing import List, Mapping, Tuple

import numpy as np
from lazy_imports import try_import

from .....utils.metacfg import AttrDict
from .....utils.settings import TypeOrStr, get_type

with try_import() as import_guard:
    from tensorpack.tfutils import collect_env_info  # pylint: disable=E0401
    from tensorpack.utils import logger  # pylint: disable=E0401

    # pylint: disable=import-error
    from tensorpack.utils.gpu import get_num_gpu

    # pylint: enable=import-error


__all__ = ["train_frcnn_config", "model_frcnn_config"]


def model_frcnn_config(config: AttrDict, categories: Mapping[int, TypeOrStr], print_summary: bool = True) -> None:
    """
    Sanity checks for Tensorpack Faster-RCNN config settings, where the focus lies on the model for predicting.
    It will update the config instance.

    :param config: Configuration instance as an AttrDict
    :param categories: Dict with category ids (int) and category names as keys and values.
    :param print_summary: Will optionally print the summary, the full learning rate schedule and the number of steps.
    """

    config.freeze(False)

    categories = {key: get_type(categories[val]) for key, val in enumerate(categories, 1)}
    categories[0] = get_type("background")
    config.DATA.CLASS_NAMES = list(categories.values())
    config.DATA.CLASS_DICT = categories
    config.DATA.NUM_CATEGORY = len(config.DATA.CLASS_NAMES) - 1

    assert config.BACKBONE.BOTTLENECK in ["resnet", "resnext_32xd4"], config.BACKBONE.BOTTLENECK
    assert config.BACKBONE.NORM in ["FreezeBN", "SyncBN", "GN", "None"], config.BACKBONE.NORM
    if config.BACKBONE.NORM != "FreezeBN":
        assert not config.BACKBONE.FREEZE_AFFINE
    assert config.BACKBONE.FREEZE_AT in [0, 1, 2]

    config.RPN.NUM_ANCHOR = len(config.RPN.ANCHOR_SIZES) * len(config.RPN.ANCHOR_RATIOS)
    assert len(config.FPN.ANCHOR_STRIDES) == len(config.RPN.ANCHOR_SIZES)
    config.FPN.RESOLUTION_REQUIREMENT = config.FPN.ANCHOR_STRIDES[3]

    size_mult = config.FPN.RESOLUTION_REQUIREMENT * 1.0
    config.PREPROC.MAX_SIZE = np.ceil(config.PREPROC.MAX_SIZE / size_mult) * size_mult
    assert config.FPN.PROPOSAL_MODE in ["Level", "Joint"]
    assert config.FPN.FRCNN_HEAD_FUNC.endswith("_head")
    if config.MODE_MASK:
        assert config.FPN.MRCNN_HEAD_FUNC.endswith("_head")
    assert config.FPN.NORM in ["None", "GN"]

    if config.FPN.CASCADE:
        # the first threshold is the proposal sampling threshold
        assert config.CASCADE.IOUS[0] == config.FRCNN.FG_THRESH
        assert len(config.CASCADE.BBOX_REG_WEIGHTS) == len(config.CASCADE.IOUS)

    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    config.NUM_GPUS = get_num_gpu()

    config.freeze()
    if print_summary:
        logger.info("Config: ------------------------------------------\n %s", config)


def train_frcnn_config(config: AttrDict) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], int]:
    """
    Enhances the config instance by some parameters which are necessary for setting up the training
    Run some sanity checks, and populate some configs from others

    :param config: The model configuration
    :return: A tuple with warmup learning rate schedule
    """

    config.freeze(False)

    train_scales = config.PREPROC.TRAIN_SHORT_EDGE_SIZE
    if isinstance(train_scales, (list, tuple)) and train_scales[1] - train_scales[0] > 100:
        # don't autotune if augmentation is on
        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    os.environ["TF_AUTOTUNE_THRESHOLD"] = "1"
    assert config.TRAINER in ["replicated"], config.TRAINER

    # setup NUM_GPUS
    assert "OMPI_COMM_WORLD_SIZE" not in os.environ
    logger.set_logger_dir(config.TRAIN.LOG_DIR, "d")
    number_gpu = get_num_gpu()
    logger.info("Environment Information:\n %s", collect_env_info())

    assert number_gpu > 0, "Has to train with GPU!"
    assert (
        number_gpu % 8 == 0 or 8 % number_gpu == 0
    ), f"Can only train with 1,2,4 or >=8 GPUs, but found {number_gpu} GPUs"

    learning_rate = config.TRAIN.LR_SCHEDULE
    if isinstance(learning_rate, str):
        if learning_rate.endswith("x"):
            lr_schedule_k_iter = {f"{k}x": [180 * k - 120, 180 * k - 40, 180 * k] for k in range(2, 10)}
            lr_schedule_k_iter["1x"] = [120, 160, 180]
            config.TRAIN.LR_SCHEDULE = [x * 1000 for x in lr_schedule_k_iter[learning_rate]]
        else:
            config.TRAIN.LR_SCHEDULE = eval(learning_rate)  # pylint: disable=W0123

    config.TRAIN.NUM_GPUS = number_gpu
    # Convert some config, that might be stored as strings
    config.TRAIN.BASE_LR = float(config.TRAIN.BASE_LR)
    config.TRAIN.WARMUP_INIT_LR = float(config.TRAIN.WARMUP_INIT_LR)
    config.TRAIN.WEIGHT_DECAY = float(config.TRAIN.WEIGHT_DECAY)
    # Compute the training schedule from the number of GPUs ...
    step_num = config.TRAIN.STEPS_PER_EPOCH
    # warmup is step based, lr is epoch based
    init_lr = config.TRAIN.WARMUP_INIT_LR * min(8.0 / config.TRAIN.NUM_GPUS, 1.0)
    warmup_schedule = [(0, init_lr), (config.TRAIN.WARMUP, config.TRAIN.BASE_LR)]
    warmup_end_epoch = config.TRAIN.WARMUP * 1.0 / step_num
    lr_schedule = [(int(warmup_end_epoch + 0.5), config.TRAIN.BASE_LR)]

    factor = 8.0 / config.TRAIN.NUM_GPUS
    for idx, steps in enumerate(config.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append((steps * float(factor) // step_num, config.TRAIN.BASE_LR * float(mult)))

    config.freeze()
    logger.info("Config: \n %s", str(config), config.to_dict())
    return warmup_schedule, lr_schedule, step_num
