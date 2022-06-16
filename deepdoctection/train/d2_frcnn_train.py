
from typing import List, Optional, Union, Type, Sequence, Dict

import torch

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data.transforms import ResizeShortestEdge, RandomFlip
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_writers


from ..utils.logger import logger
from ..datasets.base import DatasetBase
from ..datasets.adapter import DatasetAdapter
from ..mapper.d2struct import image_to_d2_training
from ..utils.utils import string_to_dict
from ..eval.base import MetricBase


def _set_config(path_config_yaml: str, conf_list: List[str]) -> "CfgNode":
    cfg = get_cfg()
    cfg.merge_from_file(path_config_yaml)
    cfg.merge_from_list(conf_list)
    cfg.TRAIN_SHORT_EDGE_SIZE = [400,600]
    cfg.MAX_SIZE = 1333
    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    return cfg


def train_d2_faster_rcnn(path_config_yaml: str,
                         dataset_train: Union[str, DatasetBase],
                         path_weights: str,
                         config_overwrite: Optional[List[str]],
                         log_dir: str = "train_log/frcnn",
                         build_train_config: Optional[Sequence[str]] = None,
                         dataset_val: Optional[DatasetBase] = None,
                         build_val_config: Optional[Sequence[str]] = None,
                         metric_name: Optional[str] = None,
                         metric: Optional[Type[MetricBase]] = None,
                         pipeline_component_name: Optional[str] = None,
                         ):

    build_train_dict: Dict[str, str] = {}
    if build_train_config is not None:
        build_train_dict = string_to_dict(",".join(build_train_config))
    if "split" not in build_train_dict:
        build_train_dict["split"] = "train"

    if config_overwrite is None:
        config_overwrite = []
    conf_list = ["MODEL.WEIGHTS", path_weights, "OUTPUT_DIR", log_dir]
    for conf in config_overwrite:
        key, val = conf.split("=", maxsplit=1)
        conf_list.extend([key, val])
    cfg = _set_config(path_config_yaml, conf_list)



    model = build_model(cfg)
    model.train()
    logger.info("Model:\n{}".format(model))

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS).get("iteration", -1) + 1
    )
    max_iter = 5 #cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    dataset = DatasetAdapter(dataset_train,False,image_to_d2_training(False),**build_train_dict)

    try:
        total_passes = (max_iter-start_iter) * cfg.SOLVER.IMS_PER_BATCH / len(dataset)
        logger.info("Total passes of the training set is: %i", total_passes)
    except TypeError:
        logger.info("Cannot evaluate size of dataflow and total passes")

    augment_list = [ResizeShortestEdge(cfg.TRAIN_SHORT_EDGE_SIZE,cfg.MAX_SIZE),RandomFlip()]
    mapper = DatasetMapper(is_train=True,augmentations=augment_list,image_format="BGR")
    data_loader = build_detection_train_loader(dataset=dataset, mapper=mapper,total_batch_size=1)

    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            if iteration - start_iter > 0 and (
                (iteration + 1) % 1 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
