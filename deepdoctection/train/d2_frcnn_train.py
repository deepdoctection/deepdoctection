
from typing import List, Optional, Union, Type, Sequence, Dict
import copy
import torch
from torch.utils.data import IterableDataset
import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage

from detectron2.config import get_cfg, CfgNode
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data.transforms import ResizeShortestEdge, RandomFlip
from detectron2.data import build_detection_train_loader, DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_writers, DefaultTrainer, EvalHook, launch


from ..utils.logger import logger
from ..datasets.base import DatasetBase
from ..datasets.adapter import DatasetAdapter
from ..mapper.d2struct import image_to_d2_training
from ..utils.utils import string_to_dict
from ..eval.base import MetricBase
from ..eval.eval import Evaluator
from ..eval.registry import metric_registry
from ..extern.d2detect import D2FrcnnDetector
from ..pipe.base import PredictorPipelineComponent
from ..pipe.registry import pipeline_component_registry


def _set_config(path_config_yaml: str, conf_list: List[str]) -> CfgNode:
    cfg = get_cfg()
    cfg.NMS_THRESH_CLASS_AGNOSTIC = 0.01
    cfg.merge_from_file(path_config_yaml)
    cfg.merge_from_list(conf_list)
    cfg.freeze()
    return cfg


class D2Trainer(DefaultTrainer):

    def __init__(self,cfg: CfgNode, torch_dataset: IterableDataset , mapper: DatasetMapper):
        self.dataset = torch_dataset
        self.mapper = mapper
        self.evaluator = None
        super().__init__(cfg)

    def build_train_loader(self,cfg: CfgNode):
        return build_detection_train_loader(dataset=self.dataset,
                                            mapper=self.mapper,
                                            total_batch_size=cfg.SOLVER.IMS_PER_BATCH)

    def eval_with_dd_evaluator(self,category_names: Union[str, List[str]],
                               sub_categories: Optional[Union[Dict[str, str], Dict[str, List[str]]]] = None,
                               **build_eval_kwargs: str):
        for comp in self.evaluator.pipe_component.pipe_components:
            comp.predictor.d2_predictor = copy.deepcopy(self.model).eval()
        scores = self.evaluator.run(category_names, sub_categories, True, **build_eval_kwargs)
        return scores

    def setup_evaluator(self, dataset: DatasetBase,
                        pipeline_component: PredictorPipelineComponent,
                        metric: Type[MetricBase]):
        self.evaluator = Evaluator(dataset, pipeline_component, metric, num_threads=torch.cuda.device_count() * 2)
        for comp in self.evaluator.pipe_component.pipe_components:
            assert isinstance(comp.predictor, D2FrcnnDetector)
            comp.predictor.d2_predictor = None


def train_d2_faster_rcnn(path_config_yaml: str,
                         dataset_train: Union[str, DatasetBase],
                         path_weights: str,
                         config_overwrite: Optional[List[str]]= None,
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

    build_val_dict: Dict[str, str] = {}
    if build_val_config is not None:
        build_val_dict = string_to_dict(",".join(build_val_config))
    if "split" not in build_val_dict:
        build_val_dict["split"] = "val"

    if config_overwrite is None:
        config_overwrite = []
    conf_list = ["MODEL.WEIGHTS", path_weights, "OUTPUT_DIR", log_dir]
    for conf in config_overwrite:
        key, val = conf.split("=", maxsplit=1)
        conf_list.extend([key, val])
    cfg = _set_config(path_config_yaml, conf_list)

    if metric_name is not None:
        metric = metric_registry.get(metric_name)

    dataset = DatasetAdapter(dataset_train,True,image_to_d2_training(False),**build_train_dict)
    augment_list = [ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN),RandomFlip()]
    mapper = DatasetMapper(is_train=True,augmentations=augment_list,image_format="BGR")

    logger.info("Config: ------------------------------------------\n %s", str(cfg))
    trainer = D2Trainer(cfg,dataset,mapper)
    trainer.resume_or_load()

    if cfg.TEST.EVAL_PERIOD > 0:
        categories = dataset_val.dataflow.categories.get_categories(filtered=True)
        category_names = dataset_val.dataflow.categories.get_categories(filtered=True, as_dict=False)
        detector = D2FrcnnDetector(path_config_yaml, path_weights,categories, config_overwrite,cfg.MODEL.DEVICE)
        pipeline_component_cls = pipeline_component_registry.get(pipeline_component_name)
        pipeline_component = pipeline_component_cls(detector)
        assert isinstance(pipeline_component, PredictorPipelineComponent)

        trainer.setup_evaluator(dataset_val, pipeline_component,metric)
        trainer.register_hooks([EvalHook(cfg.TEST.EVAL_PERIOD,lambda: trainer.eval_with_dd_evaluator(category_names = category_names,
                               sub_categories=dataset_val.dataflow.categories.cat_to_sub_cat, **build_val_dict))])
    return trainer.train()

