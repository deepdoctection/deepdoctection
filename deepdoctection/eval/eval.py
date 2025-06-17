# -*- coding: utf-8 -*-
# File: eval.py

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
Module for `Evaluator`
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Generator, Literal, Mapping, Optional, Type, Union, overload

import numpy as np
from lazy_imports import try_import

from ..dataflow import CacheData, DataFlow, DataFromList, MapData
from ..datapoint.image import Image
from ..datasets.base import DatasetBase
from ..mapper.cats import filter_cat, remove_cats
from ..mapper.d2struct import to_wandb_image
from ..mapper.misc import maybe_load_image, maybe_remove_image, maybe_remove_image_from_category
from ..pipe.base import PipelineComponent
from ..pipe.common import PageParsingService
from ..pipe.concurrency import MultiThreadPipelineComponent
from ..pipe.doctectionpipe import DoctectionPipe
from ..utils.logger import LoggingRecord, logger
from ..utils.settings import DatasetType, LayoutType, TypeOrStr, get_type
from ..utils.types import PixelValues
from ..utils.viz import interactive_imshow
from .base import MetricBase

with try_import() as wb_import_guard:
    import wandb  # pylint:disable=W0611
    from wandb import Artifact, Table

__all__ = ["Evaluator"]


class Evaluator:
    """
    The API for evaluating pipeline components or pipelines on a given dataset. For a given model, a given dataset and
    a given metric, this class will stream the dataset, call the predictor(s) and will evaluate the predictions against
    the ground truth with respect to the given metric.

    After initializing the evaluator the process itself will start after calling the `run`.

    The following takes place under the hood:

    Setup of the dataflow according to the build- and split inputs. The `datasets.DataFlowBaseBuilder.build` will
    be invoked twice as one dataflow must be kept with its ground truth while the other must go through an annotation
    erasing process and after that passing the predictor. Predicted and gt datapoints will be converted into the
    required metric input format and dumped into lists. Both lists will be passed to `MetricBase.get_distance`.

    Note:
        You can evaluate the predictor on a subset of categories by filtering the ground truth dataset. When using
        the coco metric all predicted objects that are not in the set of filtered objects will be not taken into
        account.

    Example:
        ```python
        publaynet = get_dataset("publaynet")
        publaynet.dataflow.categories.filter_categories(categories=["TEXT","TITLE"])
        coco_metric = metric_registry.get("coco")
        profile = ModelCatalog.get_profile("layout/d2_model_0829999_layout_inf_only.pt")
        path_weights = ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout_inf_only.pt")
        path_config_yaml= ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout_inf_only.pt")

        layout_detector = D2FrcnnDetector(path_config_yaml, path_weights, profile.categories)
        layout_service = ImageLayoutService(layout_detector)
        evaluator = Evaluator(publaynet, layout_service, coco_metric)

        output = evaluator.run(max_datapoints=10)
        ```

    For another example check the script in `Evaluation` of table recognition`
    """

    def __init__(
        self,
        dataset: DatasetBase,
        component_or_pipeline: Union[PipelineComponent, DoctectionPipe],
        metric: Union[Type[MetricBase], MetricBase],
        num_threads: int = 2,
        run: Optional[wandb.sdk.wandb_run.Run] = None,
    ) -> None:
        """
        Evaluating a pipeline component on a dataset with a given metric.

        Args:
            dataset: dataset
            component_or_pipeline: A pipeline component with predictor and annotation factory.
            metric: metric
        """

        self.dataset = dataset
        self.pipe_component: Optional[MultiThreadPipelineComponent] = None
        self.pipe: Optional[DoctectionPipe] = None

        # when passing a component, we will process prediction on num_threads
        if isinstance(component_or_pipeline, PipelineComponent):
            logger.info(
                LoggingRecord(
                    f"Building multi threading pipeline component to increase prediction throughput. "
                    f"Using {num_threads} threads"
                )
            )
            pipeline_components: list[PipelineComponent] = []

            for _ in range(num_threads - 1):
                copy_pipe_component = component_or_pipeline.clone()
                pipeline_components.append(copy_pipe_component)

            pipeline_components.append(component_or_pipeline)

            self.pipe_component = MultiThreadPipelineComponent(
                pipeline_components=pipeline_components,
                pre_proc_func=maybe_load_image,
                post_proc_func=maybe_remove_image,
            )
        else:
            self.pipe = component_or_pipeline

        self.metric = metric
        if not isinstance(metric, MetricBase):
            self.metric = self.metric()  # type: ignore

        self._sanity_checks()

        self.wandb_table_agent: Optional[WandbTableAgent]
        if run is not None:
            if self.dataset.dataset_info.type == DatasetType.OBJECT_DETECTION:
                self.wandb_table_agent = WandbTableAgent(
                    run,
                    self.dataset.dataset_info.name,
                    50,
                    self.dataset.dataflow.categories.get_categories(filtered=True),
                )
            elif self.dataset.dataset_info.type == DatasetType.TOKEN_CLASSIFICATION:
                if hasattr(self.metric, "sub_cats"):
                    sub_cat_key, sub_cat_val_list = list(self.metric.sub_cats.items())[0]
                    sub_cat_val = sub_cat_val_list[0]
                    sub_cats = {sub_cat_key: sub_cat_val}
                    self.wandb_table_agent = WandbTableAgent(
                        run,
                        self.dataset.dataset_info.name,
                        50,
                        self.dataset.dataflow.categories.get_categories(filtered=True),
                        self.dataset.dataflow.categories.get_sub_categories(
                            categories=sub_cat_key,
                            sub_categories=sub_cats,
                            keys=False,
                            values_as_dict=True,
                            name_as_key=False,
                        )[sub_cat_key][sub_cat_val],
                        sub_cats,
                    )
                else:
                    raise AttributeError(
                        "metric has no attribute sub_cats and cannot be used for token classification datasets"
                    )
            else:
                raise NotImplementedError()

        else:
            self.wandb_table_agent = None

    @overload
    def run(
        self, output_as_dict: Literal[False] = False, **dataflow_build_kwargs: Union[str, int]
    ) -> list[dict[str, float]]:
        ...

    @overload
    def run(self, output_as_dict: Literal[True], **dataflow_build_kwargs: Union[str, int]) -> dict[str, float]:
        ...

    def run(
        self, output_as_dict: bool = False, **dataflow_build_kwargs: Union[str, int]
    ) -> Union[list[dict[str, float]], dict[str, float]]:
        """
        Start evaluation process and return the results.

        Args:
            output_as_dict: Return result in a list or dict.
            dataflow_build_kwargs: Pass the necessary arguments in order to build the dataflow, e.g. `split`,
                                  `build_mode`, `max_datapoints` etc.

        Returns:
            dict with metric results.
        """

        df_gt = self.dataset.dataflow.build(**dataflow_build_kwargs)
        df_pr = self.dataset.dataflow.build(**dataflow_build_kwargs)

        df_pr = MapData(df_pr, deepcopy)
        df_pr = self._clean_up_predict_dataflow_annotations(df_pr)
        df_pr = self._run_pipe_or_component(df_pr)

        logger.info(LoggingRecord("Starting evaluation..."))
        result = self.metric.get_distance(df_gt, df_pr, self.dataset.dataflow.categories)
        self.metric.print_result()

        if self.wandb_table_agent:
            self.wandb_table_agent.log()

        if output_as_dict:
            return self.metric.result_list_to_dict(result)

        return result

    def _sanity_checks(self) -> None:
        assert self.dataset.dataflow.categories is not None

    def _run_pipe_or_component(self, df_pr: DataFlow) -> DataFlow:
        if self.pipe_component:
            self.pipe_component.put_task(df_pr)
            logger.info(LoggingRecord("Predicting objects..."))
            df_pr_list = self.pipe_component.start()
            if self.wandb_table_agent is not None:
                df_pr_list = [self.wandb_table_agent.dump(dp) for dp in df_pr_list]
            return DataFromList(df_pr_list)
        df_pr = MapData(df_pr, maybe_load_image)
        df_pr = self.pipe.analyze(dataset_dataflow=df_pr, output="image")  # type: ignore
        # deactivate timer for components
        for comp in self.pipe.pipe_component_list:  # type: ignore
            comp.timer_on = False
        df_pr = MapData(df_pr, maybe_remove_image)
        df_list = CacheData(df_pr).get_cache()
        if self.wandb_table_agent is not None:
            df_list = [self.wandb_table_agent.dump(dp) for dp in df_list]
        return DataFromList(df_list)

    def _clean_up_predict_dataflow_annotations(self, df_pr: DataFlow) -> DataFlow:
        # will use the first pipe component of MultiThreadPipelineComponent to get meta annotation
        pipe_or_component = self.pipe_component.pipe_components[0] if self.pipe_component is not None else self.pipe
        meta_anns = pipe_or_component.get_meta_annotation()  # type: ignore
        possible_cats_in_datapoint = self.dataset.dataflow.categories.get_categories(as_dict=False, filtered=True)

        # clean-up procedure depends on the dataset type
        if self.dataset.dataset_info.type == DatasetType.OBJECT_DETECTION:
            # we keep all image annotations that will not be generated through processing
            anns_to_keep = {ann for ann in possible_cats_in_datapoint if ann not in meta_anns.image_annotations}
            sub_cats_to_remove = meta_anns.sub_categories
            relationships_to_remove = meta_anns.relationships
            # removing annotations takes place in three steps: First we remove all image annotations. Then, with all
            # remaining image annotations we check, if the image attribute (with Image instance !) is not empty and
            # remove it as well, if necessary. In the last step we remove all sub categories and relationships, if
            # generated in pipeline.
            df_pr = MapData(df_pr, filter_cat(anns_to_keep, possible_cats_in_datapoint))  # pylint: disable=E1120
            df_pr = MapData(df_pr, maybe_remove_image_from_category(anns_to_keep))
            df_pr = MapData(
                df_pr,
                remove_cats(sub_categories=sub_cats_to_remove, relationships=relationships_to_remove),
            )

        elif self.dataset.dataset_info.type == DatasetType.SEQUENCE_CLASSIFICATION:
            summary_sub_cats_to_remove = meta_anns.summaries
            df_pr = MapData(df_pr, remove_cats(summary_sub_categories=summary_sub_cats_to_remove))

        elif self.dataset.dataset_info.type == DatasetType.TOKEN_CLASSIFICATION:
            sub_cats_to_remove = meta_anns.sub_categories
            df_pr = MapData(df_pr, remove_cats(sub_categories=sub_cats_to_remove))
        else:
            raise NotImplementedError()

        return df_pr

    def compare(self, interactive: bool = False, **kwargs: Union[str, int]) -> Generator[PixelValues, None, None]:
        """
        Visualize ground truth and prediction datapoint. Given a dataflow config it will run predictions per sample
        and concat the prediction image (with predicted bounding boxes) with ground truth image.

        Args:
            interactive: If set to True will open an interactive image, otherwise it will return a `np.array` that
                        can be displayed differently (e.g. `matplotlib`). Note that, if the interactive mode is being
                        used, more than one sample can be iteratively be displayed.
            kwargs: Dataflow configs for displaying specific image splits and visualisation configs:
                   `show_tables`, `show_layouts`, `show_table_structure`, `show_words`

        Returns:
            Image as `np.array`
        """

        show_tables = kwargs.pop("show_tables", True)
        show_layouts = kwargs.pop("show_layouts", True)
        show_table_structure = kwargs.pop("show_table_structure", True)
        show_words = kwargs.pop("show_words", False)
        show_token_class = kwargs.pop("show_token_class", True)
        ignore_default_token_class = kwargs.pop("ignore_default_token_class", False)
        floating_text_block_categories = kwargs.pop("floating_text_block_categories", None)
        include_residual_text_containers = kwargs.pop("include_residual_Text_containers", True)

        df_gt = self.dataset.dataflow.build(**kwargs)
        df_pr = self.dataset.dataflow.build(**kwargs)
        df_gt = MapData(df_gt, maybe_load_image)
        df_pr = MapData(df_pr, maybe_load_image)
        df_pr = MapData(df_pr, deepcopy)
        df_pr = self._clean_up_predict_dataflow_annotations(df_pr)

        page_parsing_component = PageParsingService(
            text_container=LayoutType.WORD,
            floating_text_block_categories=floating_text_block_categories,  # type: ignore
            include_residual_text_container=bool(include_residual_text_containers),
        )
        df_gt = page_parsing_component.predict_dataflow(df_gt)

        if self.pipe_component:
            pipe_component = self.pipe_component.pipe_components[0]
            df_pr = pipe_component.predict_dataflow(df_pr)
            df_pr = page_parsing_component.predict_dataflow(df_pr)

        elif self.pipe:
            df_pr = self.pipe.analyze(dataset_dataflow=df_pr)
        else:
            raise ValueError("Neither pipe_component nor pipe has been defined")

        df_pr.reset_state()
        df_gt.reset_state()
        for dp_gt, dp_pred in zip(df_gt, df_pr):
            img_gt, img_pred = dp_gt.viz(
                show_tables=show_tables,
                show_layouts=show_layouts,
                show_table_structure=show_table_structure,
                show_words=show_words,
                show_token_class=show_token_class,
                ignore_default_token_class=ignore_default_token_class,
            ), dp_pred.viz(
                show_tables=show_tables,
                show_layouts=show_layouts,
                show_table_structure=show_table_structure,
                show_words=show_words,
                show_token_class=show_token_class,
                ignore_default_token_class=ignore_default_token_class,
            )
            img_concat = np.concatenate((img_gt, img_pred), axis=1)
            if interactive:
                interactive_imshow(img_concat)
            else:
                yield img_concat


class WandbTableAgent:
    """
    A class that creates a W&B table of sample predictions and sends them to the W&B server.

    Example:
        ```python
        df ... # some dataflow
        agent = WandbTableAgent(myrun,"MY_DATASET",50,{"1":"FOO"})
        for dp in df:
            agent.dump(dp)

        agent.log()
        ```
    """

    def __init__(
        self,
        wandb_run: wandb.sdk.wandb_run.Run,
        dataset_name: str,
        num_samples: int,
        categories: Mapping[int, TypeOrStr],
        sub_categories: Optional[Mapping[int, TypeOrStr]] = None,
        cat_to_sub_cat: Optional[Mapping[TypeOrStr, TypeOrStr]] = None,
    ):
        """
        Args:
            wandb_run: An `wandb.run` instance for tracking. Use `run=wandb.init(project=project, config=config,
                      **kwargs)` to generate a `run`.
            dataset_name: name for tracking
            num_samples: When dumping images to a table it will stop adding samples after `num_samples` instances
            categories: dict of all possible categories
            sub_categories: dict of sub categories. If provided, these categories will define the classes for the
                            table
            cat_to_sub_cat: dict of category to sub category keys. Suppose your category `foo` has a sub category
                           defined by the key `sub_foo`. The range sub category values must then be given by
                           `sub_categories` and to extract the sub category values one must pass `{"foo": "sub_foo"}`
        """

        self.dataset_name = dataset_name
        self.num_samples = num_samples
        self.categories = categories
        self.sub_categories = sub_categories
        if cat_to_sub_cat is None:
            cat_to_sub_cat = {}
        self.cat_to_sub_cat = {get_type(cat): get_type(sub_cat) for cat, sub_cat in cat_to_sub_cat.items()}
        self._run = wandb_run
        self._counter = 0

        # Table logging utils
        self._table_cols: list[str] = ["file_name", "image"]
        self._table_rows: list[Any] = []
        self._table_ref = None

    def dump(self, dp: Image) -> Image:
        """
        Dump image to a table. Add this while iterating over samples. After `num_samples` it will stop appending samples
        to the table

        Args:
            dp: `Image` instance

        Returns:
            `Image` instance
        """
        if self.num_samples > self._counter:
            dp = maybe_load_image(dp)
            self._table_rows.append(
                to_wandb_image(self.categories, self.sub_categories, self.cat_to_sub_cat)(dp)  # pylint: disable=E1102
            )
            dp = maybe_remove_image(dp)
            self._counter += 1
        return dp

    def reset(self) -> None:
        """
        Reset table rows
        """
        self._table_rows = []
        self._counter = 0

    def _build_table(self) -> Table:
        """
        Builds `wandb.Table` instance for logging evaluation

        Returns:
            Table object to log evaluation
        """
        return Table(columns=self._table_cols, data=self._table_rows)

    def log(self) -> None:
        """
        Log WandB table and maybe send table to WandB server
        """
        table = self._build_table()
        self._run.log({self.dataset_name: table})
        self.reset()

    def _use_table_as_artifact(self) -> None:
        """
        This function logs the given table as artifact and calls `use_artifact`
        on it so tables from next iterations can use the reference of already uploaded images.
        """

        eval_art = Artifact(self._run.id + self.dataset_name, type="dataset")
        eval_art.add(self._build_table(), self.dataset_name)
        self._run.use_artifact(eval_art)
        eval_art.wait()
        self._table_ref = eval_art.get(self.dataset_name).data  # type: ignore
