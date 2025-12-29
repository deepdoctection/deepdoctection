<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>


# Evaluation

In many situation we are not interested in evaluating raw predictions of a model but on results which have been 
polished through several post-processing steps. In other situations, we want to measure accuracy/precision etc. not 
after running one but several models. 

!!! note "Example"

    For example, getting the HTML representation of a table requires output from several predictors.
    Evaluating along a pipeline allows us to see how model prediction(s) and post processing works in conjunction. 

**deep**doctection comes equipped with an Evaluator that allows us to run evaluation not on a model but on a
pipeline component or a full pipeline.   

We will take a document layout analysis model that has been trained on `Publaynet` and want to evaluate this model 
on `Doclaynet`. 

!!! info

    `Publaynet` is a dataset with images from scientific, i.e. medical research papers. It has around 350K samples with
     five different layout types: `figure`, `table`, `list`, `text` and `title`. It is quite versatile on its domain.
    `Doclaynet` is a dataset annotated by humans with images from documents from Finance, Science, Patents, 
     Tenders, Law texts and Manuals. It is more diverse than `Publaynet` and has around 80K samples.


```python
config_yaml_path = dd.ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout_inf_only.pt")
weights_path = dd.ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout_inf_only.pt")
categories = dd.ModelCatalog.get_profile("layout/d2_model_0829999_layout_inf_only.pt").categories
layout_detector = dd.D2FrcnnDetector(config_yaml_path,weights_path,categories)
layout_service = dd.ImageLayoutService(layout_detector)
```

Next, we need a metric.


```python
coco_metric = dd.get_metric("coco")
```

Now for the dataset. Doclaynet has several other labels but there is a mapping that collapses all Doclaynet labels into
Publaynet labels. 


```python
doclaynet.dataflow.categories.get_categories()
```

??? info "Output"

    <pre>
    {1: LayoutType.CAPTION, 
     2: LayoutType.FOOTNOTE, 
     3: LayoutType.FORMULA, 
     4: LayoutType.LIST, 
     5: LayoutType.PAGE_FOOTER, 
     6: LayoutType.PAGE_HEADER, 
     7: LayoutType.FIGURE, 
     8: LayoutType.SECTION_HEADER, 
     9: LayoutType.TABLE, 
     10: LayoutType.TEXT, 
     11: LayoutType.TITLE}
    </pre>


```python
doclaynet.dataflow.categories._init_sub_categories
```

??? info "Output"

    <pre>
    {LayoutType.CAPTION: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.FOOTNOTE: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.FORMULA: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.LIST: {DatasetType.PUBLAYNET: [LayoutType.LIST]},
    LayoutType.PAGE_FOOTER: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.PAGE_HEADER: {DatasetType.PUBLAYNET: [LayoutType.TITLE]},
    LayoutType.FIGURE: {DatasetType.PUBLAYNET: [LayoutType.FIGURE]},
    LayoutType.SECTION_HEADER: {DatasetType.PUBLAYNET: [LayoutType.TITLE]},
    LayoutType.TABLE: {DatasetType.PUBLAYNET: [LayoutType.TABLE]},
    LayoutType.TEXT: {DatasetType.PUBLAYNET: [LayoutType.TEXT]},
    LayoutType.TITLE: {DatasetType.PUBLAYNET: [LayoutType.TITLE]}}
    </pre>


The sub category `DatasetType.PUBLAYNET` provides the mapping into Publaynet labels.


```python
cat_to_sub_cat = doclaynet.dataflow.categories.get_sub_categories()
cat_to_sub_cat = {key:val[0] for key, val in cat_to_sub_cat.items()}
doclaynet.dataflow.categories.set_cat_to_sub_cat(cat_to_sub_cat)
```

Now, that dataset, pipeline component and metric have been setup, we can build the evaluator.


```python
evaluator = dd.Evaluator(dataset=doclaynet,
						 component_or_pipeline=layout_service, 
						 metric=coco_metric)
```

We start evaluation using the `run` method. `max_datapoints` limits the number of samples to at most 100 samples. The 
`val` split is used by default.


```python
evaluator = dd.Evaluator(doclaynet,layout_service, coco_metric)
output= evaluator.run(max_datapoints=100)
```

??? info "Output"

    ```
    creating index...
    index created!
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.12s).
    Accumulating evaluation results...
    DONE (t=0.03s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.147
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.195
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.144
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.010
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.022
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.200
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.100
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.171
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.009
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.031
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.231
    ```


The result shows that Doclaynet has a very different layout compared to Publaynet where the model has been trained on. 
To get a feeling, results on the Publaynet test split are in the range of 0.9+ !

## Example: Evaluation of Table Recognition

In this example we will be showing how to evaluate a table recognition pipeline. We will be comparing
HTML representations from the Pubtabnet evaluation set and will be using the TEDS metric as described in 
[Zhong et. all](https://arxiv.org/abs/1911.10683). We will be evaluating the HTML skeleton only and discard text.

```python
import os
from typing import List

import deepdoctection as dd


def get_table_recognizer():
    cfg = dd.set_config_by_yaml("~/.cache/deepdoctection/configs/dd/conf_dd_one.yaml")
    pipe_component_list: List[dd.PipelineComponent] = []

    crop = dd.ImageCroppingService(category_names=dd.LayoutType.TABLE)
    pipe_component_list.append(crop)

    cell_config_path = dd.ModelCatalog.get_full_path_configs("cell/d2_model_1849999_cell_inf_only.ts")
    cell_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs("cell/d2_model_1849999_cell_inf_only.ts")
    categories_cell = dd.ModelCatalog.get_profile("cell/d2_model_1849999_cell_inf_only.ts").categories
    d_cell = dd.D2FrcnnTracingDetector(cell_config_path, cell_weights_path, categories_cell)
	
    item_config_path = dd.ModelCatalog.get_full_path_configs("item/d2_model_1639999_item_inf_only.ts")
    item_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs("item/d2_model_1639999_item_inf_only.ts")
    categories_item = dd.ModelCatalog.get_profile("item/d2_model_1639999_item_inf_only.ts").categories
    d_item = dd.D2FrcnnTracingDetector(item_config_path, item_weights_path, categories_item)

	cell_detect_result_generator = dd.DetectResultGenerator(categories_name_as_key=d_cell.categories.get_categories
	(as_dict=True, name_as_key=True))
    cell = dd.SubImageLayoutService(sub_image_detector=d_cell, 
									sub_image_names=dd.LayoutType.TABLE,
									service_ids=None,
									detect_result_generator=cell_detect_result_generator)
    pipe_component_list.append(cell)
	
	item_detect_result_generator = dd.DetectResultGenerator(categories_name_as_key=d_item.categories.get_categories
	(as_dict=True, name_as_key=True))
    item = dd.SubImageLayoutService(sub_image_detector=d_item, 
									sub_image_names=dd.LayoutType.TABLE, 
									service_ids=None,
									detect_result_generator=item_detect_result_generator)
    pipe_component_list.append(item)

    table_segmentation = dd.TableSegmentationService(
        cfg.SEGMENTATION.ASSIGNMENT_RULE,
        cfg.SEGMENTATION.THRESHOLD_ROWS,
        cfg.SEGMENTATION.THRESHOLD_COLS,
        cfg.SEGMENTATION.FULL_TABLE_TILING,
        cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
        cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
        dd.LayoutType.TABLE,
        [dd.CellType.HEADER,dd.CellType.BODY,dd.LayoutType.CELL],
        [dd.LayoutType.ROW,dd.LayoutType.COLUMN],
        [dd.CellType.ROW_NUMBER, dd.CellType.COLUMN_NUMBER],
        cfg.SEGMENTATION.STRETCH_RULE,
    )
	
    pipe_component_list.append(table_segmentation)
    table_segmentation_refinement = dd.TableSegmentationRefinementService([LayoutType.TABLE],
                                                                          [LayoutType.CELL,
                                                                           CellType.COLUMN_HEADER,
                                                                           CellType.PROJECTED_ROW_HEADER,
                                                                           CellType.SPANNING,
                                                                           CellType.ROW_HEADER,
                                                                          ])
    pipe_component_list.append(table_segmentation_refinement)
    return dd.DoctectionPipe(pipeline_component_list=pipe_component_list)

	
pubtabnet = dd.Pubtabnet()
teds = dd.metric_registry.get("teds")
teds.structure_only = True
pipe = get_table_recognizer()
evaluator = dd.Evaluator(pubtabnet, pipe, teds)
out = evaluator.run(max_datapoints=1000, 
					split="val", dd_pipe_like=True)
print(out)
```

??? info "Output"

    [{'teds_score': 0.810958120214249, 'num_samples': 441}]
