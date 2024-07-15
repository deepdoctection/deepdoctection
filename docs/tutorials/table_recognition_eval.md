# Evaluation of table recognition


The following script demonstrates how to set up a table recognition pipeline and how to evaluate predicted results
on a subset of the Pubtabnet validation set. We compare html representations from the evaluation set
and the predictions using TEDS metric as described in [Zhong et. all](https://arxiv.org/abs/1911.10683) . We evaluate the html structure
only and ignore text, because many open source tools perform very poorly on text from images with low resolution as
given by Pubtabnet.


```python

    import os
    from typing import List

    import deepdoctection as dd
```

The table recognizer is identical with the table recognizing part of the dd-Analyzer. As input it expects
image datapoints with layout objects and in particular tables.

```python

def get_table_recognizer():
    cfg = dd.set_config_by_yaml("/home/janis/.cache/deepdoctection/configs/dd/conf_dd_one.yaml")
    pipe_component_list: List[PipelineComponent] = []

    crop = dd.ImageCroppingService(category_names="table")
    pipe_component_list.append(crop)

    cell_config_path = dd.ModelCatalog.get_full_path_configs(cfg.CONFIG.D2CELL)
    cell_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2CELL)
    categories_cell = dd.ModelCatalog.get_profile(cfg.WEIGHTS.D2CELL).categories
    assert categories_cell is not None
    d_cell = dd.D2FrcnnDetector(cell_config_path, cell_weights_path, categories_cell, device="gpu")
    item_config_path = dd.ModelCatalog.get_full_path_configs(cfg.CONFIG.D2ITEM)
    item_weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2ITEM)
    categories_item = dd.ModelCatalog.get_profile(cfg.WEIGHTS.D2ITEM).categories
    assert categories_item is not None
    d_item = dd.D2FrcnnDetector(item_config_path, item_weights_path, categories_item, device="gpu")

    cell = dd.SubImageLayoutService(d_cell, "table", {1: 6}, True)
    pipe_component_list.append(cell)

    item = dd.SubImageLayoutService(d_item, "table", {1: 7, 2: 8}, True)
    pipe_component_list.append(item)

    table_segmentation = dd.TableSegmentationService(
        cfg.SEGMENTATION.ASSIGNMENT_RULE,
        cfg.SEGMENTATION.IOU_THRESHOLD_ROWS
        if cfg.SEGMENTATION.ASSIGNMENT_RULE in ["iou"]
        else cfg.SEGMENTATION.IOA_THRESHOLD_ROWS,
        cfg.SEGMENTATION.IOU_THRESHOLD_COLS
        if cfg.SEGMENTATION.ASSIGNMENT_RULE in ["iou"]
        else cfg.SEGMENTATION.IOA_THRESHOLD_COLS,
        cfg.SEGMENTATION.FULL_TABLE_TILING,
        cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_ROWS,
        cfg.SEGMENTATION.REMOVE_IOU_THRESHOLD_COLS,
    )
    pipe_component_list.append(table_segmentation)
    table_segmentation_refinement = dd.TableSegmentationRefinementService([LayoutType.TABLE,
                                                                           LayoutType.TABLE_ROTATED],
                                                                          [
                                                                              LayoutType.CELL,
                                                                              CellType.COLUMN_HEADER,
                                                                              CellType.PROJECTED_ROW_HEADER,
                                                                              CellType.SPANNING,
                                                                              CellType.ROW_HEADER,
                                                                          ])
    pipe_component_list.append(table_segmentation_refinement)
    tess_ocr_config_path = os.path.join(get_configs_dir_path(), cfg.CONFIG.TESS_OCR)
    d_tess_ocr = dd.TesseractOcrDetector(tess_ocr_config_path)
    text = dd.TextExtractionService(d_tess_ocr, None, {1: 9})
    pipe_component_list.append(text)
    match = dd.MatchingService(
        parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
        child_categories="WORD",
        matching_rule=cfg.WORD_MATCHING.RULE,
        threshold=cfg.WORD_MATCHING.IOU_THRESHOLD
        if cfg.WORD_MATCHING.RULE in ["iou"]
        else cfg.WORD_MATCHING.IOA_THRESHOLD,
    )
    pipe_component_list.append(match)
    order = dd.TextOrderService(
        text_container="word",
        floating_text_block_names=["title", "text", "list"],
        text_block_names=["title", "text", "list", "cell", "head", "body"],
    )
    pipe_component_list.append(order)
    return dd.DoctectionPipe(pipeline_component_list=pipe_component_list)
```

```python

    pubtabnet = dd.Pubtabnet()
    teds = dd.metric_registry.get("teds")
    teds.structure_only = True
    pipe = get_table_recognizer()
    evaluator = dd.Evaluator(pubtabnet, pipe, teds)
    out = evaluator.run(max_datapoints=1000, split="val", dd_pipe_like=True, load_image=True)
    print(out)

    # out [{'teds_score': 0.810958120214249, 'num_samples': 441}]
    # Many samples need to be filtered before evaluation due to the fact, that OCR performs so poorly
    # (invalid string generation) such that the returned html cannot be parsed.
```
