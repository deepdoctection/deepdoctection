Evaluation of table recognition
===============================

The following script shows how to setup a table recognition pipeline and evaluating
the results on a subset of Pubtabnet validation set. We compare html representations of ground truth
and predictions as described in https://arxiv.org/abs/1911.10683 . We evaluate structure only and ignore
table text content as many open source tools perform very poorly on low level image resolutions as given
by Pubtabnet.


.. code:: ipython3

    import os
    from typing import List

    from deepdoctection.utils.metacfg import set_config_by_yaml
    from deepdoctection.utils.settings import names
    from deepdoctection.utils.systools import get_configs_dir_path
    from deepdoctection.extern.model import ModelCatalog, ModelDownloadManager
    from deepdoctection.extern.tessocr import TesseractOcrDetector
    from deepdoctection.pipe.base import PipelineComponent
    from deepdoctection.pipe.cell import SubImageLayoutService
    from deepdoctection.pipe.common import MatchingService, ImageCroppingService
    from deepdoctection.pipe.doctectionpipe import DoctectionPipe
    from deepdoctection.pipe.refine import TableSegmentationRefinementService
    from deepdoctection.pipe.segment import TableSegmentationService
    from deepdoctection.pipe.text import TextExtractionService, TextOrderService
    from deepdoctection.extern.d2detect import D2FrcnnDetector
    from deepdoctection.eval import metric_registry, Evaluator

    from deepdoctection.datasets import Pubtabnet

The table recognizer is identical with the table recognizing part of the dd-Analyzer. As input it expects
image datapoints with layout objects and in particular tables.


.. code:: ipython3

    def get_table_recognizer():
        cfg = set_config_by_yaml("/home/janis/.cache/deepdoctection/configs/dd/conf_dd_one.yaml")
        pipe_component_list: List[PipelineComponent] = []

        crop = ImageCroppingService(category_names=names.C.TAB)
        pipe_component_list.append(crop)

        cell_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2CELL)
        cell_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2CELL)
        categories_cell = ModelCatalog.get_profile(cfg.WEIGHTS.D2CELL).categories
        assert categories_cell is not None
        d_cell = D2FrcnnDetector(cell_config_path, cell_weights_path, categories_cell, device="gpu")
        item_config_path = ModelCatalog.get_full_path_configs(cfg.CONFIG.D2ITEM)
        item_weights_path = ModelDownloadManager.maybe_download_weights_and_configs(cfg.WEIGHTS.D2ITEM)
        categories_item = ModelCatalog.get_profile(cfg.WEIGHTS.D2ITEM).categories
        assert categories_item is not None
        d_item = D2FrcnnDetector(item_config_path, item_weights_path, categories_item, device="gpu")

        cell = SubImageLayoutService(d_cell, names.C.TAB, {1: 6}, True)
        pipe_component_list.append(cell)

        item = SubImageLayoutService(d_item, names.C.TAB, {1: 7, 2: 8}, True)
        pipe_component_list.append(item)

        table_segmentation = TableSegmentationService(
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
        table_segmentation_refinement = TableSegmentationRefinementService()
        pipe_component_list.append(table_segmentation_refinement)
        tess_ocr_config_path = os.path.join(get_configs_dir_path(), cfg.CONFIG.TESS_OCR)
        d_tess_ocr = TesseractOcrDetector(tess_ocr_config_path)
        text = TextExtractionService(d_tess_ocr, None, {1: 9})
        pipe_component_list.append(text)
        match = MatchingService(
            parent_categories=cfg.WORD_MATCHING.PARENTAL_CATEGORIES,
            child_categories=names.C.WORD,
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.IOU_THRESHOLD
            if cfg.WORD_MATCHING.RULE in ["iou"]
            else cfg.WORD_MATCHING.IOA_THRESHOLD,
        )
        pipe_component_list.append(match)
        order = TextOrderService(
            text_container=names.C.WORD,
            floating_text_block_names=[names.C.TITLE, names.C.TEXT, names.C.LIST],
            text_block_names=[names.C.TITLE, names.C.TEXT, names.C.LIST, names.C.CELL, names.C.HEAD, names.C.BODY],
        )
        pipe_component_list.append(order)
        return DoctectionPipe(pipeline_component_list=pipe_component_list)


.. code:: ipython3

    pubtabnet = Pubtabnet()
    teds = metric_registry.get("teds")
    teds.structure_only = True
    pipe = get_table_recognizer()
    evaluator = Evaluator(pubtabnet, pipe, teds)
    out = evaluator.run(max_datapoints=1000, split="val", dd_pipe_like=True, load_image=True)
    print(out)
    # out [{'teds_score': 0.810958120214249, 'num_samples': 441}]
    # Many samples need to be filtered before evaluation due to the fact, that OCR performs so poorly
    # (invalid string generation) such that the returned html cannot be parsed.

