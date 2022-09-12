Changing the layout parsing structure
=====================================

Layout structures are usually determined by the model. As a consequence,
these must be parsed differently depending on the model and the pipeline
in order to ensure optimal further processing.

It is to be represented here, which possibilities there are to adapt the
parsing of the document in the page format. This depends naturally also
substantially on which form of the processing in the pipeline.

.. figure:: ./pics/dd_text_order.png
   :alt: title

We assume that we get a layout structure like in the Publaynet dataset
from the model. As OCR component we further assume that we get back text
in line-items. Line items can consist of several words and represent the
smallest coherent unit above the single word. In a layout with multiple
columns, a line item represents one row in a column. Compare the
diagram.

.. code:: ipython3

    import os
    
    from deepdoctection.utils import get_configs_dir_path
    from deepdoctection.extern import TPFrcnnDetector, TextractOcrDetector
    from deepdoctection.pipe import ImageLayoutService, TextExtractionService
    
    from deepdoctection.utils import set_config_by_yaml

.. code:: ipython3

    # setting up layout detector and layout service
    categories_layout = {"1": "TEXT", "2": "TITLE", "3": "LIST", "4": "TABLE", "5": "FIGURE"}
    
    layout_config_path = os.path.join(get_configs_dir_path(), "dd/tp/conf_frcnn_layout.yaml")
    
    layout_weights_path = "/path/to/dir/model-820000.data-00000-of-00001"
    d_layout = TPFrcnnDetector(layout_config_path, layout_weights_path, categories_layout)
    layout = ImageLayoutService(d_layout, to_image=True)
    
    # setting up OCR detector and ocr service
    ocr = TextractOcrDetector(text_lines=True)
    text = TextExtractionService(ocr)
    
    # setting up first two pipeline components
    pipe_comp =[layout,text]

Matching Service
----------------

We assign line items to layout blocks. The line items are text
containers and contain a separate container annotation CHARS. The OCR
service also returns the WORD results, but these are not used further.

.. code:: ipython3

    from deepdoctection.pipe import MatchingService
    
    match = MatchingService(
            parent_categories=["TEXT",
                               "TITLE",
                               "LIST",
                               "TABLE"],
            child_categories="LINE",
            matching_rule=cfg.WORD_MATCHING.RULE,
            threshold=cfg.WORD_MATCHING.IOU_THRESHOLD
            if cfg.WORD_MATCHING.RULE in ["iou"]
            else cfg.WORD_MATCHING.IOA_THRESHOLD
            )
    pipe_comp.append(match)

Text order service
------------------

Determining the reading order of a text is done in two stages:

1.) Ordering the text within text blocks. Here, the assigned text
containers are arranged one below the other. If necessary, lines are
created and the lines are arranged to a reading order. Currently, text
is assumed to be read as in the Latin alphabet (i.e. from left to right
or from top to bottom).

2.) Ordering of the text blocks themselves. Here it must be determined
which text blocks are to be considered at all. For example, if a table
detection is included in the pipeline, a table cell would be a text
block, on the other hand, it would not necessarily make sense to include
cells in the reading order of the text blocks themselves. Therefore,
these text blocks are also called floating text blocks in the text order
service.

It may happen that text containers are not assigned to text blocks. If
you do not want to lose any text, you can consider unassigned text
containers as text blocks and include them in the order according to
2.).

.. code:: ipython3

    from deepdoctection.pipe import TextOrderService
    
    text_order = TextOrderService(text_container="LINE",
                                  floating_text_block_names=["TEXT",
                                                             "TITLE",
                                                             "LIST",
                                                             "TABLE"],
                                  text_block_names=["TEXT",
                                                    "TITLE",
                                                    "LIST",
                                                    "TABLE"],
                                  text_containers_to_text_block=True)
    pipe_comp.append(text_order)

Page parsing
------------

This is where the page document is created. We can be brief here and
recommend using exactly the same parameters as those used for the
TextOrderService.

.. code:: ipython3

    from deepdoctection.pipe import PageParsingService
    
    page_parsing = PageParsingService(text_container="LINE",
                                      floating_text_block_names=["TEXT",
                                                                 "TITLE",
                                                                 "LIST",
                                                                 "TABLE"],
                                      text_block_names=["TEXT",
                                                        "TITLE",
                                                        "LIST",
                                                        "TABLE"],
                                      text_containers_to_text_block=True)
    pipe_comp.append(page_parsing)

.. code:: ipython3

    from deepdoctection.pipe import DoctectionPipe
    
    pipe = DoctectionPipe(pipe_comp)
    
    path = "/path/to/dir/deepdoctection_images"
    df = pipe.analyze(path=path, output="page")
    
    for dp in df:
        print(dp.get_text())

