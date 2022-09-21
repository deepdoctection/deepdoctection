Changing the layout parsing structure
=====================================

Layout structures are usually determined by the model. As a consequence,
the pipeline must ensure optimal processing depending on what layout structures
were made available.

When analyzing complex structured documents you will be confronted with
the question how to organize the output from layout structures and its
enclosed text.

Let's say you have a model that has been trained on Publaynet and returns
the same layout structure. You also have an OCR system that returns text on
text line level and you somehow need to relate the results obtained.

.. code:: ipython3

    import os
    import deepdoctection as dd

.. code:: ipython3

    # setting up layout detector and layout service
    categories_layout = {"1": "TEXT", "2": "TITLE", "3": "LIST", "4": "TABLE", "5": "FIGURE"}

    layout_config_path = os.path.join(get_configs_dir_path(), "dd/tp/conf_frcnn_layout.yaml")

    layout_weights_path = "/path/to/dir/model-820000.data-00000-of-00001"
    d_layout = dd.TPFrcnnDetector(layout_config_path, layout_weights_path, categories_layout)
    layout = dd.ImageLayoutService(d_layout, to_image=True)

    # setting up OCR detector and ocr service
    ocr = dd.TextractOcrDetector(text_lines=True)
    text = dd.TextExtractionService(ocr)

    # setting up first two pipeline components
    pipe_comp =[layout,text]

Next:

- You will have to assign word/text line to Publaynet layout structures.
- You will have to take care of words/text line that cannot be assigned (because not all layout
  structures might have been discovered and therefore not every word/text line has been covered by layout structures).
- You will have to infer a reading order on one hand side on the level within one
  layout structure and on the other hand by ordering the higher level text structures.
- When presenting text you need to think of what text blocks have to be considered as entity.
  Suppose you have text blocks and tables. Will it make sense to include table content
  into the contiguous text ?

The following diagram sets up the terminology we are going to use.

.. figure:: ./pics/dd_text_order.png
   :alt: title

We have

- `text containers`: Objects that contain text on word/text line level and come with
                     surrounding bounding boxes.
- `text_block`: Higher level text blocks that form a entity like a section or title. All text
                containers assigned to one text block will be sorted to give a contiguous block
                of text.
- `floating text blocks`: Sub set of text blocks. After ordering words within a text block
                          floating text blocks themselves will be ordered as well to give contiguous
                          text of a page. Text blocks not belonging to floating text blocks will
                          not be considered.


Matching Service
----------------

We now have to assign text lines to text blocks by looking how do text lines overlap with
the underlying coarse structure. **deep**doctection provides a service to generate a hierarchy based
on parent categories (coarse layout sections), child categories and a matching rule according to which
a parental/child relationship will be established provided a threshold has been exceeded.

.. code:: ipython3
    
    match = dd.MatchingService(
            parent_categories=["TEXT",
                               "TITLE",
                               "LIST",
                               "TABLE"],
            child_categories="LINE",
            matching_rule="ioa",
            threshold=0.9
            )
    pipe_comp.append(match)

Text order service
------------------

Determining the reading order of a text is done in two stages:

1.) Ordering text within text blocks. Currently, text is assumed to be read
as in the Latin alphabet (i.e from line wise from top to bottom).

2.) Ordering of floating text blocks.

It may happen that text containers are not assigned to text blocks. If
you do not want to loose any text, you can view unassigned text
containers as a floating text block and incorporate this block in the sorting
process (`text_containers_to_text_block=True`).

.. code:: ipython3
    
    text_order = dd.TextOrderService(text_container="LINE",
                                  floating_text_block_names=["TEXT",
                                                             "TITLE",
                                                             "LIST"],
                                  text_block_names=["TEXT",
                                                    "TITLE",
                                                    "LIST",
                                                    "TABLE"],
                                  text_containers_to_text_block=True)
    pipe_comp.append(text_order)


Page parsing
------------

Now, that layout structures and text lines have been related and ordered it is now
time to create an output structure (:class:`Page`) you can work with. When parsing
into the target structure you have to take into account what intrinsic structure you
have generated and therefore have you need to apply the previous setting when defining
text containers, text blocks etc.


.. code:: ipython3
    
    page_parsing = dd.PageParsingService(text_container="LINE",
                                         floating_text_block_names=["TEXT",
                                                                    "TITLE",
                                                                    "LIST"],
                                         text_block_names=["TEXT",
                                                           "TITLE",
                                                           "LIST",
                                                           "TABLE"],
                                         text_containers_to_text_block=True)
    pipe_comp.append(page_parsing)

.. code:: ipython3

    
    pipe = dd.DoctectionPipe(pipe_comp)
    
    path = "/path/to/dir/deepdoctection_images"
    df = pipe.analyze(path=path, output="page")
    
    for dp in df:
        print(dp.get_text())

