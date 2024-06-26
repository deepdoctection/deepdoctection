# Changing the layout parsing structure

The parsing of the layout structure depends on the customizing of the pipeline components, but of course also on the 
results of the layout object detectors and OCR results. 

This tutorial builds on the [Pipeline](pipelines_notebook.md) tutorial, but describes the interaction of the pipeline 
components in more detail. 

This is a niche topic that you will need, if you train a layout detection model or sub layout detection model (cell in
tables, line items in invoices) or if you want to change the text extraction structure (e.g. ignoring specified 
layout sections in narrative text).

Let assume we've defined a pipeline for document parsing and text extraction. The best way to analyze the results per 
page is to use the `Page` class that works like a view to the raw objects collected by the pipeline.

```python

    df = pipeline.analyze(path="path/to/image_dir")
    df.reset_state()

    for dp in df:
       for layout in dp.layouts:
          print(layout.category_name)  # getting coarse layout blocks.
```


Layout structures are usually determined by a layout model. The model trained on *Publaynet* defines five different
layout structures: titles, text, list, figure, and table. However, you do not want to have table objects in 
narrative text. The default setting of the analyzer silently assumes this requirement. In the code above, tables are
not part of `dp.layouts` and if you call `dp.text` the table content will not be displayed. 

But what, if you want to have the table content as being part of narrative text? 

Well, you must first ensure that words (or more general text containers) will be assigned to table cells via 
`MatchingService`. If you do not really mind about the table structure but only want to have a simple ordering of
words within the table bounding box you can also assign text containers directly to tables similar to the other
layout blocks.

```python3

    import deepdoctection as dd

    # setting up layout detector and layout service
    categories_layout = {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"}

    layout_config_path = dd.ModelCatalog.get_full_path_configs("dd/tp/conf_frcnn_layout.yaml")
    layout_weights_path = "/path/to/dir/model-820000.data-00000-of-00001" # some model trained on custom dataset
    d_layout = dd.TPFrcnnDetector(layout_config_path, layout_weights_path, categories_layout)
    layout = dd.ImageLayoutService(d_layout, to_image=True)

    # setting up OCR detector and ocr service
    ocr = dd.TextractOcrDetector()
    text = dd.TextExtractionService(ocr)

    # setting up first two pipeline components
    pipe_comp =[layout,text]
```

For now, words and layout blocks are not related to each other. We will assign words to layout blocks using the 
`MatchingService`


## Matching Service

**deep**doctection provides a service to generate a hierarchy based
on parent categories (layout blocks), child categories (words) and a matching rule according to which
a parental/child relationship will be established provided once a given threshold has been exceeded.

Note, that we haven't added `dd.LayoutType.figure` to the `parent_categories`. This will ignore all figure-type
layout section and keep all words within as orphan.

```python3

match = dd.MatchingService(
    parent_categories=[dd.LayoutType.TEXT,
                       dd.LayoutType.TITLE,
                       dd.LayoutType.LIST,
                       dd.LayoutType.TABLE],
    child_categories=dd.LayoutType.WORD,
    matching_rule="ioa",
    threshold=0.9
)
pipe_comp.append(match)
``` 

## Text order service

Next:

- You will have to infer a reading order on one hand side on a high level, i.e. layout blocks and on the other hand 
  by ordering words within a layout block.
- You need to think of what text blocks are part of the narrative text. As discussed before we aim to add text, title,
  list and tables to a narrative text.
- What happens to orphan words? Should they be completely ignored? Where this might sense in some cases you do want to
  have some control and collect and arrange words not assigned to layout sections.  

The following diagram sets up the terminology we are going to use.

![](./_imgs/dd_text_order.png)

We have

- `text containers`: Objects that contain text on word/text line level and come with
                     surrounding bounding boxes. In general, they have `category_name`s like `LayoutTypes.word`, 
                    `LayoutTypes.line`.
- `text_block_categories`: High level layout structures. All text containers assigned to one text block will be sorted 
                   to give a contiguous block of text. Do not confuse with `floating_text_block_categories`. Adding
                   a layout section to this argument will only guarantee that text containers assigned to this section
                   will get a reading order relative to the text block. That is, if `layout` is an element of 
                   `dp.layouts` then `layout.text` will give you ordered text lying within its layout section.
- `floating_text_block_categories`: Sub set of `text_block_categories`. This will order the text block categories 
                   themselves. The ordering is following heuristics and will not work for all layout structures. 
                   Adding an ordering algorithm based on data is something that we would like to integrate in the future.
- `include_residual_text_container`: Flag to incorporate orphan words into narrative text. This will group text 
                   containers into (synthetic) lines/sub lines (if necessary) and consider these synthetic line sections 
                   as floating layout sections. 

  
Determining the reading order of a text is done in two stages:

1. Ordering text within layout structures. Currently, text is assumed to be read
as in the Latin alphabet (i.e. line wise from top to bottom).

2. Ordering of floating layout structures. As already mentioned, the algorithm uses heuristics that might fail in various
use-cases: 

Layout sections are divided into columns. Columns are not assumed to extend from the top of the page to the bottom, but
may be interrupted (for example, by a title, a table, a figure). 
Then columns are divided into context components. As explained in the diagram, this is to merge columns so that
horizontally adjacent columns are read block by block from left to right. 
Subsequently, a numbering is derived from this arrangement. 

The algorithm provides various parameters (`starting_point_tolerance`, `broken_line_tolerance`, ...) that decide
whether layout sections belong to columns or columns belong to contiguous components. For more information on these
parameters, please refer to the API documentation.

![pipelines](./_imgs/dd_connected_blocks.png)

Coming back to our problem, we want to order all text within our layout sections that have been declared as 
`parent_categories` in the `MatchingService`. And we also want to consider all these layout sections to be in narrative
text. Finally, we want to have orphan words to be in narrative text. The configuration therefore looks like this:

```python    
    text_order = dd.TextOrderService(text_container="line",
                                     text_block_categories= [dd.LayoutType.TEXT,
                                                             dd.LayoutType.TITLE,
                                                             dd.LayoutType.LIST,
                                                             dd.LayoutType.TABLE],
                                     floating_text_block_categories=[dd.LayoutType.TEXT,
                                                             dd.LayoutType.TITLE,
                                                             dd.LayoutType.LIST,
                                                             dd.LayoutType.TABLE],
                                     include_residual_text_container=True)
    pipe_comp.append(text_order)
```

We can be done now, for example, if we want to store the result of the pipeline in a JSON. 

On the other hand, if we want to operate with the text or tables themselves we have to perform a final transformation, 
namely the conversion of the `Image` structure into the `Page` structure. We refer to the 
[data structure tutorial](data_structure_notebook.md) for more information about these objects. 


## Page parsing

The `PageParsingService` contains parameters that can also be found in the `TextOrderService`. In contrast to the 
latter, the purpose of the conformation is to ensure that the layout sections are actually taken into account in the 
formation of the narrative text string. It is therefore advisable to use the same configuration as in the 
`TextOrderService`. 

However, one can deviate from this a little bit. For example, in `floating_text_block_categories` you can pass only a 
subset of the set defined in `TextOrderService`. In this case some layout sections will be skipped. You can also set 
`include_residual_text_container=False`. 

However, it is not possible to include layout sections in the selection that were not defined in the `TextOrderService`, 
because these would not contain a `reading_order` attribute and the processing would raise an error.


```python
    
    page_parsing = dd.PageParsingService(text_container="word",
                                         floating_text_block_categories=[dd.LayoutType.TEXT,
                                                                         dd.LayoutType.TITLE,
                                                                         dd.LayoutType.LIST,
                                                                         dd.LayoutType.TABLE],
                                         include_residual_text_container=True)
    
    pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp,
                             page_parsing_service=page_parsing)
    
    path = "/path/to/dir/deepdoctection_images"
    df = pipe.analyze(path=path)
    
    for dp in df:
        print(dp.text)
``` 
