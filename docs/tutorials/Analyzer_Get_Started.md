<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>


# Parsing

**deep**doctection is a package that can be used to extract text from complex structured documents. It also allows you
to run vision/text and multi-modal models in an end-to-end pipeline. Inputs can be native PDFs or images. It is very
versatile.

Compared to most other parsers, **deep**doctection offers extensive configurability. 

This section will introduce you to the essential basics of document parsing with **deep**doctection.

!!! warning ""

    TensorFlow will not be supported anymore starting from **deep**doctection version `1.0.0` 
    Python 3.10 in **deep**doctection. But also for lower versions, we recommend not using the TensorFlow anymore.

First, we instantiate the **deep**doctection analyzer. We will be using the default configuration.

```python
import deepdoctection as dd

analyzer = dd.get_dd_analyzer()
```

If the analyzer uses deep learning models (which is generally the case), they are now loaded into memory.

!!! info 

    The analyzer is an example of a pipeline that can be built depending on the problem you want to tackle. This 
    particular pipeline is built from various building blocks. We will come back to this later. 


## Analyze method

Once all models have been loaded, we can process a directory with images (.png, .jpf), a single multi page 
PDF-document or outputs of another **deep**doctection pipeline.


=== "Image directory"

    ```python
    path ="path/to/image_dir"
    
    df = analyzer.analyze(path=path)
    df.reset_state() # (1)
    ```

    1. Necessary to initialize the Dataflow. Throws an error if not called before iteration.

=== "PDF"

    ```python
    path ="path/to/your_doc.pdf"
    
    df = analyzer.analyze(path=path)
    df.reset_state() # (1)
    ```

    1. Necessary to initialize the `Dataflow`. Throws an error if not called before iteration.

=== "JSON"

    ```python
    path ="path/to/dd_output.json"

    page_list = [dd.Page.from_file(path)] # (1) 
    df = dd.DataFromList(lst=page_list) # (2)
    df = analyzer.analyze(dataset_dataflow=df)
    df.reset_state()  # (3)
    ```

    1. Parsing the JSON output into the internal format
    2. Creating a DataFlow (a generator) from the list of pages
    3. Necessary to initialize the Dataflow. Throws an error if not called before iteration.
    

!!! info 

    The method ```analyzer.analyze(path=path)``` does not (yet) return a JSON object, but rather a specialized subclass
    of the ```DataFlow``` class. Essentially, it behaves like a [generator](https://wiki.python.org/moin/Generators). 


Now we can traverse through all the values of the `Dataflow` simply by using a `for`-loop or the `next` function.


```python
doc=iter(df)
page = next(doc)
```


## Page

For each iteration, i.e. for each page document we receive a `Page` object.  Let's also have a look on some top 
level information. 


```python
print(f" height: {page.height}
         width: {page.width}
         file_name: {page.file_name}
         document_id: {page.document_id}
         image_id: {page.image_id}\n")
```

??? info "Output"

     ```
     height: 2339  
     width: 1654 
     file_name: sample_2.png 
     document_id: c1776412-857f-3102-af7c-1869139a278d 
     image_id: c1776412-857f-3102-af7c-1869139a278d
     ```

!!! info 

    `document_id` and `image_id` are the same. The reason is because we only process a single image. The naming 
    convention silently assumes that we deal with a one page document. Once we process multi page PDFs `document_id` 
    and `image_id` differ.

With `get_attribute_names()` we get a list of all attributes. 


```python
page.get_attribute_names()
```

??? info "Output"

    ```
    {'angle',
     'chunks',
     'document_id',
     'document_summary',
     'document_type',
     'figures',
     'file_name',
     'language',
     'layouts',
     'location',
     'pad_bottom',
     'pad_left',
     'pad_right',
     'pad_top',
     'page_number',
     'residual_layouts',
     'size',
     'tables',
     'text',
     'words'}
    ```


Some attributes do not have values because the pipeline component is either deactivated or not part of the pipeline.

## Layout segments

We can visualize detected layout segments like `text`, `title` or `line`.  


```python
page.viz(interactive=True,
         show_tables=True,
         show_layouts= True,
         show_figures=True,
         show_residual_layouts=True) # (1)
```

1. If you set `interactive=True` a viewer will pop up. Use `+` and `-` to zoom out/in. Use `q` to close the page. If you
   set `interactive=False` the image will be returned as a numpy array. You can visualize it e.g. with matplotlib.

![title](../_imgs/analyzer_get_started_02.png)

We can get layout segments from the `layouts` attribute.

```python
for layout in page.layouts:
    print(f"Layout segment: {layout.category_name}, \n 
            score: {layout.score}, \n 
            reading_order: {layout.reading_order}, \n
            bounding_box: {layout.bounding_box}, \n 
            annotation_id: {layout.annotation_id} \n \n 
            text: {layout.text} \n")
```


??? info "Output"

    ```
    Layout segment: text, 
    score: 0.9416185021400452, 
    reading_order: 5, 
    bounding_box: Bounding Box(absolute_coords: True,ulx: 137, uly: 768, lrx: 1518, lry: 825),
    annotation_id: 4dba19ad-12d7-346d-902c-aff8c602d724 

    text: Nach der hervorragenden Entwicklung im Jahr 2017 hatte die globale easealpibande 2018 mit einigen 
          Schwierigkeiten zu kâmpfen. Grûnde waren unguinstige Marktbedin- gungen, stârkere geopolitische Spannungen 
          und die negative Stimmung unter den Anlegern, vor allem am europàischen Retail-Markt. Auch die DWS Gruppe 
          blieb von dieser Entwicklung nicht verschont.
    ```



There are other layout segments that have their own attributes. They depend on one hand side on the type of sections 
that a layout model is able to detect, on the other hand they depend on the analyzer configuration.  

=== "Tables"

    ```python
    page.tables
    ```

=== "Figures"

    ```python
    page.figures
    ```

=== "Residual Layouts"

    ```python
    page.residual_layouts # (1)
    ```

    1. Residual layout segments are currently configured as page headers and page footers. They also do not belong to 
       the narrative text.

### Chunks

Layout segments can also be returned as so-called chunks. Chunks are tuples that, in addition to the text of the layout
segment, contain additional metadata.

```python
page.chunks[0]
```

??? info "Output"

    ```
    ('c1776412-857f-3102-af7c-1869139a278d',  # (1)
     'c1776412-857f-3102-af7c-1869139a278d', # (2)
     0, # (3)
     'e967096b-8c4a-3e3e-99dd-99b02ea0bff4', # (4)
     1, # (5)
     <LayoutType.TEXT>, # (6)
     'Die W-Pools der DWS Gruppe werden einer angemessenen Anpassung der Risiken unterzogen, die die Adjustierung ex 
      ante als auch ex post umfasst...).')
    ```

     1. `document_id`: The ID of the document.
     2. `image_id`: The ID of the image.
     3. `page_number`: The page number of the document.
     4. `annotation_id`: The ID of the layout segment.
     5. `reading_order`: The reading order of the layout segment.
     6. `category_name`: The type of the layout segment, e.g. text, table, figure, etc.

### Tables


```python
table = page.tables[0]
table.get_attribute_names()
```
    
??? info "Output"

    ```
    {'bbox',
     'cells',
     'columns',
     'csv',
     'html',
     'item',
     'layout_link',
     'max_col_span',
     'max_row_span',
     'np_image',
     'number_of_columns',
     'number_of_rows',
     'reading_order',
     'rows',
     'text',
     'words'}
    ```

```python
print(f" number of rows: {table.number_of_rows} \n 
         number of columns: {table.number_of_columns} \n 
         reading order:{table.reading_order}, \n 
         score: {table.score}")
```

??? info "Output"

     ```
     number of rows: 8 
     number of columns: 2 
     reading order: None, 
     score: 0.8250970840454102
     ```

=== "csv"

    ```python
    table.csv # (1)
    ```
    
    1. ```
       pd.DataFrame(table.csv, columns=["Key", "Value"])
       ```

    <pre>
    [['Jahresdurchschnitt der Mitarbeiterzahl ', '139 '],
     ['Gesamtvergutung? ', 'EUR 15.315. .952 '],
     ['Fixe Vergutung ', 'EUR 13.151.856 '],
     ['Variable Vergutung ', 'EUR 2.164.096 '],
     ['davon: Carried Interest ', 'EURO '],
     ['Gesamtvergutung fur Senior Management ', 'EUR 1.468.434 '],
     ['Gesamtvergutung fûr sonstige Risikotrâger ', 'EUR 324.229 '],
     ['Gesamtvergutung fur Mitarbeiter mit Kontrollfunktionen ', 'EUR 554.046 ']]
    </pre>

=== "html"

    ```python
    HTML(table.html)
    ```

    <table><tr><td>Jahresdurchschnitt der Mitarbeiterzahl</td><td>139</td></tr><tr><td>Gesamtvergutung?</td><td>EUR 15.315. .952</td></tr><tr><td>Fixe Vergutung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergutung</td><td>EUR 2.164.096</td></tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergutung fur Senior Management</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergutung fûr sonstige Risikotrâger</td><td>EUR 324.229</td></tr><tr><td>Gesamtvergutung fur Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>

=== "text"

    ```python
    table.text
    ```

    ```
    Jahresdurchschnitt der Mitarbeiterzahl  139  \n Gesamtvergutung?  EUR 15.315. .952  \n 
    Fixe Vergutung  EUR 13.151.856  \n Variable Vergutung  EUR 2.164.096  \n davon: Carried Interest  EURO  \n 
    Gesamtvergutung fur Senior Management  EUR 1.468.434  \n Gesamtvergutung fûr sonstige Risikotrâger  EUR 324.229  \n
    Gesamtvergutung fur Mitarbeiter mit Kontrollfunktionen  EUR 554.046  \n
    ```


The method `kv_header_rows(row_number)` allows returning column headers and cell contents as key-value pairs for entire
rows. 


```python
table.kv_header_rows(2)
```

??? info "Key-Value: Header-Rows"

    ```
    {(1, 'Jahresdurchschnitt der Mitarbeiterzahl'): 'Gesamtvergutung?',
     (2, '139'): 'EUR 15.315. .952'} # (1)
    ```

1. We receive a dictionary with the schema:
   ```{(column_number, column_header(column_number)): cell(row_number, column_number).text}```

### Cells

```python
cell = table.cells[0]
cell.get_attribute_names()
```

??? info "Cell attributes"

    {'bbox',
     'body',
     'column_header',
     'column_number',
     'column_span',
     'header',
     'layout_link',
     'np_image',
     'projected_row_header',
     'reading_order',
     'row_header',
     'row_number',
     'row_span',
     'spanning',
     'text',
     'words'}


```python
print(f"column number: {cell.column_number} \n 
      row_number: {cell.row_number}  \n 
      bounding_box: {cell.bounding_box}\n 
      text: {cell.text} \n 
      annotation_id: {cell.annotation_id}")
      annotation_id}")
```

??? info "Output"

    ```
     column number: 1 
     row_number: 8  
     bounding_box: Bounding Box(absolute_coords: True,ulx: 1, uly: 183, lrx: 640, lry: 210) 
     text: Gesamtvergutung fur Mitarbeiter mit Kontrollfunktionen 
     annotation_id: ad4eba10-411c-357f-941e-8084685e8bf8
    ```

### Words

We can get words for various layout segments, e.g. for `text`, `title` or `cell`.  


```python
word = page.words[0] 
word.get_attribute_names()
```

??? info "Output"

    <pre>
    {'bbox',
     'block',
     'character_type',
     'characters',
     'handwritten',
     'layout_link',
     'np_image',
     'printed',
     'reading_order',
     'tag',
     'text_line',
     'token_class',
     'token_tag'}
    </pre>

As already mentioned, the reading order determines the position of text in a larger text block. There are two levels of
reading orders: 

- Reading order at the level of words in layout sections. 
- Reading order at the level of layout section in a page.

Ordering text is a huge challenge, especially when ordering layout sections. Documents can have a very complex layout
structure and if you use a heuristic ordering approach you need to compromise to some extent. Reading order at the level
of layout sections is basically ordering words in a rectangle. This is easier.

Let's look at some more attributes.

```python

print(f"score: {word.score} \n 
        characters: {word.characters} \n 
        reading_order: {word.reading_order} \n 
        bounding_box: {word.bounding_box}")
```

??? info "Output"

    ```
    score: 0.6492854952812195 
    characters: Kontrollfunktionen 
    reading_order: 5 
    bounding_box: Bounding Box(absolute_coords: False,ulx: 0.25488281, uly: 0.63085938, lrx: 0.33984375, lry: 0.64160156)
    ```


## Text

We can use the `text` property to get the content of the document.

```python
print(page.text)
```

Text from tables is separated from the narrative text. This allows filtering tables from other content.



## Saving and reading


```python
page.save(image_to_json=True, # (1)
          highest_hierarchy_only=True, # (2)
          path="path/to/dir/sample_2.json") # (3)
```

1. The image will be saved as a base64 encoded string in the JSON file.
2. Reduce superfluous information that can be reconstructed later.
3. Either specify the file name or the directory only. The later will save the JSON with its `image_id`.


## Re-loading JSON

```python
page = dd.Page.from_file(file_path="path/to/dir/sample.json")
```

<div class="grid cards" markdown>
- :material-arrow-right-bold: [More about parsing](Analyzer_More_On_Parsing.md)
- :material-arrow-right-bold: [Analyzer Configuration](Analyzer_Configuration.md)  
</div>
