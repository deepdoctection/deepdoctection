Building a custom pipeline
==========================

In this tutorial we will discuss how to create a pipeline with special
components for text extraction.

This extensive tutorial already discusses many of the core components of
this package.

**Suppose we want to perform text extraction from complex structured
documents. The documents essentially consist of text blocks and titles.
There are no tables. We want to use the OCR payment service from AWS
Textract. We also want to have a reading order for the text block, as
the documents contain multiple columns. A JSON file is to be output that
contains all layout and text extractions including the original image.**

Processing steps
----------------

To continue we need to set a processing order. For the construction of
the pipeline, we want to carry out the following steps.

-  Call Textract OCR service
-  Call layout analysis
-  Assign words to layouts blocks via an intersection based rule
-  Determine reading order at the level of layout blocks and further at
   the level within layout blocks.

Pipeline component OCR service
------------------------------

A pipeline component is a building block that carries out certain steps
to accomplish a task.

TextExtractionService is a component that calls a selected OCR service
and transforms the returned results into the internal data model. It is
possible to plug in any OCR Detector into the pipeline component. This
allows a certain flexibility with the composition of pipelines.

Important! Textract is an AWS paid service and you will need an AWS
account to call the client. Alternatively, you can also instantiate a
open sourced OCR service like Tesseract.

.. code:: ipython3

    import os
    from deep_doctection.extern import TextractOcrDetector
    from deep_doctection.pipe import TextExtractionService, DoctectionPipe
    from deep_doctection.utils.systools import get_package_path

.. code:: ipython3

    ocr_detector = TextractOcrDetector()
    
    textract_service = TextExtractionService(ocr_detector,None)
    pipeline_component_list = [textract_service]

Pipeline
--------

We use the DoctectionPipe, which already contains functions for loading
and outputting the extracts.

.. code:: ipython3

    pipeline = DoctectionPipe(pipeline_component_list)

.. code:: ipython3

    path = os.path.join(get_package_path(),"notebooks/pics/samples/sample_3")

.. figure:: ./pics/samples/sample_3/sample_3.png
   :alt: title

   title

We build the pipeline by calling the analyze method and want the results
returned as an image. An image is the core object where everything
grapped from detectors and pipeline components is stored.

Note, that the default output “page” will not return anything, as this
type requires additional layout detections which we will adress later.

.. code:: ipython3

    df = pipeline.analyze(path=path, output="image")
    doc = next(iter(df))


.. parsed-literal::

    [32m[1216 07:53:52 @common.py:558][0m [JoinData] Size check failed for the list of dataflow to be joined!
    processing sample_3.png


It does not make much sense to dig deeper into the image structure. It
is important to know, that it captures all fine graded information from
the OCR result in an ImageAnnotation object. E.g. each single word is
stored with some uuid, bounding box and value (the recorded text).

.. code:: ipython3

    len(doc.annotations), doc.annotations[0]




.. parsed-literal::

    (551,
     ImageAnnotation(active=True, annotation_id='172d1585-9e41-3e79-b7ac-65c81e55340f', category_name='WORD', category_id='1', score=0.9716712951660156, sub_categories={'CHARS': ContainerAnnotation(active=True, annotation_id='3bb03560-00ea-3a21-bab9-c3aa0ec938d3', category_name='CHARS', category_id='None', score=None, sub_categories={}, relationships={}, value='Anleihemärkte'), 'BLOCK': CategoryAnnotation(active=True, annotation_id='b7f36a28-09b4-3954-a002-9064471c365e', category_name='BLOCK', category_id='None', score=None, sub_categories={}, relationships={}), 'LINE': CategoryAnnotation(active=True, annotation_id='f152b47f-61f9-31b3-9904-bfc52a47c003', category_name='LINE', category_id='None', score=None, sub_categories={}, relationships={})}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=137.22318817675114, uly=155.71465119719505, lrx=474.8347396850586, lry=196.48566928505898, height=40.77101808786392, width=337.61155150830746)))



Adding layout elements
----------------------

The current information does not help much so far. An arrangement of
word coordinates from left to right would not result in a meaningful
reading order, as the layout incorporates several columns. One rather
has to determine additional text blocks that frame individual columns. A
built-in layout detector and the associated ImageLayoutService as a
pipeline component are suitable for this.

We use the model config and the weights of the built-in analyzer. If you
haven’t got through the starter tutorial you can download weights using
the ModelDownloadManager:

::

   from ..extern.model import ModelDownloadManager
   ModelDownloadManager.maybe_download_weights("layout/model-2026500.data-00000-of-00001")

.. code:: ipython3

    from deep_doctection.extern import TPFrcnnDetector    
    from deep_doctection.pipe import ImageLayoutService
    from deep_doctection.utils.systools import get_weights_dir_path, get_configs_dir_path

.. code:: ipython3

    config_yaml_path = os.path.join(get_configs_dir_path(),"tp/layout/conf_frcnn_layout.yaml")
    weights_path = os.path.join(get_weights_dir_path(),"layout/model-2026500.data-00000-of-00001")
    categories_layout = {"1": "TEXT", "2": "TITLE", "3": "LIST", "4": "TABLE", "5": "FIGURE"}
    layout_detector = TPFrcnnDetector(config_yaml_path,weights_path,categories_layout)

The ImageLayoutService does need a detector and an additional attribute
that we will not discuss here.

.. code:: ipython3

    layout_service = ImageLayoutService(layout_detector,to_image=True)

Detecting text and layouts are independent tasks, hence the can be
placed in any order within the component.

.. code:: ipython3

    pipeline_component_list.append(layout_service)

Let’s rebuild a new pipeline and start the process again.

.. code:: ipython3

    pipeline = DoctectionPipe(pipeline_component_list)

.. code:: ipython3

    df = pipeline.analyze(path=path, output="image")
    doc = next(iter(df))
    len(doc.annotations), doc.annotations[0]


.. parsed-literal::

    [32m[1216 08:15:13 @common.py:558][0m [JoinData] Size check failed for the list of dataflow to be joined!
    processing sample_3.png




.. parsed-literal::

    (561,
     ImageAnnotation(active=True, annotation_id='172d1585-9e41-3e79-b7ac-65c81e55340f', category_name='WORD', category_id='1', score=0.9716712951660156, sub_categories={'CHARS': ContainerAnnotation(active=True, annotation_id='3bb03560-00ea-3a21-bab9-c3aa0ec938d3', category_name='CHARS', category_id='None', score=None, sub_categories={}, relationships={}, value='Anleihemärkte'), 'BLOCK': CategoryAnnotation(active=True, annotation_id='b7f36a28-09b4-3954-a002-9064471c365e', category_name='BLOCK', category_id='None', score=None, sub_categories={}, relationships={}), 'LINE': CategoryAnnotation(active=True, annotation_id='f152b47f-61f9-31b3-9904-bfc52a47c003', category_name='LINE', category_id='None', score=None, sub_categories={}, relationships={})}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=137.22318817675114, uly=155.71465119719505, lrx=474.8347396850586, lry=196.48566928505898, height=40.77101808786392, width=337.61155150830746)))



Add matching and reading order
------------------------------

Now, that layout and words can be extracted we now have to assign each
detected word to a text box (if this is possible). For that we use the
pre built MatchingService. In our configuration child categories have to
be mapped to parent categories. We use a intersection over are matching
rule with a threshold of 0.9. In other terms, if a word box overlays
with at least 0.9 of its area to a text block it will be assigned to
that box.

.. code:: ipython3

    from deep_doctection.pipe import MatchingService

.. code:: ipython3

    matching_service = MatchingService(parent_categories=["TEXT","TITLE","CELL","LIST","TABLE","FIGURE"],
                            child_categories="WORD",
                            matching_rule="ioa",
                            ioa_threshold=0.9)
    
    pipeline_component_list.append(matching_service )

Reading order service has a straight forward setup.

.. code:: ipython3

    from deep_doctection.pipe import TextOrderService

.. code:: ipython3

    reading_order_service = TextOrderService()

.. code:: ipython3

    pipeline_component_list.append(reading_order_service)

.. code:: ipython3

    pipeline = DoctectionPipe(pipeline_component_list)


We can eventually fire up the custom build analyzer. As we have
everything we need to build the lightweight page object we can change
the output accordingly

.. code:: ipython3

    df = pipeline.analyze(path=path, output="page")
    page = next(iter(df))


.. parsed-literal::

    [32m[1216 08:26:10 @common.py:558][0m [JoinData] Size check failed for the list of dataflow to be joined!
    processing sample_3.png


We can eventually print the OCRed text in reading order with the
get_text method.

.. code:: ipython3

    print(page.get_text())


.. parsed-literal::

    
    Anleihemärkte im Geschäftsjahr bis zum 31.12.2018
    Schwieriges Marktumfeld
    Zinswende nach Rekordtiefs bei Anleiherenditen?
    Die internationalen Anleihe- märkte entwickelten sich im Geschäftsjahr 2018 unter- schiedlich und phasenweise sehr volatil. Dabei machte sich bei den Investoren zunehmend Nervosität breit, was in steigen- den Risikoprämien zum Aus- druck kam. Grund hierfür waren Turbulenzen auf der weltpoli- tischen Bühne, die die politi- schen Risiken erhöhten. Dazu zählten unter anderem populis- tische Strömungen nicht nur in den USA und Europa, auch in den Emerging Markets, wie zuletzt in Brasilien und Mexiko, wo Populisten in die Regie- rungen gewählt wurden. Der eskalierende Handelskonflikt zwischen den USA einerseits sowie Europa und China ande- rerseits tat sein übriges. Zudem ging Italien im Rahmen seiner Haushaltspolitik auf Konfronta- tionskurs zur Europäischen Uni- on (EU). Darüber hinaus verun- sicherte weiterhin der drohende Brexit die Marktteilnehmer, insbesondere dahingehend, ob der mögliche Austritt des Ver- einigten Königreiches aus der EU geordnet oder - ohne ein Übereinkommen - ungeordnet vollzogen wird. Im Gegensatz zu den politischen Unsicher- heiten standen die bislang eher zuversichtlichen, konventionel- len Wirtschaftsindikatoren. So expandierte die Weltwirtschaft kräftig, wenngleich sich deren Wachstum im Laufe der zwei- ten Jahreshälfte 2018 etwas verlangsamte. Die Geldpolitik war historisch gesehen immer noch sehr locker, trotz der welt- weit sehr hohen Verschuldung und der Zinserhöhungen der US-Notenbank.
    Im Berichtszeitraum kam es an den Anleihemärkten - wenn auch uneinheitlich und unter- schiedlich stark ausgeprägt - unter Schwankungen zu stei- genden Renditen auf teilweise immer noch sehr niedrigem Niveau, begleitet von nachge- benden Kursen. Dabei konnten sich die Zinsen vor allem in den USA weiter von ihren histori- schen Tiefs lösen. Gleichzeitig wurde die Zentralbankdivergenz zwischen den USA und dem Euroraum immer deutlicher. An- gesichts des Wirtschaftsbooms in den USA hob die US-Noten- bank Fed im Berichtszeitraum den Leitzins in vier Schritten weiter um einen Prozentpunkt auf einen Korridor von 2,25% - 2,50% p. a. an. Die Europäische Zentralbank (EZB) hingegen hielt an ihrer Nullzinspolitik fest und die Bank of Japan beließ ihren Leitzins bei -0,10% p. a. Die Fed begründete ihre Zinser- höhungen mit der Wachstums- beschleunigung und der Voll- beschäftigung am Arbeitsmarkt in den USA. Zinserhöhungen ermöglichten der US-Notenbank einer Überhitzung der US-Wirt- schaft vorzubeugen, die durch die prozyklische expansive
    Entwicklung der Leitzinsen in den USA und im Euroraum % p.a.
    
    Fiskalpolitik des US-Präsidenten Donald Trump in Form von Steuererleichterungen und einer Erhöhung der Staatsausgaben noch befeuert wurde. Vor die- sem Hintergrund verzeichneten die US-Bondmärkte einen spür- baren Renditeanstieg, der mit merklichen Kursermäßigungen einherging. Per saldo stiegen die Renditen zehnjähriger US- Staatsanleihen auf Jahressicht von 2,4% p.a. auf 3,1% p. a.
    Diese Entwicklung in den USA hatte auf den Euroraum jedoch nur phasenweise und partiell, insgesamt aber kaum einen zinstreibenden Effekt auf Staats- anleihen aus den europäischen Kernmärkten wie beispielsweise Deutschland und Frankreich. So gaben zehnjährige deutsche Bundesanleihen im Jahresver- lauf 2018 unter Schwankungen per saldo sogar von 0,42% p.a. auf 0,25% p. a. nach. Vielmehr standen die Anleihemärkte der Euroländer - insbeson- dere ab dem zweiten Quartal 2018 - unter dem Einfluss der politischen und wirtschaftlichen Entwicklung in der Eurozone, vor allem in den Ländern mit hoher Verschuldung und nied- rigem Wirtschaftswachstum. In den Monaten Mai und Juni

