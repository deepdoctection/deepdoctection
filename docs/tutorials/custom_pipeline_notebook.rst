Building a custom pipeline
==========================

In this tutorial we will discuss how to create a pipeline with special
components for text extraction.

This extensive tutorial already discusses many of the core components of
this package.

We formulate the requirement as follows:

**Suppose we want to perform text extraction from complex structured
documents. The documents essentially consist of text blocks and titles.
There are no tables. We want to use the OCR payment service from AWS
Textract. We also want to have a reading order for text blocks, as the
documents contain multiple columns. The analysis results are to be
returned in a JSON structure that contains all layout information as
well as the full text and the original image.**

Processing steps
----------------

To continue we need to set a processing order. For the construction of
the pipeline, we want to carry out the following steps.

-  Call Textract OCR service
-  Call layout analysis
-  Assign words to layouts blocks via an intersection based rule
-  Determine reading order at the level of layout blocks and at the word
   level within one layout block.

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
open sourced OCR service like Tesseract. Just disable
``TextractOcrDetector()`` and uncomment the following two lines!
Moreover, to allow the TextractOcrDetector to work you will need the
package extension ‘source-all-tf’.

.. code:: ipython3

    import os
    from deepdoctection.extern import TextractOcrDetector, TesseractOcrDetector
    from deepdoctection.pipe import TextExtractionService, DoctectionPipe
    from deepdoctection.utils.systools import get_package_path, get_configs_dir_path

.. code:: ipython3

    ocr_detector = TextractOcrDetector()
    
    #tess_ocr_config_path = os.path.join(get_configs_dir_path(),"dd/conf_tesseract.yaml")
    #ocr_detector = TesseractOcrDetector(tess_ocr_config_path, config_overwrite=["LANGUAGES=deu"])
    
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
    df.reset_state()
    doc = next(iter(df))

It does not make much sense to dig deeper into the image structure. It
is important to know, that it captures all fine graded information from
the OCR result in an ImageAnnotation object. E.g. each single word is
stored with some uuid, bounding box and value (the recorded text).

.. code:: ipython3

    len(doc.annotations), doc.annotations[0]




.. parsed-literal::

    (545,
     ImageAnnotation(active=True, _annotation_id='3be39a8e-880b-3a18-b0d7-80e05beb68f4', category_name=<LayoutType.word>, _category_name=<LayoutType.word>, category_id='1', score=0.9221703338623047, sub_categories={<WordType.characters>: ContainerAnnotation(active=True, _annotation_id='e68e2072-ff7c-3152-ab6b-d8fc6156dc02', category_name=<WordType.characters>, _category_name=<WordType.characters>, category_id='None', score=0.9221703338623047, sub_categories={}, relationships={}, value='Anleihemärkte')}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=134.921634465456, uly=157.1062769368291, lrx=472.318872153759, lry=195.05085966736078, height=37.94458273053169, width=337.397237688303)))



Adding layout elements
----------------------

The current information does not help much so far. An arrangement of
word coordinates from left to right would not result in a meaningful
reading order, as the layout incorporates several columns. One rather
has to determine additional text blocks that frame individual columns. A
built-in layout detector and the associated ImageLayoutService as a
pipeline component are suitable for this.

At this point it starts to depend on whether the DL framework Tensorflow
or PyTorch will be used. We assume that Tensorflow is installed, hence
we need to import the Tensorflow related Detector TPFrcnnDetector. Use
D2FrcnnDetector for PyTorch.

We use the model config and the weights of the built-in analyzer. If you
haven’t got through the starter tutorial you can download weights using
the ModelDownloadManager.

::

   from deepdoctection.extern.model import ModelDownloadManager
   ModelDownloadManager.maybe_download_weights_and_configs("layout/model-800000_inf_only.data-00000-of-00001")

Download ``"layout/d2_model-800000-layout.pkl"`` instead, in case you
use PyTorch.

.. code:: ipython3

    from deepdoctection.extern import TPFrcnnDetector, ModelCatalog    
    from deepdoctection.pipe import ImageLayoutService
    from deepdoctection.utils.systools import get_weights_dir_path, get_configs_dir_path

When the model is downloaded from the hub, both the weights and the
config file are loaded into the cache. The paths to both files are
required in order to instantiate the detector. You can use the
ModelCatalog to build the path. Moreover, the ModelCatalog provides a
brief model card of all registered models.

It is also necessary to pass a dict with the category-id/category names
pairs. This mapping is standard and results from the dataset Publaynet
on which this model was trained.

.. code:: ipython3

    profile = ModelCatalog.get_profile("layout/model-800000_inf_only.data-00000-of-00001")
    profile.as_dict()




.. parsed-literal::

    {'name': 'layout/model-800000_inf_only.data-00000-of-00001',
     'description': 'Tensorpack layout model for inference purposes trained on Publaynet',
     'size': [274552244, 7907],
     'tp_model': True,
     'config': 'dd/tp/conf_frcnn_layout.yaml',
     'hf_repo_id': 'deepdoctection/tp_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only',
     'hf_model_name': 'model-800000_inf_only',
     'hf_config_file': ['conf_frcnn_layout.yaml'],
     'urls': None,
     'categories': {'1': <LayoutType.text>,
      '2': <LayoutType.title>,
      '3': <LayoutType.list>,
      '4': <LayoutType.table>,
      '5': <LayoutType.figure>}}



.. code:: ipython3

    config_yaml_path = ModelCatalog.get_full_path_configs("layout/model-800000_inf_only.data-00000-of-00001")
    weights_path = ModelCatalog.get_full_path_weights("layout/model-800000_inf_only.data-00000-of-00001") 
    categories_layout = profile.categories
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
    df.reset_state()
    doc = next(iter(df))
    len(doc.annotations), doc.annotations[0]

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

    from deepdoctection.pipe import MatchingService

.. code:: ipython3

    matching_service = MatchingService(parent_categories=["TEXT","TITLE","CELL","LIST","TABLE","FIGURE"],
                            child_categories="WORD",
                            matching_rule="ioa",
                            threshold=0.9)
    
    pipeline_component_list.append(matching_service )

Reading order service has a straight forward setup.

.. code:: ipython3

    from deepdoctection.pipe import TextOrderService

.. code:: ipython3

    reading_order_service = TextOrderService(text_container="WORD",floating_text_block_names=["TEXT","TITLE","LIST"],
                                             text_block_names=["TEXT","TITLE","LIST","TABLE","FIGURE"])

.. code:: ipython3

    pipeline_component_list.append(reading_order_service)

.. code:: ipython3

    pipeline = DoctectionPipe(pipeline_component_list)


We can eventually fire up the custom build analyzer. As we have
everything we need to build the lightweight page object we can change
the output accordingly

.. code:: ipython3

    df = pipeline.analyze(path=path, output="page")
    df.reset_state()
    page = next(iter(df))

We can eventually print the OCRed text in reading order with the
get_text method.

.. code:: ipython3

    print(page.get_text())


.. parsed-literal::

    
    Anleihemärkte im Geschäftsjahr bis zum 31.12.2018
    Die internationalen Anleihe- märkte entwickelten sich im Geschäftsjahr 2018 unter- schiedlich und phasenweise sehr volatil. Dabei machte sich bei den Investoren zunehmend Nervosität breit, was in steigen- den Risikoprämien zum Aus- druck kam. Grund hierfür waren Turbulenzen auf der weltpoli- tischen Bühne, die die politi- schen Risiken erhöhten. Dazu zählten unter anderem populis- tische Strömungen nicht nur in den USA und Europa, auch in den Emerging Markets, wie zuletzt in Brasilien und Mexiko, wo Populisten in die Regie- rungen gewählt wurden. Der eskalierende Handelskonflikt zwischen den USA einerseits sowie Europa und China ande- rerseits tat sein übriges. Zudem ging Italien im Rahmen seiner Haushaltspolitik auf Konfronta- tionskurs zur Europäischen Uni- on (EU). Darüber hinaus verun- sicherte weiterhin der drohende Brexit die Marktteilnehmer, insbesondere dahingehend, ob der mögliche Austritt des Ver- einigten Königreiches aus der EU geordnet oder ohne ein Übereinkommen ungeordnet vollzogen wird. Im Gegensatz den politischen Unsicher- heiten standen die bislang eher zuversichtlichen, konventionel- len Wirtschaftsindikatoren So expandierte die Weltwirtschaft kräftig, wenngleich sich deren Wachstum im Laufe der zwei- ten Jahreshälfte 2018 etwas verlangsamte. Die Geldpolitik war historisch gesehen immer noch sehr locker, trotz der welt- weit sehr hohen Verschuldung und der Zinserhöhungen der US-Notenbank.
    Entwicklung der Leitzinsen in den USA und im Euroraum % p. a.
    Zinswende nach Rekordtiefs bei Anleiherenditen? Im Berichtszeitraum kam es an den Anleihemärkten - wenn auch uneinheitlich und unter- schiedlich stark ausgeprägt unter Schwankungen zu stei- genden Renditen auf teilweise immer noch sehr niedrigem Niveau, begleitet von nachge- benden Kursen. Dabei konnten sich die Zinsen vor allem in den USA weiter von ihren histori- schen Tiefs lösen. Gleichzeitig wurde die Zentralbankdivergenz zwischen den USA und dem Euroraum immer deutlicher. An- gesichts des Wirtschaftsbooms in den USA hob die US-Noten- bank Fed im Berichtszeitraum den Leitzins in vier Schritten weiter um einen Prozentpunkt auf einen Korridor von 2,25% 2,50% p.a. an. Die Europäische Zentralbank (EZB) hingegen hielt an ihrer Nullzinspolitik fest und die Bank of Japan beließ ihren Leitzins bei -0,10% p.a. Die Fed begründete ihre Zinser- höhungen mit der Wachstums- beschleunigung und der Voll- beschäftigung am Arbeitsmarkt in den USA. Zinserhöhungen ermöglichten der US-Notenbank einer Überhitzung der US-Wirt- schaft vorzubeugen, die durch die prozyklische expansive
    Fiskalpolitik des US-Präsidenten Donald Trump in Form von Steuererleichterungen und einer Erhöhung der Staatsausgaben noch befeuert wurde. Vor die- sem Hintergrund verzeichneten die US-Bondmärkte einen spür- baren Renditeanstieg, der mit merklichen Kursermäßigungen einherging. Per saldo stiegen die Renditen zehnjähriger US- Staatsanleihen auf Jahressicht von 2,4% p.a. auf 3,1% p.a.
    Diese Entwicklung in den USA hatte auf den Euroraum jedoch nur phasenweise und partiell, insgesamt aber kaum einen zinstreibenden Effekt auf Staats- anleihen aus den europäischen Kernmärkten wie beispielsweise Deutschland und Frankreich. So gaben zehnjährige deutsche Bundesanleihen im Jahresver- lauf 2018 unter Schwankungen per saldo sogar von 0,42% p.a. auf 0,25% p. a. nach. Vielmehr standen die Anleihemärkte der Euroländer insbeson- dere ab dem zweiten Quartal 2018 unter dem Einfluss der politischen und wirtschaftlichen Entwicklung in der Eurozone, vor allem in den Ländern mit hoher Verschuldung und nied- rigem Wirtschaftswachstum In den Monaten Mai und Juni


How to continue
===============

In the next step we recommend the tutorial **Datasets_and_Eval**. Here,
the data model of the package is explained in more detail. It also
explains how to evaluate the precision of models using labeled data.
