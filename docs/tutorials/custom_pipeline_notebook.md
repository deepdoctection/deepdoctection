# Building a custom pipeline

The **deep**doctection analyzer is an example of a document layout analysis pipeline. In this tutorial we'll show you 
the concepts so that you can build a pipeline yourself.


```python
from pathlib import Path
import deepdoctection as dd
```
&nbsp;

The idea is not that difficult: There are models that fulfill a given task, there are pipeline components or pipeline backbones that invoke models and take care of pre- and post-processing results. There are also pipeline backbones that do not invoke models but only consolidate results. 

And there is the pipeline that puts everything together.

## Catalog and registries

You can get the essential information for pre-trained model from the `ModelCatalog`. 


```python
dd.print_model_infos(add_description=False,add_config=False,add_categories=False)
```


    ╒═════════════════════════════════════════════════════╕
    │ name                                                │
    ╞═════════════════════════════════════════════════════╡
    │ layout/model-800000_inf_only.data-00000-of-00001    │
    ├─────────────────────────────────────────────────────┤
    │ cell/model-1800000_inf_only.data-00000-of-00001     │
    ├─────────────────────────────────────────────────────┤
    │ item/model-1620000_inf_only.data-00000-of-00001     │
    ├─────────────────────────────────────────────────────┤
    │ item/model-1620000.data-00000-of-00001              │
    ├─────────────────────────────────────────────────────┤
    │ layout/model-800000.data-00000-of-00001             │
    ├─────────────────────────────────────────────────────┤
    │ cell/model-1800000.data-00000-of-00001              │
    ├─────────────────────────────────────────────────────┤
    │ layout/d2_model-800000-layout.pkl                   │
    ├─────────────────────────────────────────────────────┤
    │ layout/d2_model_0829999_layout_inf_only.pt          │
    ├─────────────────────────────────────────────────────┤
    │ layout/d2_model_0829999_layout.pth                  │
    ├─────────────────────────────────────────────────────┤
    │ cell/d2_model-1800000-cell.pkl                      │
    ├─────────────────────────────────────────────────────┤
    │ cell/d2_model_1849999_cell_inf_only.pt              │
    ├─────────────────────────────────────────────────────┤
    │ cell/d2_model_1849999_cell.pth                      │
    ├─────────────────────────────────────────────────────┤
    │ item/d2_model-1620000-item.pkl                      │
    ├─────────────────────────────────────────────────────┤
    │ item/d2_model_1639999_item.pth                      │
    ├─────────────────────────────────────────────────────┤
    │ item/d2_model_1639999_item_inf_only.pt              │
    ├─────────────────────────────────────────────────────┤
    │ microsoft/layoutlm-base-uncased/pytorch_model.bin   │
    ├─────────────────────────────────────────────────────┤
    │ microsoft/layoutlm-large-uncased/pytorch_model.bin  │
    ├─────────────────────────────────────────────────────┤
    │ microsoft/layoutlmv2-base-uncased/pytorch_model.bin │
    ├─────────────────────────────────────────────────────┤
    │ microsoft/layoutxlm-base/pytorch_model.bin          │
    ├─────────────────────────────────────────────────────┤
    │ microsoft/layoutlmv3-base/pytorch_model.bin         │
    ├─────────────────────────────────────────────────────┤
    │ fasttext/lid.176.bin                                │
    ╘═════════════════════════════════════════════════════╛


Let's select fasttext language detector. We need the categories that the model predicts and the model wrapper. 
`fasttext/lid.176.bin` is just an artifact. 


```python
categories=dd.ModelCatalog.get_profile("fasttext/lid.176.bin").categories
```


```python
dd.ModelCatalog.get_profile("fasttext/lid.176.bin").model_wrapper
```




    'FasttextLangDetector'

&nbsp;

We can download `lid.176.bin` with help of the `ModelDownloadManager`.


```python
path_weights=dd.ModelDownloadManager.maybe_download_weights_and_configs("fasttext/lid.176.bin")
```


## Model wrapper

We know from the `ModelCatalog` which wrapper we must use for the fasttext model.


```python
fast_text = dd.FasttextLangDetector(path_weights, categories)
```

    Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.


We are not done yet, because we still need to choose how to extract text. Let's simply stick to Tesseract and use the default english setting.


```python
tess_ocr_config_path = dd.get_configs_dir_path() / "dd/conf_tesseract.yaml"  # This file will be in you .cache if you ran the analyzer before. 
# Otherwise make sure to copy the file from 'configs/conf_tesseract.yaml'

tesseract_ocr = dd.TesseractOcrDetector(tess_ocr_config_path.as_posix())
```

## Pipeline backbone

Similar to models we have a pipeline component registry. Having this starting point we can select the right backbone. Check the API documentation to see what the components are used for.


```python
dd.pipeline_component_registry.get_all()
```




    {'SubImageLayoutService': deepdoctection.pipe.cell.SubImageLayoutService,
     'ImageCroppingService': deepdoctection.pipe.common.ImageCroppingService,
     'MatchingService': deepdoctection.pipe.common.MatchingService,
     'PageParsingService': deepdoctection.pipe.common.PageParsingService,
     'LanguageDetectionService': deepdoctection.pipe.language.LanguageDetectionService,
     'ImageLayoutService': deepdoctection.pipe.layout.ImageLayoutService,
     'LMTokenClassifierService': deepdoctection.pipe.lm.LMTokenClassifierService,
     'LMSequenceClassifierService': deepdoctection.pipe.lm.LMSequenceClassifierService,
     'TableSegmentationRefinementService': deepdoctection.pipe.refine.TableSegmentationRefinementService,
     'TableSegmentationService': deepdoctection.pipe.segment.TableSegmentationService,
     'TextExtractionService': deepdoctection.pipe.text.TextExtractionService,
     'TextOrderService': deepdoctection.pipe.text.TextOrderService,
     'SimpleTransformService': deepdoctection.pipe.transform.SimpleTransformService}



## Fasttext language detector


```python
lang_detect_comp = dd.LanguageDetectionService(fast_text,text_detector=tesseract_ocr)
```

We can now build our very simple pipeline.


```python
pipe_comp_list = [lang_detect_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)
```


```python
image_path = Path.cwd() / "pics/samples/sample_3" 
```

![title](./_imgs/sample_3.png)

When running the pipeline, we get the language in which the document was written. 


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.language
```




    <Languages.german>



When getting the text, the response is somewhat disappointing.


```python
dp.text
```


    ''



The reason for that is that `LanguageDetectionService` is not responsible for extracting text. It has an OCR model, 
but the output is only used as input feed to the language detector. The text however is not persisted. If we had 
added a `TextExtractionService` before `LanguageDetectionService` we could have omitted the OCR model in the 
`LanguageDetectionService`. 

## Tesseract OCR detector


```python
tesseract_ocr = dd.TesseractOcrDetector(tess_ocr_config_path.as_posix(),["LANGUAGES=deu"])
```


```python
tesseract_ocr.config
```


    {'LANGUAGES': 'deu', 'LINES': False, 'psm': 11}




```python
# setting run_time_ocr_language_selection=True will dynamically select the OCR model for text extraction based on 
# the predicted languages. This helps to get much improved OCR results, if you have documents with various languages.

text_comp = dd.TextExtractionService(tesseract_ocr, run_time_ocr_language_selection=True)
pipe_comp_list.append(text_comp)
```


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.text
```



    ''



This is something unexpected. Why don't we generate any text? We can clearly see that the `TextExtractionService` 
did its job.


```python
word_sample = dp.words[0]
len(dp.words), word_sample.characters, word_sample.bbox, word_sample.reading_order 
```


    (553, 'Anleihemärkte', [137.0, 158.0, 472.0, 195.0], None)



## Text ordering

The reason is, that we do not have inferred a reading order. If there is no reading order, there is no contiguous text. 
We treat text extraction as a character recognition problem only. If we want a reading order of predicted words, we 
need to do it ourself. So let's add the `TextOrderService`.


```python
order_comp = dd.TextOrderService(text_container=dd.LayoutType.word)
pipe_comp_list.append(order_comp)
```
&nbsp;

At least, we got some text. The beginning sounds good. But once the text comes to the region where the second and 
third column also have text lines, the order service does not distinguish between columns. So we must identify columns. 
For that we use the layout analyzer.


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.text
```
    |1/?[00:00<00:00,1212.23it/s]
    [1219 17:03.25 @doctectionpipe.py:124]Processing sample_3.png
    [32m[1219 17:03.28 @context.py:131]LanguageDetectionService total: 2.674 sec.
    [32m[1219 17:03.30 @context.py:131][37mTextExtractionService total: 2.8881 sec.
    [32m[1219 17:03.30 @context.py:131][37mTextOrderService total: 0.0146 sec.



    ' Anleihemärkte im Geschäftsjahr bis zum 31.12.2018 Schwieriges Marktumfeld Entwicklung der Leitzinsen in den USA und im Euroraum Die internationalen Anleihe- %p.a. märkte entwickelten sich im Geschäftsjahr 2018 unter- schiedlich und phasenweise sehr volatil. Dabei machte sich bei den Investoren zunehmend Nervosität breit, was in steigen- -1 u u u u u u u u u u u den Risikoprämien zum Aus- 12/08 12/09 12/10 12/11 12/12 12/13 12/14 12/15 12/16 12/17 12/18 druck kam. Grund hierfür waren BE Fed-Leitzins Quelle: Thomson Financial Datastream Turbulenzen auf der weltpoli- BE E28-Leitzins Stand: 31.12.2018 tischen Bühne, die die politi- schen Risiken erhöhten. Dazu zählten unter anderem populis- Zinswende nach Rekordtiefs Fiskalpolitik des US-Präsidenten tische Strömungen nicht nur bei Anleiherenditen? Donald Trump in Form von in den USA und Europa, auch Im Berichtszeitraum kam es an Steuererleichterungen und einer in den Emerging Markets, wie den Anleihemärkten - wenn Erhöhung der Staatsausgaben zuletzt in Brasilien und Mexiko, auch uneinheitlich und unter- noch befeuert wurde. Vor die- wo Populisten in die Regie- schiedlich stark ausgeprägt - sem Hintergrund verzeichneten rungen gewählt wurden. Der unter Schwankungen zu stei- die US-Bondmärkte einen spür- eskalierende Handelskonflikt genden Renditen auf teilweise baren Renditeanstieg, der mit zwischen den USA einerseits immer noch sehr niedrigem merklichen Kursermäßigungen sowie Europa und China ande- Niveau, begleitet von nachge- einherging. Per saldo stiegen rerseits tat sein übriges. Zudem benden Kursen. Dabei konnten die Renditen zehnjähriger US- ging Italien im Rahmen seiner sich die Zinsen vor allem in den Staatsanleihen auf Jahressicht Haushaltspolitik auf Konfronta- USA weiter von ihren histori- von 2,4% p.a. auf 3,1% p.a. tionskurs zur Europäischen Uni- schen Tiefs lösen. Gleichzeitig on (EU). Darüber hinaus verun- wurde die Zentralbankdivergenz Diese Entwicklung in den USA sicherte weiterhin der drohende zwischen den USA und dem hatte auf den Euroraum jedoch Brexit die Marktteilnehmer, Euroraum immer deutlicher. An- nur phasenweise und partiell, insbesondere dahingehend, ob gesichts des Wirtschaftsbooms insgesamt aber kaum einen der mögliche Austritt des Ver- in den USA hob die US-Noten- zinstreibenden Effekt auf Staats- einigten Königreiches aus der bank Fed im Berichtszeitraum anleihen aus den europäischen EU geordnet oder - ohne ein den Leitzins in vier Schritten Kernmärkten wie beispielsweise weiter um einen Prozentpunkt Deutschland und Frankreich. Übereinkommen - ungeordnet vollzogen wird. Im Gegensatz auf einen Korridor von 2,25% - So gaben zehnjährige deutsche zu den politischen Unsicher- 2,50% p.a. an. Die Europäische Bundesanleihen im Jahresver- heiten standen die bislang eher Zentralbank (EZB) hingegen lauf 2018 unter Schwankungen zuversichtlichen, konventionel- hielt an ihrer Nullzinspolitik fest per saldo sogar von 0,42% p.a. len Wirtschaftsindikatoren. So und die Bank of Japan beließ auf 0,25% p. a. nach. Vielmehr expandierte die Weltwirtschaft ihren Leitzins bei -0,10% p.a. standen die Anleihemärkte kräftig, wenngleich sich deren Die Fed begründete ihre Zinser- der Euroländer - insbeson- Wachstum im Laufe der zwei- dere ab dem zweiten Quartal höhungen mit der Wachstums- ten Jahreshälfte 2018 etwas beschleunigung und der Voll- 2018 - unter dem Einfluss der verlangsamte. Die Geldpolitik beschäftigung am Arbeitsmarkt politischen und wirtschaftlichen war historisch gesehen immer in den USA. Zinserhöhungen Entwicklung in der Eurozone, noch sehr locker, trotz der welt- vor allem in den Ländern mit ermöglichten der US-Notenbank weit sehr hohen Verschuldung einer Überhitzung der US-Wirt- hoher Verschuldung und nied- und der Zinserhöhungen der schaft vorzubeugen, die durch rigem Wirtschaftswachstum. US-Notenbank. die prozyklische expansive In den Monaten Mai und Juni'



## Layout service

It now depends on whether we use Tensorflow or PyTorch. We opt for PyTorch, just because the model runs on a CPU.
Make sure, that the model has been loaded to your .cache.


```python
path_weights = dd.ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout_inf_only.pt")
path_configs = dd.ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout_inf_only.pt")
categories = dd.ModelCatalog.get_profile("layout/d2_model_0829999_layout_inf_only.pt").categories

categories
```




    {'1': <LayoutType.text>,
     '2': <LayoutType.title>,
     '3': <LayoutType.list>,
     '4': <LayoutType.table>,
     '5': <LayoutType.figure>}




```python
layout_detector = dd.D2FrcnnDetector(path_configs,path_weights,categories,device="cpu")
layout_comp = dd.ImageLayoutService(layout_detector)
```



We need to make sure, that the `ImageLayoutService` has been invoked before the `TextOrderService`.  


```python
pipe_comp_list.insert(0,layout_comp)
```


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.layouts
```

                                                                  
    [1219 17:03.32 @doctectionpipe.py:124]Processing sample_3.png
    [1219 17:03.34 @context.py:131]ImageLayoutService total: 1.7362 sec.
    [1219 17:03.36 @context.py:131]LanguageDetectionService total: 2.6793 sec.
    [1219 17:03.39 @context.py:131]TextExtractionService total: 2.7716 sec.
    [1219 17:03.39 @context.py:131]TextOrderService total: 0.0139 sec.



    [Layout(active=True, _annotation_id='59f04295-78f8-341d-afb6-9e3ac973a9a4', category_name=<LayoutType.text>, _category_name=<LayoutType.text>, category_id='1', score=0.9831385016441345, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=138.1197509765625, uly=393.2305908203125, lrx=552.4398193359375, lry=2188.858154296875, height=1795.6275634765625, width=414.320068359375)),
     Layout(active=True, _annotation_id='2e93a316-8eba-353c-aae9-510e99f2e61a', category_name=<LayoutType.text>, _category_name=<LayoutType.text>, category_id='1', score=0.9732153415679932, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=618.3724365234375, uly=872.4000854492188, lrx=1036.1072998046875, lry=2191.201416015625, height=1318.8013305664062, width=417.73486328125)),
     Layout(active=True, _annotation_id='771f645a-3153-3a02-99ff-cfb65593ebc2', category_name=<LayoutType.title>, _category_name=<LayoutType.title>, category_id='2', score=0.9648762941360474, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=135.84925842285156, uly=141.39108276367188, lrx=885.5213623046875, lry=274.14312744140625, height=132.75204467773438, width=749.6721038818359)),
     Layout(active=True, _annotation_id='2e737fb3-2323-321b-b96a-db83664e2e60', category_name=<LayoutType.text>, _category_name=<LayoutType.text>, category_id='1', score=0.9539102911949158, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=1102.441650390625, uly=1358.807373046875, lrx=1520.0106201171875, lry=2187.287841796875, height=828.48046875, width=417.5689697265625)),
     Layout(active=True, _annotation_id='7c22c1f3-8bdd-3eb4-9a6f-bf7cee6e5be4', category_name=<LayoutType.text>, _category_name=<LayoutType.text>, category_id='1', score=0.939558207988739, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=1099.98486328125, uly=851.87939453125, lrx=1520.9635009765625, lry=1322.9500732421875, height=471.0706787109375, width=420.9786376953125)),
     Layout(active=True, _annotation_id='8d7b85e6-f2f7-3967-80aa-0e3cedd10989', category_name=<LayoutType.text>, _category_name=<LayoutType.text>, category_id='1', score=0.6196556091308594, sub_categories={}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=615.2217407226562, uly=383.4162292480469, lrx=1244.133056640625, lry=444.7840576171875, height=61.367828369140625, width=628.9113159179688))]




```python
dp.text, dp.layouts[0].text
```




    ('', '')



Now this looks weird again, doesn't it? However the reason is still quite simple. We now get an empty text string because once we have a non-empty `dp.layouts` the routine responsible for creating `dp.text` will try to get the text from the `Layout`'s. But we haven't run any method that maps a `word` to some `Layout` object. We need to specify this by applying a `MatchingService`. We will also have to slightly change the configuration of the  `TextOrderService`.


```python
map_comp = dd.MatchingService(parent_categories=["text","title","list","table","figure"], child_categories=["word"],
                             matching_rule = 'ioa', threshold=0.6) # same setting as for the deepdoctection analyzer

order_comp = dd.TextOrderService(text_container=dd.LayoutType.word,
                                 floating_text_block_names=["text","title","list", "figure"],
                                 text_block_names=["text","title","list","table","figure"])

pipe_comp_list = [layout_comp, lang_detect_comp, text_comp, map_comp, order_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)
```


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.text
```


    [1219 17:03.41 @doctectionpipe.py:124]Processing sample_3.png
    [1219 17:03.43 @context.py:131]ImageLayoutService total: 1.6445 sec.
    [1219 17:03.45 @context.py:131]LanguageDetectionService total: 2.6846 sec.
    [1219 17:03.48 @context.py:131]TextExtractionService total: 2.8263 sec
    [1219 17:03.48 @context.py:131]MatchingService total: 0.0041 sec.
    [1219 17:03.48 @context.py:131]TextOrderService total: 0.0387 sec.





    '\nAnleihemärkte im Geschäftsjahr bis zum 31.12.2018\nSchwieriges Marktumfeld Die internationalen Anleihe- märkte entwickelten sich im Geschäftsjahr 2018 unter- schiedlich und phasenweise sehr volatil. Dabei machte sich bei den Investoren zunehmend Nervosität breit, was in steigen- den Risikoprämien zum Aus- druck kam. Grund hierfür waren Turbulenzen auf der weltpoli- tischen Bühne, die die politi- schen Risiken erhöhten. Dazu zählten unter anderem populis- tische Strömungen nicht nur in den USA und Europa, auch in den Emerging Markets, wie zuletzt in Brasilien und Mexiko, wo Populisten in die Regie- rungen gewählt wurden. Der eskalierende Handelskonflikt zwischen den USA einerseits sowie Europa und China ande- rerseits tat sein übriges. Zudem ging Italien im Rahmen seiner Haushaltspolitik auf Konfronta- tionskurs zur Europäischen Uni- on (EU). Darüber hinaus verun- sicherte weiterhin der drohende Brexit die Marktteilnehmer, insbesondere dahingehend, ob der mögliche Austritt des Ver- einigten Königreiches aus der EU geordnet oder - ohne ein Übereinkommen - ungeordnet vollzogen wird. Im Gegensatz zu den politischen Unsicher- heiten standen die bislang eher zuversichtlichen, konventionel- len Wirtschaftsindikatoren. So expandierte die Weltwirtschaft kräftig, wenngleich sich deren Wachstum im Laufe der zwei- ten Jahreshälfte 2018 etwas verlangsamte. Die Geldpolitik war historisch gesehen immer noch sehr locker, trotz der welt- weit sehr hohen Verschuldung und der Zinserhöhungen der US-Notenbank.\nEntwicklung der Leitzinsen in den USA und im Euroraum %p.a.\nFiskalpolitik des US-Präsidenten Donald Trump in Form von Steuererleichterungen und einer Erhöhung der Staatsausgaben noch befeuert wurde. Vor die- sem Hintergrund verzeichneten die US-Bondmärkte einen spür- baren Renditeanstieg, der mit merklichen Kursermäßigungen einherging. Per saldo stiegen die Renditen zehnjähriger US- Staatsanleihen auf Jahressicht von 2,4% p.a. auf 3,1% p.a.\nbei Anleiherenditen? Im Berichtszeitraum kam es an den Anleihemärkten - wenn auch uneinheitlich und unter- schiedlich stark ausgeprägt - unter Schwankungen zu stei- genden Renditen auf teilweise immer noch sehr niedrigem Niveau, begleitet von nachge- benden Kursen. Dabei konnten sich die Zinsen vor allem in den USA weiter von ihren histori- schen Tiefs lösen. Gleichzeitig wurde die Zentralbankdivergenz zwischen den USA und dem Euroraum immer deutlicher. An- gesichts des Wirtschaftsbooms in den USA hob die US-Noten- bank Fed im Berichtszeitraum den Leitzins in vier Schritten weiter um einen Prozentpunkt auf einen Korridor von 2,25% - 2,50% p.a. an. Die Europäische Zentralbank (EZB) hingegen hielt an ihrer Nullzinspolitik fest und die Bank of Japan beließ ihren Leitzins bei -0,10% p.a. Die Fed begründete ihre Zinser- höhungen mit der Wachstums- beschleunigung und der Voll- beschäftigung am Arbeitsmarkt in den USA. Zinserhöhungen ermöglichten der US-Notenbank einer Überhitzung der US-Wirt- schaft vorzubeugen, die durch die prozyklische expansive\nDiese Entwicklung in den USA hatte auf den Euroraum jedoch nur phasenweise und partiell, insgesamt aber kaum einen zinstreibenden Effekt auf Staats- anleihen aus den europäischen Kernmärkten wie beispielsweise Deutschland und Frankreich. So gaben zehnjährige deutsche Bundesanleihen im Jahresver- lauf 2018 unter Schwankungen per saldo sogar von 0,42% p.a. auf 0,25% p. a. nach. Vielmehr standen die Anleihemärkte der Euroländer - insbeson- dere ab dem zweiten Quartal 2018 - unter dem Einfluss der politischen und wirtschaftlichen Entwicklung in der Eurozone, vor allem in den Ländern mit hoher Verschuldung und nied- rigem Wirtschaftswachstum. In den Monaten Mai und Juni'



Finally, we got it!
