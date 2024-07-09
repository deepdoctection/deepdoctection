# Building a custom pipeline

The **deep**doctection analyzer is an example of a Document Layout Analysis pipeline. In this tutorial we'll show you the concepts so that you can build a pipeline yourself and according the needs you have.


```python
from pathlib import Path
import deepdoctection as dd
```

The idea is not that difficult: There are **models** that fulfill a given task, there are **pipeline components** or **pipeline backbones** that invoke models and take care of pre- and post-processing results. There are also pipeline backbones that do not invoke models but only consolidate results. 

And there is the **pipeline** that puts everything together.

## Catalog and registries

You can get the essential information for pre-trained model from the `ModelCatalog`. 


```python
dd.print_model_infos(add_description=False,add_config=False,add_categories=False)
```

Let's select fasttext language detector. We need the categories that the model predicts and the model wrapper. `fasttext/lid.176.bin` is just an artifact. 


```python
categories=dd.ModelCatalog.get_profile("fasttext/lid.176.bin").categories
```


```python
dd.ModelCatalog.get_profile("fasttext/lid.176.bin").model_wrapper
```




    'FasttextLangDetector'



We can download `lid.176.bin` with help of the `ModelDownloadManager`.


```python
path_weights=dd.ModelDownloadManager.maybe_download_weights_and_configs("fasttext/lid.176.bin")
```

## Model wrapper

We know from the `ModelCatalog` which wrapper we must use for the fasttext model, namely `FasttextLangDetector`.


```python
fast_text = dd.FasttextLangDetector(path_weights, categories)
```

We are not done yet, because we still need to choose how to extract text. Let's simply stick to Tesseract and use the default english setting.


```python
tess_ocr_config_path = dd.get_configs_dir_path() / "dd/conf_tesseract.yaml"  # This file will be in your .cache if you ran the analyzer before. 
# Otherwise make sure to copy the file from 'configs/conf_tesseract.yaml'

tesseract_ocr = dd.TesseractOcrDetector(tess_ocr_config_path.as_posix())
```

## Pipeline backbone

Similar to models we have a pipeline component registry. Having this starting point we can select the right backbone. Check the API documentation to see what the components are used for.


```python
dd.pipeline_component_registry.get_all()
```

    {'SubImageLayoutService': deepdoctection.pipe.cell.SubImageLayoutService,
     'ImageCroppingService': deepdoctection.pipe.concurrency.MultiThreadPipelineComponent,
     'MatchingService': deepdoctection.pipe.common.MatchingService,
     'PageParsingService': deepdoctection.pipe.common.PageParsingService,
     'AnnotationNmsService': deepdoctection.pipe.common.AnnotationNmsService,
     'ImageParsingService': deepdoctection.pipe.common.ImageParsingService,
     'LanguageDetectionService': deepdoctection.pipe.language.LanguageDetectionService,
     'ImageLayoutService': deepdoctection.pipe.layout.ImageLayoutService,
     'LMTokenClassifierService': deepdoctection.pipe.lm.LMTokenClassifierService,
     'LMSequenceClassifierService': deepdoctection.pipe.lm.LMSequenceClassifierService,
     'TextOrderService': deepdoctection.pipe.order.TextOrderService,
     'TableSegmentationRefinementService': deepdoctection.pipe.refine.TableSegmentationRefinementService,
     'TableSegmentationService': deepdoctection.pipe.segment.TableSegmentationService,
     'TextExtractionService': deepdoctection.pipe.text.TextExtractionService,
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

When running the pipe, we get the language the document was written. 


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.language
```

When getting the text, the response is somewhat disappointing.


```python
dp.text
```




    ''



The reason for that is that `LanguageDetectionService` is not responsible for extracting text. It has an OCR model, but the output is only used as input feed to the language detector. The text however is not persisted. If we had added a `TextExtractionService` before `LanguageDetectionService` we could have omitted the OCR model in the `LanguageDetectionService`. 

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

This is something unexpected. Why don't we generate any text? We can clearly see that the `TextExtractionService` did its job.

```python
word_sample = dp.words[0]
len(dp.words), word_sample.CHARACTERS, word_sample.bbox, word_sample.READING_ORDER 
```


    (553, 'Anleihemärkte', [137.0, 158.0, 472.0, 195.0], None)



## Text ordering

The reason is, that we do not have inferred a reading order. If there is no reading order, there is no contiguous text. We treat text extraction as a character recognition problem only. If we want a reading order of predicted words, we need to do it ourself. So let's add the `TextOrderService`.

```python
order_comp = dd.TextOrderService(text_container=dd.LayoutType.WORD)
pipe_comp_list.append(order_comp)
```

At least, we got some text. The beginning sounds good. But once the text comes to the region where the second and third column also have text lines, the order service does not distinguish between columns. So we must identify columns. For that we use the layout analyzer.


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.text
```

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

We need to make sure, that the `ImageLayoutService` has been invoked before `TextOrderService`.  


```python
pipe_comp_list.insert(0,layout_comp)
```


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.layouts
```


```python
dp.text, dp.layouts[0].text
```


    ('\nAnleihemärkte im Geschäftsjahr\nbis zum 31.12.2018\n\nUS-Notenbank.\nSchwieriges Marktumfeld\nDie internationalen Anleihe- %p.a.\nmärkte entwickelten sich im\nGeschäftsjahr 2018 unter-\nschiedlich und phasenweise\nsehr volatil. Dabei machte sich\nbei den Investoren zunehmend\nNervosität breit, was in steigen-\nden Risikoprämien zum Aus-\n12/08 12/09 12/10 12/11 12/12 12/13 12/14 12/15 12/16 12/17 12/18\ndruck kam. Grund hierfür waren\nBE Fed-Leitzins\nQuelle: Thomson Financial Datastream\nTurbulenzen auf der weltpoli-\nBE E28-Leitzins\nStand: 31.12.2018\ntischen Bühne, die die politi-\nschen Risiken erhöhten. Dazu\n\nzählten unter anderem populis-\nZinswende nach Rekordtiefs\nFiskalpolitik des US-Präsidenten\ntische Strömungen nicht nur\nbei Anleiherenditen?\nDonald Trump in Form von\nin den USA und Europa, auch\nIm Berichtszeitraum kam es an\nSteuererleichterungen und einer\nin den Emerging Markets, wie\nden Anleihemärkten - wenn\nErhöhung der Staatsausgaben\nzuletzt in Brasilien und Mexiko,\nauch uneinheitlich und unter-\nnoch befeuert wurde. Vor die-\nwo Populisten in die Regie-\nschiedlich stark ausgeprägt -\nsem Hintergrund verzeichneten\nHaushaltspolitik auf Konfronta-\nUSA weiter von ihren histori-\nvon 2,4% p.a. auf 3,1% p.a.\ntionskurs zur Europäischen Uni-\nschen Tiefs lösen. Gleichzeitig\non (EU). Darüber hinaus verun-\nwurde die Zentralbankdivergenz\nDiese Entwicklung in den USA\n\nsicherte weiterhin der drohende\nzwischen den USA und dem\nhatte auf den Euroraum jedoch\nBrexit die Marktteilnehmer,\nEuroraum immer deutlicher. An-\nnur phasenweise und partiell,\ninsbesondere dahingehend, ob\ngesichts des Wirtschaftsbooms\ninsgesamt aber kaum einen\nder mögliche Austritt des Ver-\nin den USA hob die US-Noten-\nzinstreibenden Effekt auf Staats-\neinigten Königreiches aus der\nbank Fed im Berichtszeitraum\nanleihen aus den europäischen\nEU geordnet oder - ohne ein\nden Leitzins in vier Schritten\nKernmärkten wie beispielsweise\nÜbereinkommen - ungeordnet\nweiter um einen Prozentpunkt\nDeutschland und Frankreich.\nvollzogen wird. Im Gegensatz\nauf einen Korridor von 2,25% -\nSo gaben zehnjährige deutsche\nzu den politischen Unsicher-\n2,50% p.a. an. Die Europäische\nBundesanleihen im Jahresver-\nheiten standen die bislang eher\nZentralbank (EZB) hingegen\nlauf 2018 unter Schwankungen\nIn den Monaten Mai und Juni\n\nEntwicklung der Leitzinsen in den USA und im Euroraum\n\n\nrungen gewählt wurden. Der\nunter Schwankungen zu stei-\ndie US-Bondmärkte einen spür-\neskalierende Handelskonflikt\ngenden Renditen auf teilweise\nbaren Renditeanstieg, der mit\nzwischen den USA einerseits\nimmer noch sehr niedrigem\nmerklichen Kursermäßigungen\nsowie Europa und China ande-\nNiveau, begleitet von nachge-\neinherging. Per saldo stiegen\nrerseits tat sein übriges. Zudem\nbenden Kursen. Dabei konnten\ndie Renditen zehnjähriger US-\nging Italien im Rahmen seiner\nsich die Zinsen vor allem in den\nStaatsanleihen auf Jahressicht\nzuversichtlichen, konventionel-\nhielt an ihrer Nullzinspolitik fest\nper saldo sogar von 0,42% p.a.\nlen Wirtschaftsindikatoren. So\nund die Bank of Japan beließ\nauf 0,25% p. a. nach. Vielmehr\nexpandierte die Weltwirtschaft\nihren Leitzins bei -0,10% p.a.\nstanden die Anleihemärkte\nkräftig, wenngleich sich deren\nDie Fed begründete ihre Zinser-\nder Euroländer - insbeson-\nWachstum im Laufe der zwei-\nhöhungen mit der Wachstums-\ndere ab dem zweiten Quartal\nten Jahreshälfte 2018 etwas\nbeschleunigung und der Voll-\n2018 - unter dem Einfluss der\nverlangsamte. Die Geldpolitik\nbeschäftigung am Arbeitsmarkt\npolitischen und wirtschaftlichen\nwar historisch gesehen immer\nin den USA. Zinserhöhungen\nEntwicklung in der Eurozone,\nnoch sehr locker, trotz der welt-\nermöglichten der US-Notenbank\nvor allem in den Ländern mit\nweit sehr hohen Verschuldung\neiner Überhitzung der US-Wirt-\nhoher Verschuldung und nied-\nund der Zinserhöhungen der\nschaft vorzubeugen, die durch\nrigem Wirtschaftswachstum.\ndie prozyklische expansive\n-1 u\nu\nu\nu\nu\nu\nu\nu\nu\nu u\n',
     '')



Now this looks weird again, doesn't it? However the reason is still quite simple. We now get an empty text string because once we have a non-empty `dp.layouts` the routine responsible for creating `dp.text` will try to get the text from the `Layout`'s. But we haven't run any method that maps a `word` to some `Layout` object. We need to specify this by applying a `MatchingService`. We will also have to slightly change the configuration of the  `TextOrderService`.

```python
map_comp = dd.MatchingService(parent_categories=["text", "title", "list", "table", "figure"], child_categories=["word"],
                              matching_rule='ioa', threshold=0.6)  # same setting as for the deepdoctection analyzer

order_comp = dd.TextOrderService(text_container=dd.LayoutType.WORD,
                                 floating_text_block_categories=["text", "title", "list", "figure"],
                                 text_block_categories=["text", "title", "list", "table", "figure"])

pipe_comp_list = [layout_comp, lang_detect_comp, text_comp, map_comp, order_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)
```


```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.text
```


    'Anleihemärkte im Geschäftsjahr bis zum 31.12.2018\nSchwieriges Marktumfeld Die internationalen Anleihe- märkte entwickelten sich im Geschäftsjahr 2018 unter- schiedlich und phasenweise sehr volatil. Dabei machte sich bei den Investoren zunehmend Nervosität breit, was in steigen- den Risikoprämien zum Aus- druck kam. Grund hierfür waren Turbulenzen auf der weltpoli- tischen Bühne, die die politi- schen Risiken erhöhten. Dazu zählten unter anderem populis- tische Strömungen nicht nur in den USA und Europa, auch in den Emerging Markets, wie zuletzt in Brasilien und Mexiko, wo Populisten in die Regie- rungen gewählt wurden. Der eskalierende Handelskonflikt zwischen den USA einerseits sowie Europa und China ande- rerseits tat sein übriges. Zudem ging Italien im Rahmen seiner Haushaltspolitik auf Konfronta- tionskurs zur Europäischen Uni- on (EU). Darüber hinaus verun- sicherte weiterhin der drohende Brexit die Marktteilnehmer, insbesondere dahingehend, ob der mögliche Austritt des Ver- einigten Königreiches aus der EU geordnet oder - ohne ein Übereinkommen - ungeordnet vollzogen wird. Im Gegensatz zu den politischen Unsicher- heiten standen die bislang eher zuversichtlichen, konventionel- len Wirtschaftsindikatoren. So expandierte die Weltwirtschaft kräftig, wenngleich sich deren Wachstum im Laufe der zwei- ten Jahreshälfte 2018 etwas verlangsamte. Die Geldpolitik war historisch gesehen immer noch sehr locker, trotz der welt- weit sehr hohen Verschuldung und der Zinserhöhungen der US-Notenbank.\nEntwicklung der Leitzinsen in den USA und im Euroraum %p.a.\nZinswende nach Rekordtiefs\nbei Anleiherenditen? Im Berichtszeitraum kam es an den Anleihemärkten - wenn auch uneinheitlich und unter- schiedlich stark ausgeprägt - unter Schwankungen zu stei- genden Renditen auf teilweise immer noch sehr niedrigem Niveau, begleitet von nachge- benden Kursen. Dabei konnten sich die Zinsen vor allem in den USA weiter von ihren histori- schen Tiefs lösen. Gleichzeitig wurde die Zentralbankdivergenz zwischen den USA und dem Euroraum immer deutlicher. An- gesichts des Wirtschaftsbooms in den USA hob die US-Noten- bank Fed im Berichtszeitraum den Leitzins in vier Schritten weiter um einen Prozentpunkt auf einen Korridor von 2,25% - 2,50% p.a. an. Die Europäische Zentralbank (EZB) hingegen hielt an ihrer Nullzinspolitik fest und die Bank of Japan beließ ihren Leitzins bei -0,10% p.a. Die Fed begründete ihre Zinser- höhungen mit der Wachstums- beschleunigung und der Voll- beschäftigung am Arbeitsmarkt in den USA. Zinserhöhungen ermöglichten der US-Notenbank einer Überhitzung der US-Wirt- schaft vorzubeugen, die durch die prozyklische expansive\n-1 u u u u u u u u u u u 12/08 12/09 12/10 12/11 12/12 12/13 12/14 12/15 12/16 12/17 12/18\nBE Fed-Leitzins\nQuelle: Thomson Financial Datastream\nBE E28-Leitzins\nStand: 31.12.2018\nFiskalpolitik des US-Präsidenten Donald Trump in Form von Steuererleichterungen und einer Erhöhung der Staatsausgaben noch befeuert wurde. Vor die- sem Hintergrund verzeichneten die US-Bondmärkte einen spür- baren Renditeanstieg, der mit merklichen Kursermäßigungen einherging. Per saldo stiegen die Renditen zehnjähriger US- Staatsanleihen auf Jahressicht von 2,4% p.a. auf 3,1% p.a.\nDiese Entwicklung in den USA hatte auf den Euroraum jedoch nur phasenweise und partiell, insgesamt aber kaum einen zinstreibenden Effekt auf Staats- anleihen aus den europäischen Kernmärkten wie beispielsweise Deutschland und Frankreich. So gaben zehnjährige deutsche Bundesanleihen im Jahresver- lauf 2018 unter Schwankungen per saldo sogar von 0,42% p.a. auf 0,25% p. a. nach. Vielmehr standen die Anleihemärkte der Euroländer - insbeson- dere ab dem zweiten Quartal 2018 - unter dem Einfluss der politischen und wirtschaftlichen Entwicklung in der Eurozone, vor allem in den Ländern mit hoher Verschuldung und nied- rigem Wirtschaftswachstum. In den Monaten Mai und Juni\n'


Finally, we got it!
