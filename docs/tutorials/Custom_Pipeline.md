<p align="center">
  <img src="https://github.com/deepdoctection/deepdoctection/raw/master/docs/tutorials/_imgs/dd_logo.png" alt="Deep Doctection Logo" width="60%">
  <h3 align="center">
  </h3>
</p>


# Project: Building a custom pipeline

The **deep**doctection analyzer is an example of a Document Layout Analysis pipeline. In this tutorial we'll build
a custom pipeline by adding successively components that are needed to fulfill a given task.  

We want a pipeline that extracts text from a document and detects the language. For language detection we will be
using a [Fasttext](https://github.com/facebookresearch/fastText) model.

## Getting the pre-trained model

```python
categories=dd.ModelCatalog.get_profile("fasttext/lid.176.bin").categories
categories_orig = dd.ModelCatalog.get_profile("fasttext/lid.176.bin").categories_orig
dd.ModelCatalog.get_profile("fasttext/lid.176.bin").model_wrapper # (1) 
```

??? info "Output"

    'FasttextLangDetector'

1. `model_wrapper` is the name of the model wrapper we need to use from the `deepdoctection.extern` module.


```python
path_weights=dd.ModelDownloadManager.maybe_download_weights_and_configs("fasttext/lid.176.bin")
```

## Model wrapper

We know from the `ModelCatalog` which wrapper we must use for the fasttext model, namely `FasttextLangDetector`.


```python
fast_text = dd.FasttextLangDetector(path_weights=path_weights, 
									categories=categories, 
									categories_orig=categories_orig)
```

We still need to choose how to extract text. 


```python
tess_ocr_config_path = dd.get_configs_dir_path() / "dd/conf_tesseract.yaml" # (1)
tesseract_ocr = dd.TesseractOcrDetector(tess_ocr_config_path)
```

1. If this file is not in your `.cache` you can find it in `deepdoctection.configs`.


## Language detection service and pipeline


```python
lang_detect_comp = dd.LanguageDetectionService(language_detector=fast_text,
											   text_detector=tesseract_ocr)
```

We can now build our very simple pipeline.


```python
pipe = dd.DoctectionPipe(pipeline_component_list=[lang_detect_comp])
```


When running the pipe, we get the language the document was written. 


```python
df = pipe.analyze(path="/path/to/image_dir")
df.reset_state()

dp = next(iter(df))
dp.language
```

??? info "Output"

    Languages.GERMAN


But, when getting the text, the response is somewhat disappointing: `dp.text` returns an empty string.

!!! info

    The reason for that is that `LanguageDetectionService` is not responsible for extracting text. It has an OCR model, 
    but the output is only used as input feed to the language detector. The text however is not persisted. If we had 
    added a `TextExtractionService` before `LanguageDetectionService` we could have omitted the OCR model in 
    the `LanguageDetectionService`. 


## Tesseract OCR detector

Next, we add a component for extracting text.

```python
tesseract_ocr = dd.TesseractOcrDetector(tess_ocr_config_path.as_posix(),["LANGUAGES=deu"])
tesseract_ocr.config
```

??? info "Output"

    {'LANGUAGES': 'deu', 'LINES': False, 'psm': 11}


```python
text_comp = dd.TextExtractionService(text_extract_detector=tesseract_ocr, 
                                     run_time_ocr_language_selection=True) # (1)
pipe_comp_list=[lang_detect_comp, text_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)
```

1. Setting `run_time_ocr_language_selection=True` will dynamically select the OCR model for text extraction based on 
   the predicted languages. This helps to get much improved OCR results, if you have documents with various languages.



```python
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
```


There is something unexpected: `dp.text` is still an empty string. On the other hand, we can clearly see that the 
`TextExtractionService` did its job.


```python
word_sample = dp.words[0]
len(dp.words), word_sample.characters, word_sample.bbox, word_sample.reading_order 
```

??? info "Output"

    (553, 'Anleihemärkte', [137.0, 158.0, 472.0, 195.0], None)


## Text ordering

The reason is, that we do not have inferred a reading order. If there is no reading order, there is no contiguous text. 
We treat text extraction as a character recognition problem only. If we want a reading order of predicted words, 
we need to do it by adding a designated service. 


```python
order_comp = dd.TextOrderService(text_container=dd.LayoutType.WORD)

pipe_comp_list=[lang_detect_comp, text_comp, order_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)

df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))
dp.text
```

??? info "Output"

     Anleihemärkte im Geschäftsjahr\nbis zum 31.12.2018\nSchwieriges Marktumfeld\nDie internationalen Anleihe-\nmärkte
     entwickelten sich im\nGeschäftsjahr 2018 unter-\nschiedlich und phasenweise\nsehr volatil. Dabei machte sich\nbei 
     den Investoren zunehmend\nNervosität breit, was in steigen-\nden Risikoprämien zum Aus-\ndruck kam. Grund hierfür 
     waren\nTurbulenzen auf der weltpoli-\ntischen Bühne, die die politi-\nschen Risiken erhöhten. Dazu\nzählten unter 
     anderem populis-\ntische Strömungen nicht nur\nin den USA und Europa, auch\nin den Emerging Markets, 
     wie\nzuletzt in Brasilien und Mexiko,\nwo Populisten in die Regie-\nrungen gewählt wurden. Der\neskalierende 
     Handelskonflikt\nzwischen den USA einerseits\nsowie Europa und China ande-\nrerseits tat sein übriges. Zudem\nging 
     Italien im Rahmen seiner\nHaushaltspolitik auf Konfronta-\ntionskurs zur Europäischen Uni-\non (EU). Darüber 
     hinaus verun-\nsicherte weiterhin der drohende\nBrexit die Marktteilnehmer,\ninsbesondere dahingehend, ob\nder 
     mögliche Austritt des Ver-\neinigten Königreiches aus der\nEU geordnet oder - ohne ein\nÜbereinkommen - 
     ungeordnet\nvollzogen wird. Im Gegensatz\nzu den politischen Unsicher-\nheiten standen die bislang 
     eher\nzuversichtlichen, konventionel-\nlen Wirtschaftsindikatoren. So\nexpandierte die Weltwirtschaft\nkräftig, 
     wenngleich sich deren\nWachstum im Laufe der zwei-\nten Jahreshälfte 2018 etwas\nverlangsamte. Die 
     Geldpolitik\nwar historisch gesehen immer\nnoch sehr locker, trotz der welt-\nweit sehr hohen Verschuldung\nund 
     der Zinserhöhungen der\nUS-Notenbank.\nEntwicklung der Leitzinsen in den USA und im Euroraum\n%p.a.
     \nu\nu\nu\nu\nu\nu\nu\nu\nu\nu\n12/08 12/09 12/10 12/11 12/12 12/13 12/14 12/15 12/16 12/17 12/18\nBE 
     Fed-Leitzins\nQuelle: Thomson Financial Datastream\nBE E28-Leitzins\nStand: 31.12.2018\n-1 u\nZinswende nach 
     Rekordtiefs\nbei Anleiherenditen?\nIm Berichtszeitraum kam es an\nden Anleihemärkten - wenn\nauch uneinheitlich 
     und unter-\nschiedlich stark ausgeprägt -\nunter Schwankungen zu stei-\ngenden Renditen auf teilweise\nimmer noch 
     sehr niedrigem\nNiveau, begleitet von nachge-\nbenden Kursen. Dabei konnten\nsich die Zinsen vor allem in den\nUSA 
     weiter von ihren histori-\nschen Tiefs lösen. Gleichzeitig\nwurde die Zentralbankdivergenz\nzwischen den USA und 
     dem\nEuroraum immer deutlicher. An-\ngesichts des Wirtschaftsbooms\nin den USA hob die US-Noten-\nbank Fed im 
     Berichtszeitraum\nden Leitzins in vier Schritten\nweiter um einen Prozentpunkt\nauf einen Korridor von 2,25% -\n2,
     50% p.a. an. Die Europäische\nZentralbank (EZB) hingegen\nhielt an ihrer Nullzinspolitik fest\nund die Bank of 
     Japan beließ\nihren Leitzins bei -0,10% p.a.\nDie Fed begründete ihre Zinser-\nhöhungen mit der 
     Wachstums-\nbeschleunigung und der Voll-\nbeschäftigung am Arbeitsmarkt\nin den USA. 
     Zinserhöhungen\nermöglichten der US-Notenbank\neiner Überhitzung der US-Wirt-\nschaft vorzubeugen, die durch\ndie 
     prozyklische expansive\nFiskalpolitik des US-Präsidenten\nDonald Trump in Form von\nSteuererleichterungen und 
     einer\nErhöhung der Staatsausgaben\nnoch befeuert wurde. Vor die-\nsem Hintergrund verzeichneten\ndie 
     US-Bondmärkte einen spür-\nbaren Renditeanstieg, der mit\nmerklichen Kursermäßigungen\neinherging. Per saldo 
     stiegen\ndie Renditen zehnjähriger US-\nStaatsanleihen auf Jahressicht\nvon 2,4% p.a. auf 3,1% p.a.\nDiese 
     Entwicklung in den USA\nhatte auf den Euroraum jedoch\nnur phasenweise und partiell,\ninsgesamt aber kaum 
     einen\nzinstreibenden Effekt auf Staats-\nanleihen aus den europäischen\nKernmärkten wie 
     beispielsweise\nDeutschland und Frankreich.\nSo gaben zehnjährige deutsche\nBundesanleihen im Jahresver-\nlauf 
     2018 unter Schwankungen\nper saldo sogar von 0,42% p.a.\nauf 0,25% p. a. nach. Vielmehr\nstanden die 
     Anleihemärkte\nder Euroländer - insbeson-\ndere ab dem zweiten Quartal\n2018 - unter dem Einfluss der\npolitischen 
     und wirtschaftlichen\nEntwicklung in der Eurozone,\nvor allem in den Ländern mit\nhoher Verschuldung und 
     nied-\nrigem Wirtschaftswachstum.\nIn den Monaten Mai und Juni


At least, we got some text. The beginning sounds good. But once the text comes to the region where the second and 
third column also have text lines, the order service does not distinguish between columns. So we must identify columns. 
For that we use the layout analyzer.

## Layout service


```python
path_weights = dd.ModelCatalog.get_full_path_weights("layout/d2_model_0829999_layout_inf_only.pt")
path_configs = dd.ModelCatalog.get_full_path_configs("layout/d2_model_0829999_layout_inf_only.pt")
categories = dd.ModelCatalog.get_profile("layout/d2_model_0829999_layout_inf_only.pt").categories

layout_detector = dd.D2FrcnnDetector(path_yaml=path_configs,
                                     path_weights=path_weights,
                                     categories=categories,
                                     device="cpu")
layout_comp = dd.ImageLayoutService(layout_detector)
```

We need to make sure, that the `ImageLayoutService` has been invoked before `TextOrderService`.  


```python
pipe_comp_list=[layout_comp, lang_detect_comp, text_comp, order_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)


df = pipe.analyze(path=image_path)
df.reset_state()

dp = next(iter(df))
dp.text, dp.layouts[0].text
```

??? info "Output"

    ('\nAnleihemärkte im Geschäftsjahr\nbis zum 31.12.2018\nSchwieriges Marktumfeld\n\nDie internationalen Anleihe-
    \nmärkte entwickelten sich im\nGeschäftsjahr 2018 unter-\nschiedlich und phasenweise\nsehr volatil. Dabei machte 
    sich\nbei den Investoren zunehmend\nNervosität breit, was in steigen-\nden Risikoprämien zum Aus-\ndruck kam. 
    Grund hierfür waren\nTurbulenzen auf der weltpoli-\ntischen Bühne, die die politi-\nschen Risiken erhöhten. 
    Dazu\nzählten unter anderem populis-\ntische Strömungen nicht nur\nin den USA und Europa, auch\nin den Emerging 
    Markets, wie\nzuletzt in Brasilien und Mexiko,\nwo Populisten in die Regie-\nrungen gewählt wurden. 
    Der\neskalierende Handelskonflikt\nzwischen den USA einerseits\nsowie Europa und China ande-\nrerseits tat sein 
    übriges. Zudem\nging Italien im Rahmen seiner\nHaushaltspolitik auf Konfronta-\ntionskurs zur Europäischen 
    Uni-\non (EU). Darüber hinaus verun-\nsicherte weiterhin der drohende\nBrexit die Marktteilnehmer,\ninsbesondere 
    dahingehend, ob\nder mögliche Austritt des Ver-\neinigten Königreiches aus der\nEU geordnet oder - ohne 
    ein\nÜbereinkommen - ungeordnet\nvollzogen wird. Im Gegensatz\nzu den politischen Unsicher-\nheiten standen die 
    bislang eher\nzuversichtlichen, konventionel-\nlen Wirtschaftsindikatoren. So\nexpandierte die 
    Weltwirtschaft\nkräftig, wenngleich sich deren\nWachstum im Laufe der zwei-\nten Jahreshälfte 2018 
    etwas\nverlangsamte. Die Geldpolitik\nwar historisch gesehen immer\nnoch sehr locker, trotz der welt-\nweit sehr 
    hohen Verschuldung\nund der Zinserhöhungen der\nUS-Notenbank.\n\nBE Fed-Leitzins\nQuelle: Thomson Financial 
    Datastream\nBE E28-Leitzins\nStand: 31.12.2018\n\nUSA weiter von ihren histori-\nvon 2,4% p.a. auf 3,1% p.a.
    \nschen Tiefs lösen. Gleichzeitig\nwurde die Zentralbankdivergenz\nDiese Entwicklung in den USA\n\nzwischen den 
    USA und dem\nhatte auf den Euroraum jedoch\nEuroraum immer deutlicher. An-\nnur phasenweise und partiell,\ngesichts 
    des Wirtschaftsbooms\ninsgesamt aber kaum einen\nin den USA hob die US-Noten-\nzinstreibenden Effekt auf 
    Staats-\nbank Fed im Berichtszeitraum\nanleihen aus den europäischen\nden Leitzins in vier Schritten\nKernmärkten 
    wie beispielsweise\nweiter um einen Prozentpunkt\nDeutschland und Frankreich.\nauf einen Korridor von 2,25% 
    -\nSo gaben zehnjährige deutsche\n2,50% p.a. an. Die Europäische\nBundesanleihen im Jahresver-\nZentralbank (EZB) 
    hingegen\nlauf 2018 unter Schwankungen\nIn den Monaten Mai und Juni\n\nEntwicklung der Leitzinsen in den USA und 
    im Euroraum\n%p.a.\n-1 u\nu\nu\nu\nu\nu\nu\nu\nu\nu\nu\n12/08 12/09 12/10 12/11 12/12 12/13 12/14 12/15 12/16 
    12/17 12/18\nZinswende nach Rekordtiefs\n\nFiskalpolitik des US-Präsidenten\nbei Anleiherenditen?\nDonald Trump 
    in Form von\nIm Berichtszeitraum kam es an\nSteuererleichterungen und einer\nden Anleihemärkten - wenn\nErhöhung 
    der Staatsausgaben\nauch uneinheitlich und unter-\nnoch befeuert wurde. Vor die-\nschiedlich stark ausgeprägt 
    -\nsem Hintergrund verzeichneten\nunter Schwankungen zu stei-\ndie US-Bondmärkte einen spür-\ngenden Renditen auf 
    teilweise\nbaren Renditeanstieg, der mit\nimmer noch sehr niedrigem\nmerklichen Kursermäßigungen\nNiveau, 
    begleitet von nachge-\neinherging. Per saldo stiegen\nbenden Kursen. Dabei konnten\ndie Renditen zehnjähriger 
    US-\nsich die Zinsen vor allem in den\nStaatsanleihen auf Jahressicht\nhielt an ihrer Nullzinspolitik fest\nper 
    saldo sogar von 0,42% p.a.\nund die Bank of Japan beließ\nauf 0,25% p. a. nach. Vielmehr\nihren Leitzins bei -0,
    10% p.a.\nstanden die Anleihemärkte\nDie Fed begründete ihre Zinser-\nder Euroländer - insbeson-\nhöhungen mit 
    der Wachstums-\ndere ab dem zweiten Quartal\nbeschleunigung und der Voll-\n2018 - unter dem Einfluss 
    der\nbeschäftigung am Arbeitsmarkt\npolitischen und wirtschaftlichen\nin den USA. Zinserhöhungen\nEntwicklung in 
    der Eurozone,\nermöglichten der US-Notenbank\nvor allem in den Ländern mit\neiner Überhitzung der US-Wirt-\nhoher 
    Verschuldung und nied-\nschaft vorzubeugen, die durch\nrigem Wirtschaftswachstum.\ndie prozyklische expansive',
     '')



Now this looks weird again. However the reason is still quite simple. We now get an empty text string 
because once we have a non-empty `dp.layouts` the routine responsible for creating `dp.text` will try to get the text 
from the `Layout`s. But we haven't run any method that maps a `word` to some `Layout` object. We need to specify this 
by running a `MatchingService`. We will also have to slightly change the configuration of the  `TextOrderService`.


```python
map_comp = dd.MatchingService(parent_categories=["text","title","list","table","figure"], 
                              child_categories=["word"],
                              matching_rule = 'ioa', 
                              threshold=0.6)

order_comp = dd.TextOrderService(text_container=dd.LayoutType.WORD,
                                 floating_text_block_categories=["text","title","list", "figure"],
                                 text_block_categories=["text","title","list","table","figure"])

pipe_comp_list = [layout_comp, lang_detect_comp, text_comp, map_comp, order_comp]
pipe = dd.DoctectionPipe(pipeline_component_list=pipe_comp_list)
df = pipe.analyze(path=image_path)
df.reset_state()
dp = next(iter(df))

dp.text
```

??? info "Output"

    Anleihemärkte im Geschäftsjahr bis zum 31.12.2018\nSchwieriges Marktumfeld Die internationalen Anleihe- märkte
    entwickelten sich im Geschäftsjahr 2018 unter- schiedlich und phasenweise sehr volatil. Dabei machte sich bei den 
    Investoren zunehmend Nervosität breit, was in steigen- den Risikoprämien zum Aus- druck kam. Grund hierfür waren 
    Turbulenzen auf der weltpoli- tischen Bühne, die die politi- schen Risiken erhöhten. Dazu zählten unter anderem 
    populis- tische Strömungen nicht nur in den USA und Europa, auch in den Emerging Markets, wie zuletzt in 
    Brasilien und Mexiko, wo Populisten in die Regie- rungen gewählt wurden. Der eskalierende Handelskonflikt zwischen 
    den USA einerseits sowie Europa und China ande- rerseits tat sein übriges. Zudem ging Italien im Rahmen seiner 
    Haushaltspolitik auf Konfronta- tionskurs zur Europäischen Uni- on (EU). Darüber hinaus verun- sicherte weiterhin 
    der drohende Brexit die Marktteilnehmer, insbesondere dahingehend, ob der mögliche Austritt des Ver- einigten 
    Königreiches aus der EU geordnet oder - ohne ein Übereinkommen - ungeordnet vollzogen wird. Im Gegensatz zu den 
    politischen Unsicher- heiten standen die bislang eher zuversichtlichen, konventionel- len Wirtschaftsindikatoren. 
    So expandierte die Weltwirtschaft kräftig, wenngleich sich deren Wachstum im Laufe der zwei- ten Jahreshälfte 
    2018 etwas verlangsamte. Die Geldpolitik war historisch gesehen immer noch sehr locker, trotz der welt- weit 
    sehr hohen Verschuldung und der Zinserhöhungen der US-Notenbank.\nEntwicklung der Leitzinsen in den USA und im 
    Euroraum %p.a.\nZinswende nach Rekordtiefs\nbei Anleiherenditen? Im Berichtszeitraum kam es an den Anleihemärkten 
    - wenn auch uneinheitlich und unter- schiedlich stark ausgeprägt - unter Schwankungen zu stei- genden Renditen auf 
    teilweise immer noch sehr niedrigem Niveau, begleitet von nachge- benden Kursen. Dabei konnten sich die Zinsen 
    vor allem in den USA weiter von ihren histori- schen Tiefs lösen. Gleichzeitig wurde die Zentralbankdivergenz 
    zwischen den USA und dem Euroraum immer deutlicher. An- gesichts des Wirtschaftsbooms in den USA hob die US-Noten- 
    bank Fed im Berichtszeitraum den Leitzins in vier Schritten weiter um einen Prozentpunkt auf einen Korridor von 2,
    25% - 2,50% p.a. an. Die Europäische Zentralbank (EZB) hingegen hielt an ihrer Nullzinspolitik fest und die Bank 
    of Japan beließ ihren Leitzins bei -0,10% p.a. Die Fed begründete ihre Zinser- höhungen mit der Wachstums- 
    beschleunigung und der Voll- beschäftigung am Arbeitsmarkt in den USA. Zinserhöhungen ermöglichten der 
    US-Notenbank einer Überhitzung der US-Wirt- schaft vorzubeugen, die durch die prozyklische expansive\n-1 u u u u 
    u u u u u u u 12/08 12/09 12/10 12/11 12/12 12/13 12/14 12/15 12/16 12/17 12/18\nBE Fed-Leitzins\nQuelle: 
    Thomson Financial Datastream\nBE E28-Leitzins\nStand: 31.12.2018\nFiskalpolitik des US-Präsidenten Donald Trump in 
    Form von Steuererleichterungen und einer Erhöhung der Staatsausgaben noch befeuert wurde. Vor die- sem Hintergrund 
    verzeichneten die US-Bondmärkte einen spür- baren Renditeanstieg, der mit merklichen Kursermäßigungen einherging. 
    Per saldo stiegen die Renditen zehnjähriger US- Staatsanleihen auf Jahressicht von 2,4% p.a. auf 3,1% p.a.
    \nDiese Entwicklung in den USA hatte auf den Euroraum jedoch nur phasenweise und partiell, insgesamt aber kaum 
    einen zinstreibenden Effekt auf Staats- anleihen aus den europäischen Kernmärkten wie beispielsweise Deutschland 
    und Frankreich. So gaben zehnjährige deutsche Bundesanleihen im Jahresver- lauf 2018 unter Schwankungen per saldo 
    sogar von 0,42% p.a. auf 0,25% p. a. nach. Vielmehr standen die Anleihemärkte der Euroländer - insbeson- dere ab 
    dem zweiten Quartal 2018 - unter dem Einfluss der politischen und wirtschaftlichen Entwicklung in der Eurozone, vor 
    allem in den Ländern mit hoher Verschuldung und nied- rigem Wirtschaftswachstum. In den Monaten Mai und Juni



Finally, we got it!
