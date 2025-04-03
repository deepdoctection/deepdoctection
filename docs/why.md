# About **deep**doctection


## Documents

Documents in the business environment as well as in private use are ubiquitous and represent an essential source of
information of various kinds for readers. Even more, documents represent an essential basis for humanity to record
information and make it consumable for others.

Good documents live from their content, from their structure, but also from the visual presentation of their content.
The visual aspect should possibly simplify the search for certain information, but also make the core
information easily removable. What is important about these points is that they address people's perception so that
they can internalize the content.

A priori, machines cannot benefit from those features at all. This makes automated information extraction from documents
so difficult. Multi-column layouts with intuitive, but perhaps not always conventional reading orders, complex
structured tables that only make sense in the context of a preceding title or note, figures and diagrams or even forms
that store key-value pairs like a structured interview, all of these structures pose enormous challenges to the
acquisition of information for machines. The problems in this environment could go on forever!

With the new possibilities that deep learning makes available, there are also completely new ways of tackling the
problem of information extraction. The capacity of the models to generate features from low-level structures and thus
to represent good probability distributions on the basis of large amounts of data can lead to solutions that are far
superior to conventional approaches with human-defined rules.


## Document AI

Quoted from [Document AI: Benchmarks, Models and Applications](https://arxiv.org/abs/2111.08609):

    *[...]Document AI, or Document Intelligence, is a booming research topic with increased industrial
    demand in recent years. It mainly refers to the process of automated understanding, classifying
    and extracting information with rich typesetting formats from webpages, digital-born documents or
    scanned documents through AI technology. Due to the diversity of layouts and formats, low-quality
    scanned document images, and the complexity of the template structure, Document AI is a very
    challenging task and has attracted widespread attention in related research areas.[...]*


## Purpose

Document AI as a discipline continues to develop and not a day goes by without a new paper with new
results on this topic being published on arxiv. In many cases, researchers and developers have also established
to provide a repo with code for verification or evaluation, experiments and further exploration. One key question,
however, remains: How can this great work be made more usable? **deep**doctection aims to make this work accessible
through **wrappers** of third party libraries. The prerequisite for this is that pre-trained models are available and
the source code is robust.

Document Analysis and Visual Document Understanding consists of many facets, but often models try to solve one task. 
In order to achieve a usable service, one has to integrate different solutions into one framework. **deep**doctection 
offers this framework so that you can call different services one after another in a **pipeline** using just a few 
command lines.


## Pipelines

Not every document class needs to be processed in the same way: 

- Native PDF documents do not require a computationally intensive OCR process. 
- For pure text extraction, you may want to consider tables separately or not at all. 
- However, you may also want to perform text or token classification with Foundation Models. 

With **deep**doctection, individual tasks (layout analysis, OCR, PDF text mining, LayoutLM call) can be combined in a 
pipeline. The components of a pipeline can be some that call models but also components that are purely rule-based. 

[**deep**doctection's analyzer](./tutorials/get_started_notebook.md) is an example of a pipeline.  


## Datasets

Document AI datasets, especially labeled datasets are rare. The reason for this:

- Documents in companies are internal and not intended for the public. 
- Datasets often have to be labeled manually. 

**deep**doctection does not offer a labeling framework but once you have a labeled dataset it offers the possibility 
to set up [**datasets**](./tutorials/datasets.md) from a template so that they can be used for fine-tuning models. 
There are some examples of public datasets (e.g. [Publaynet][deepdoctection.datasets.instances.publaynet], 
[XFund][deepdoctection.datasets.instances.xfund], ...) that can be used to train models and serve as blue-print


## Fine-tuning

The likelihood that the model knows what your documents look like and process them to your satisfaction is low. 

But with **transfer-learning/fine-tuning** models can be adjusted to your documents. If the models are pre-trained, 
you need much less labelled data.

**deep**doctection offers [**training scripts**][deepdoctection.train] for many of the models provided, with which 
fine-tuning on a custom dataset can be triggered quickly. On top of set, there is a very powerful 
[**evaluator**](./tutorials/datasets_and_eval_notebook.md) so that you can compare the predictions with your gold 
standard.
