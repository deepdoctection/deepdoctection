Why deepdoctection ?
==============================

Documents
--------------------------------

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


Document AI
--------------------------------

Quoted from `Document AI: Benchmarks, Models and Applications  <https://arxiv.org/abs/2111.08609>`_:

    *[...]Document AI, or Document Intelligence, is a booming research topic with increased industrial
    demand in recent years. It mainly refers to the process of automated understanding, classifying
    and extracting information with rich typesetting formats from webpages, digital-born documents or
    scanned documents through AI technology. Due to the diversity of layouts and formats, low-quality
    scanned document images, and the complexity of the template structure, Document AI is a very
    challenging task and has attracted widespread attention in related research areas.[...]*


Purpose
--------------------------------

Document AI as a discipline continues to develop and not a day goes by without a new paper with new
results on this topic being published on arxiv. In many cases, researchers and developers have also established
to provide a repo with code for verification or evaluation, experiments and further exploration. One key question,
however, remains: How can this great work be made more usable? Deep-Doctection aims to make this work accessible through
**wrappers or new implementations**. The prerequisite for this is that pre-trained models are available or the
underlying datasets are publicly accessible.

Document Analysis consists of many facets, but often models try to solve one task. In order to achieve a usable service,
one has to integrate different solutions in a framework. **deep**\doctection offers a framework so that you can call
different services one after the other in a **pipeline** using just a few command lines.

:ref:`Pipelines`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in **deep**doctection Analyzer performs layout analysis (title, text, figure, list, table), table
segmentation and extraction, text extraction via OCR or (and later PDF text mining also) and reading order deduction.
Another feature is the implementation of a **generic data model** with which all essential information (image, objects,
annotations, text, relations between objects) can be transported uniformly through various components.

Fine Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fine tuning** plays an important role for making a model applicable in specific tasks. Documents are so versatile
that it is impossible for anyone to provide a representative dataset. This is made even more difficult by the fact
that business documents represent business secrets and are therefore not freely available. Fine-tuning allows the model
to get a bias in such a way that predictions are optimized on the class of documents they are fine tuned on.
**deep**\doctection offers a simple training interface for the models, so that pre-trained models can be fine-tuned with
specially created datasets and new tasks can also be carried out.


:ref:`Datasets`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As soon as the term training comes up, you naturally have to think about **datasets** as well. Deep-Doctection provides
an interface (inspired by **Huggingface** datasets) for creating datasets. Interfaces for benchmark datasets (Publaynet,
Pubtabnet, ...) are already stored. Conversion scripts allow essential document layout analysis tasks to be fine-tuned
or trained from scratch with a few lines of code.


Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **evaluation**\, tools such as metrics are necessary to monitor the performance of the learning development using
objective criteria. **deep**\doctection provides a simple evaluation interface so that an evaluation of the predictor can
be scripted in just a few lines by selecting the predictor, metric and dataset.

