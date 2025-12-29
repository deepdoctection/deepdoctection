
# About **deep**doctection

## Documents

Documents are everywhere: in business, administration, and private life. They are a foundational medium for storing and 
communicating information. Well-crafted documents combine content, structure, and visual layout to guide the reader 
and highlight the most important data.

Machines, however, do not benefit from these human-centric features. Extracting structured information from documents 
is inherently difficult due to diverse layouts, complex tables, figures, and handwritten or visually embedded 
information. Multi-column text, non-standard reading orders, and forms with implicit key-value structures are just a 
few of the many challenges.

Deep learning offers powerful tools to address these challenges. Instead of relying on fixed, human-defined rules, 
neural networks can learn representations from large-scale data and capture structural patterns that enable accurate 
information extraction.

## Document AI

> *Document AI, or Document Intelligence, is a booming research topic with increased industrial demand in recent 
   years. It mainly refers to the process of automated understanding, classifying and extracting information with rich 
   typesetting formats from webpages, digital-born documents or scanned documents through AI technology.*
> [Document AI: Benchmarks, Models and Applications](https://arxiv.org/abs/2111.08609)

Document AI is a fast-evolving discipline. New research, datasets, and models appear regularly. While many projects 
offer pre-trained models and code, the usability in production systems is often limited.

## Purpose

**deep**doctection was built to bridge the gap between academic research and practical document automation. It wraps 
third-party models into a unified framework, making them easy to use and combine. When models are robust and 
pre-trained weights are available, **deep**doctection enables quick experimentation and deployment.

Many models specialize in single tasks (e.g., table detection, NER, OCR). But real-world use cases require combining 
multiple components in a single workflow. **deep**doctection offers such pipelines and supports chaining 
deeplearning and rule-based modules in a flexible and modular architecture.

## Pipelines

Not every document requires the same processing pipeline:

* Native PDFs may not need OCR.
* Tables may or may not be relevant.
* Classification tasks might rely on visual and textual features combined.

With **deep**doctection, pipelines can be constructed using components for layout analysis, OCR, PDF text extraction, 
visual-language models, and more.

The [**analyzer**](./tutorials/Analyzer_Get_Started.md) is an example of a composable pipeline.

## Datasets

Document AI relies on datasets, but labeled data is rare:

* Most business documents are confidential.
* Annotation is expensive and domain-specific.

While **deep**doctection does not provide labeling tools, it supports converting your labeled data into a structured 
[**dataset format**](./concepts/Datasets.md) for training and evaluation. Templates exist for several public datasets 
such as [Publaynet][dd_datasets.instances.publaynet] or [XFUND][dd_datasets.instances.xfund], 
which can serve as blueprints.

## Fine-tuning

Pretrained models may not generalize well to your documents. But with **transfer learning**, they can be adapted using 
limited labeled data.

**deep**doctection provides [training scripts][deepdoctection.train] and an 
[evaluator](./concepts/Evaluation.md) to train and validate models on custom datasets.

## Large Language Models (LLMs) and Vision-Language Models (VLLMs)

Large Language Models and multimodal Vision-Language Models are increasingly being used for document understanding 
tasks such as classification, key-value extraction, and summarization. However, there are critical challenges that 
must be addressed:

### 1. Lack of Grounding / Traceability

LLMs typically output plain text without bounding boxes or positional references. This makes it difficult to:

* Verify whether the extracted information actually exists in the original document.
* Detect hallucinations without human inspection.

A key requirement for trustworthy document extraction is **back-tracing**: the ability to map model outputs back to 
document coordinates or text regions. This traceability is a core principle of **deep**doctection.

### 2. Valid Evaluation Datasets

Many public document datasets may have been included in the training data of large foundation models. As a result:

* Evaluation on such data may be biased or over-optimistic.
* Truly informative benchmarks require private, proprietary datasets.

**deep**doctection facilitates dataset integration for such secure and domain-specific benchmarks.

### 3. Integration Roadmap

We are actively working to make LLMs and VLLMs first-class citizens in the **deep**doctection ecosystem. Future pipeline 
components will support:

* Structured prompt generation from visual layout.
* Answer-to-position alignment.
* Hybrid pipelines combining traditional layout parsing with LLM inference.

Stay tuned as we bring the capabilities of foundation models into structured and verifiable document pipelines.
