Pipelines
==========================

The pipeline is the object with which the individual stages of the Document Analysis tasks can be carried out one after
the other.

A pipeline consists of a list of pipeline components that must be executed in the order in which they should be called.

The DoctectionPipe class is suitable for a quick start, as a: meth:`_entry` is already implemented there, in which
documents or individual pages are successively read in a directory and transferred to the core data model.

Example:

We want to build a pipeline in which AWS Textract is to be used as the OCR component and the deep doctection's own
table extraction component. A .JSONL file is to be output which contains the complete Deep Doctection Core data model.