Notebooks, Tutorials, Scripts and Notes
________________________________________________________

The notebooks are a markdown copy of the original Jupyter notebooks, which can be found in the repo's directory of
the same name.

Tutorials offer a deeper introduction to **deep**\doctection and cover topics to enable experiments and further
development.

The scripts contain code snippets that set out configurations and hyperparameter settings for training or contain
evaluation scripts that can be used to understand the benchmarks.

In the notes you will find some further questions about certain models. In contrast to the scripts, model variants that
require additional implementation are often examined. Essentially, I would like to pursue questions as to whether
certain adjustments are worthwhile from the point of view of cost-benefit analysis.

.. toctree::
  :maxdepth: 1
  :caption: Notebooks

  get_started_notebook
  custom_pipeline_notebook
  datasets_and_eval_notebook
  using_layoutlm_for_sequence_classification

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  architecture
  layout_parsing_structure
  datasets
  layoutlm_for_token_classification

.. toctree::
  :maxdepth: 1
  :caption: Scripts

  eval_tp_vs_d2
  scripts
  table_recognition_eval

.. toctree::
  :maxdepth: 1
  :caption: Notes

  layoutlm_various_topics_pt1
  layoutlm_various_topics_pt2