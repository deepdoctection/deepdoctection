
::: deepdoctection.datasets
    options:
        docstring_style: google


::: deepdoctection.datasets.adapter
    options:
        docstring_style: google

::: deepdoctection.datasets.base
    options:
        docstring_style: google

::: deepdoctection.datasets.dataflow_builder
    options:
        docstring_style: google
        filters:
            - "!categories"
            - "!splits"

::: deepdoctection.datasets.info
    options:
        docstring_style: google
        filters:
            - "!cat_to_sub_cat"
            - "!_get_dict"

::: deepdoctection.datasets.registry
    options:
        docstring_style: google

::: deepdoctection.datasets.save
    options:
        docstring_style: google
