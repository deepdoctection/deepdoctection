
::: deepdoctection.extern
    options:
        docstring_style: google

::: deepdoctection.extern.base
    options:
        docstring_style: google
        filters: 
            - "!categories"
            - "!accepts_batch"

::: deepdoctection.extern.d2detect
    options:
        docstring_style: google

::: deepdoctection.extern.deskew
    options:
        docstring_style: google

::: deepdoctection.extern.doctrocr
    options:
        docstring_style: google
        filters: 
            - "!_get_doctr_requirements"
            - "!_load_model"

::: deepdoctection.extern.fastlang
    options:
        docstring_style: google

::: deepdoctection.extern.hfdetr
    options:
        docstring_style: google
        filters: 
            - "!_detr_post_processing"

::: deepdoctection.extern.hflayoutlm
    options:
        docstring_style: google

::: deepdoctection.extern.hflm
    options:
        docstring_style: google

::: deepdoctection.extern.model
    options:
        docstring_style: google

::: deepdoctection.extern.pdftext
    options:
        docstring_style: google

::: deepdoctection.extern.tessocr
    options:
        docstring_style: google
        filters:
            - "!TesseractError"

::: deepdoctection.extern.texocr
    options:
        docstring_style: google

::: deepdoctection.extern.tpdetect
    options:
        docstring_style: google
