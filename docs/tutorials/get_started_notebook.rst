.. figure:: ./pics/dd_logo.png
   :alt: title

   title

Getting started
===============

**deep**\ doctection is a package that can be used to extract text from
complex structured documents. These can be native PDFs but also scans.
In contrast to various text miners **deep**\ doctection makes use of
deep learning models either for solving OCR, vision or language
embedding problems. Neural networks and object detectors have proven to
not only identify objects on images, but also to detect structures like
titles, tables, figures or lists. Another advantage is that deep
learning models can be trained on your own data to improve accuracy.

This introductory notebook showcases the **deep**\ doctection analyzer.
The analyzer is an example of a built-in pipeline, which offers a
rudimentary framework to identify layout structures in documents and to
extract text and tables. We will be starting with a text extraction task
of business document.

Before starting, however, we have to say:

All models used when invoking the analyzer were trained on publicly
available data sets for document layout analysis (Publaynet, Pubtabnet).
These datasets contain document pages and tables from medical research
articles. This means that there is already a bias in the training data
set and it is not to be expected that layout analysis would deliver
precise results on documents of different domains. To improve precision
we refer to the **Fine Tuning Tutorial**, where we deal with improving
the parsing results of business reports.

Check also this `Huggingface
space <https://huggingface.co/spaces/deepdoctection/deepdoctection>`__
where models have been trained on a more diverse data set.

Choosing the kernel
-------------------

We assume that the installation was carried out as described in the
guidelines. If a virtual environment and a kernel have been created, the
deep-doc kernel can be chosen using the kernel selection at the upper
right corner.

.. figure:: ./pics/dd_kernel.png
   :alt: title

   title

You can check if the installation was successful by activating the next
cell.

.. code:: ipython3

    import os
    import cv2
    from matplotlib import pyplot as plt
    from IPython.core.display import HTML
    
    import deepdoctection as dd

Sample
------

Let’s first look at a sample page we want to process. (You will probably
need to change ``image_path``.)

.. code:: ipython3

    image_path = os.path.join(dd.get_package_path(),"notebooks/pics/samples/sample_2/sample_2.png")
    image = cv2.imread(image_path)
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(image)

.. figure:: ./pics/samples/sample_2/sample_2.png
   :alt: title

   title

Analyzer
--------

We now start by introducing the **deep**\ doctection analyzer. There is
a factory function ``get_dd_analyzer`` for that outputs a pre-configured
version.

Knowing the language in advance will increase the text output
significantly. As the language is german, we will pass:
``language='deu'``.

.. code:: ipython3

    analyzer = dd.get_dd_analyzer(language='deu')

Pipeline components
-------------------

The analyzer is an example of a pipeline that can be built depending on
the problem you want to tackle. The pipeline is made up of the building
blocks as described in the diagram

.. figure:: ./pics/dd_pipeline.png
   :alt: title

   title

The default setting performs layout recognition, table segmentation and
OCR extraction. You can turn table segmentation and OCR off in order to
get less but quicker results.

Beside detection and OCR tasks, some other components are needed
e.g. text matching and reading order. Text matching for instance tries
to match words to detected layout regions by measuring intersection over
area.

Both matching and reading order are purely rule based components.

Analyze methods
---------------

The ``analyze`` method has various transfer parameters. The ``path``
parameter can be used to transfer a path to a directory to the analyzer
or to a PDF document. If the path points to a directory, all individual
pages can processed successively provided they have a file name suffix
‘.png’, ‘.jpg’ or ‘.tif’. If your path points to a PDF document with
multiple pages the analyzer will work iteratively through all document
pages.

.. code:: ipython3

    path = os.path.join(dd.get_package_path(),"notebooks/pics/samples/sample_2")
    df = analyzer.analyze(path=path)
    df.reset_state()

You can see when activating the cell that not much has happened. Indeed,
the ``analyze`` method returns a generator and you need to create an
iterator so you can loop over the pages you wan to process.

We use the ``iter`` / ``next`` method here. The image is only processed
when the ``next`` function is called.

.. code:: ipython3

    doc=iter(df)
    page = next(doc)

Page object
-----------

A Page object is returned, which has some handy tools for vizualising a
retrieving the detected results. There are some attributes that store
meta data about the input.

.. code:: ipython3

    page.height, page.width, page.file_name




.. parsed-literal::

    (2339, 1654, 'sample_2.png')



.. code:: ipython3

    image = page.viz()

The viz method draws the identified layout bounding box components into
the image. These can be visualized with matplotlib.

The layout analysis reproduces the title, text and tables. In addition,
lists and figures, if any, are identified. We can see here that a table
with table cells was recognized on the page. In addition, the
segmentations such as rows and columns were framed. The row and column
positions can be seen in the cell names.

.. code:: ipython3

    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(image)

.. figure:: ./pics/output_16_1.png
   :alt: title

   title

We can use the ``get_text`` method to output the running text only.
Table content is not included in the output.

.. code:: ipython3

    print(page.get_text())


.. parsed-literal::

    
    Festlegung der VV und angemessene Risikoadjustierung
    Die VV-Pools der DWS Gruppe werden einer angemessenen Anpassung der Risiken unterzogen, die die Adjustierung ex ante als auch ex post umfasst. Die angewandte robuste Methode soll sicherstellen, dass bei der Festlegung der VV sowohl der risikoadjustierten Leistung als auch der Kapital- und Liquiditätsausstattung der DWS Gruppe Rechnung getragen wird. Die Er- mittlung des Gesamtbetrags der VV orientiert sich primär an (i) der Tragfähigkeit für die DWS Gruppe (das heißt, was „kann” die DWS Gruppe langfristig an VV im Einklang mit regulatorischen ‚Anforderungen gewähren) und (il) der Leistung (das heißt, was „sollte” die DWS Gruppe an VV gewähren, um für eine angemessene leistungsbezogene Vergütung zu sorgen und gleichzeitig den langfristigen Erfolg des Unternehmens zu sichern)
    Die DWS Gruppe hat für die Festlegung der VV auf Ebene der individuellen Mitarbeiter die „Grundsätze für die Festlegung der variablen Vergütung” eingeführt. Diese enthalten Informationen über die Faktoren und Messgrößen, die bei Entscheidungen zur IVV berücksichtigt werden müssen. Dazu zählen beispielsweise Investmentperformance, Kundenbindung, Erwägungen zur Unternehmenskultur sowie Zielvereinbarungen und Leistungsbeurteilung im Rahmen des „Ganzheitliche Leistung“-Ansatzes. Zudem werden Hinweise der Kontrollfunktionen und Diszipli- narmaßnahmen sowie deren Einfluss auf die VV einbezogen
    Bei per Ermessensentscheidung erfolgenden Sub-Pool-Zuteilungen verwendet das DWS DCC die internen (finanziellen und nichtfinanziellen) Balanced Scorecard-Kennzahlen zur Erstellung differenzierter und leistungsbezogener VV-Pools,
    Vergütung für das Jahr 2018
    Nach der hervorragenden Entwicklung im Jahr 2017 hatte die globale Vermögensverwaltungsbranche 2018 mit einigen Schwierigkeiten zu kämpfen. Gründe waren ungünstige Marktbedin- gungen, stärkere geopolitische Spannungen und die negative Stimmung unter den Anlegern, vor allem am europäischen Retail-Miarkt. Auch die DWS Gruppe blieb von dieser Entwicklung nicht verschont.
    Vor diesem Hintergrund hat das DCC die Tragfähigkeit der VV für das Jahr 2018 kontrolliert und festgestellt, dass die Kapital- und Liquiditätsausstattung der DWS Gruppe unter Berücksichti- ‚gung des Ergebnisses vor und nach Steuern klar über den regulatorisch vorgeschriebenen Mindestanforderungen und dem internen Schwellenwert für die Risikotoleranz liegt.
    Als Teil der im März 2019 für das Performance-Jahr 2018 gewährten VV wurde die Gruppenkomponente allen berechtigten Mitarbeitern auf Basis der Bewertung der vier festgelegten Leistungs- kennzahlen gewährt. Der Vorstand der Deutsche Bank AG hat für 2018 unter Berücksichtigung der beträchtlichen Leistungen der Mitarbeiter und in seinem Ermessen einen Zielerreichungsgrad von 70 % festgelegt
    Identifi ierung von Risikoträgern
    Gemäß Gesetz vom 17. Dezember 2010 über die Organismen für gemeinsame Anlagen (in seiner jeweils gültigen Fassung) sowie den ESMA-Leitlinien unter Berücksichtigung der OGAW- Richtlinie hat die Gesellschaft Mitarbeiter mit wesentlichem Einfluss auf das Risikoprofil der Gesellschaft ermittelt („Risikoträger"). Das Identifizierungsverfahren basiert auf der Bewertung des Einflusses folgender Kategorien von Mitarbeitern auf das Risikoprofil der Gesellschaft oder einen von ihr verwalteten Fonds: (a) Geschäftsführung/Senior Management, (b) Portfolio-/ Investmentmanager, (c) Kontrollfunktionen, (d) Mitarbeiter mit Leitungsfunktionen in Verwaltung, Marketing und Human Resources, (e) sonstige Mitarbeiter (Risikoträger) mit wesentlichem Einfluss, (f} sonstige Mitarbeiter in der gleichen Vergütungsstufe wie sonstige Risikoträger. Mindestens 40 % der VV für Risikoträger werden aufgeschoben vergeben. Des Weiteren werden für wichtige Anlageexperten mindestens 50 % sowohl des direkt ausgezahlten als auch des aufgeschobenen Teils in Form von aktienbasierten oder fondsbasierten Instrumenten der DWS Gruppe gewährt. Alle aufgeschobenen Komponenten sind bestimmten Leistungs- und Verfallbedingungen unterworfen, um eine angemessene nachträgliche Risikoadjustierung zu gewähr- leisten. Bei einem VV-Betrag von weniger als EUR 50.000 erhalten Risikoträger ihre gesamte \VV in bar und ohne Aufschub.
    Zusammenfassung der Informationen zur Vergütung für die Gesellschaft für 2018 '
    \ Vergütungsdaten für Delegierte, die die Gesellschaft Portfolio- oder Risikomanagementaufgaben übertragen hat, sind nicht der Tabelle erfasst. an in Unter Berücksichtigung diverser Vergütungsbestandteile entsprechend den Definitionen in den ESMA-Leitlinien, die Geldzahlungen oder leistungen (wie Bargeld, Anteile, Optionsscheine, Rentenbeiträge) oder Nicht-(direkte) Geldleistungen (wie Gehaltsnebenleistungen oder Sondervergütungen für Fahrzeuge, Mobiltelefone, usw.) umfassen 3 „Senior Management” umfasst nur den Vorstand der Gesellschaft. Der Vorstand erfüllt die Definition als Führungskräfte der Gesellschaft. Uber den Vorstand hinaus wurden keine weiteren Führungskräfte identifiziert.


Tables are stored in ``page.tables`` which is a python list of table
objects. Obviously, only one table has been detected.

.. code:: ipython3

    len(page.tables)




.. parsed-literal::

    1



.. code:: ipython3

    page.tables[0].text




.. parsed-literal::

    ' Jahresdurchschnitt der Mitarbeiterzahl 139\n Gesamtvergütung ? EUR 15.315.952\n Fixe Vergütung EUR 13.151.856\n Variable Vergütung EUR 2.164.096\n davon: Carried Interest EURO\n Gesamtvergütung für Senior Management ® EUR 1.468.434\n Gesamtvergütung für sonstige Risikoträger EUR 324.229\n Gesamtvergütung für Mitarbeiter mit Kontrollfunktionen EUR 554.046\n'



In addition, an HTML version is generated that visually reproduces the
recognized structure.

.. code:: ipython3

    HTML(page.tables[0].html)




.. raw:: html

    <table><tr><td>Jahresdurchschnitt der Mitarbeiterzahl</td><td>139</td></tr><tr><td>Gesamtvergütung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe Vergütung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergütung</td><td>EUR 2.164.096</td></tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergütung für Senior Management ®</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergütung für sonstige Risikoträger</td><td>EUR 324.229</td></tr><tr><td>Gesamtvergütung für Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>



Finally, you can save the full results to a JSON file.

.. code:: ipython3

    page.save(image_path)

How to continue
===============

In this notebook we have shown how to use the built-in analyzer for text
extraction from image/PDF documents.

We recommend that the next step is to explore the notebook
**Custom_Pipeline**. Here we go into more detail about the composition
of pipelines and explain with an example how you can build a pipeline
yourself.
