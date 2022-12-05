.. figure:: ./pics/dd_logo.png
   :alt: title

Getting started
===============

**deep**\ doctection is a package that can be used to extract text from
complex structured documents. These can be native PDFs but also scans.
In contrast to various text miners **deep**\ doctection makes use of
deep learning models from powerful third party libraries for solving
OCR, vision or language embedding problems.

This notebook will give you a quick tour so that you can get started
straight away.

.. code:: ipython3

    import cv2
    from pathlib import Path
    from matplotlib import pyplot as plt
    from IPython.core.display import HTML
    
    import deepdoctection as dd

Sample
------

Take an image (e.g. .png, .jpg, …). If you take the example below you’ll
maybe need to change ``image_path``.

.. code:: ipython3

    image_path = Path(dd.get_package_path()) / "notebooks/pics/samples/sample_2/sample_2.png"
    image = cv2.imread(image_path.as_posix())
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(image)

.. figure:: ./pics/sample_2.png
   :alt: title

   title

Analyzer
--------

Next, we instantiate the **deep**\ doctection analyzer. There is a
built-in pipeline you can use. The analyzer is an example of a pipeline
that can be built depending on the problem you want to tackle. This
particular pipeline is built from various building blocks as shown in
the diagram.

There is a lot going on under the hood. The analyzer calls three object
detectors to structure the page and an OCR engine to extract the text.
However, this is clearly not enough. On top of that, words have to be
mapped to layout structures and a reading order has to be inferred.

.. figure:: ./pics/dd_pipeline.png
   :alt: title

.. code:: ipython3

    analyzer = dd.get_dd_analyzer(language='deu')

The language of the sample is german and passing the argument
``language='deu'`` will use a Tesseract model that has been trained on a
german corpus giving much better result than the default english
version.

Analyze methods
---------------

Now, that once all models have been loaded, we can process single pages
or documents. You can either set ``path=path/to/dir`` if you have a
folder of scans or ``path=path/to/my/doc.pdf`` if you have a single pdf
document.

.. code:: ipython3

    path = Path(dd.get_package_path()) / "notebooks/pics/samples/sample_2"
    
    df = analyzer.analyze(path=path)
    df.reset_state()  # This method must be called just before starting the iteration. It is part of the API.


You can see when activating the cell that not much has happened yet. The
reason is that ``analyze`` is a generator function. We need a ``for``
loop or ``next`` to start the process.

.. code:: ipython3

    doc=iter(df)
    page = next(doc)

Page
----

Let’s see what we got back. We start with some header information about
the page. With ``get_attribute_names()`` you get a list of all
attributes.

.. code:: ipython3

    page.height, page.width, page.file_name, page.location




.. parsed-literal::

    (2339.0,
     1654.0,
     'sample_2.png',
     '/home/janis/Public/deepdoctection_pt/deepdoctection/notebooks/pics/samples/sample_2/sample_2.png')



.. code:: ipython3

    page.get_attribute_names()




.. parsed-literal::

    {<PageType.document_type>, <PageType.language>, 'layouts', 'tables', 'text'}



``page.document_type`` returns None. The reason is that this pipeline is
not built for document classification. You can easily build a pipeline
containing a document classifier, though. Check the docs for further
information.

.. code:: ipython3

    print(page.document_type)


.. parsed-literal::

    None


We can visualize the detected segments. If you set ``interactive=True``
a viewer will pop up. Use + and - to zoom out/in. Use q to close the
page.

Alternatively, you can visualize the output with matplotlib.

.. code:: ipython3

    image = page.viz()
    plt.figure(figsize = (25,17))
    plt.axis('off')
    plt.imshow(image)

.. figure:: ./pics/output_16_1.png
   :alt: title

Let’s have a look at other attributes. We can use the ``text`` property
to get the content of the document. You will notice that the table is
not included. You can therefore filter tables from the other content. In
fact you can even filter on every layout.

.. code:: ipython3

    print(page.text)


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


.. code:: ipython3

    for layout in page.layouts:
        if layout.category_name=="title":
            print(f"Title: {layout.text}")


.. parsed-literal::

    Title: Identifi ierung von Risikoträgern
    Title: Vergütung für das Jahr 2018
    Title: Festlegung der VV und angemessene Risikoadjustierung


Tables are stored in ``page.tables`` which is a python list of table
objects. Obviously, only one table has been detected. Let’s have a
closer look at the table. Most attributes are hopefully self explained.
If you ``print(page.tables)`` you will get a very cryptic ``__repr__``
output.

.. code:: ipython3

    len(page.tables)

.. parsed-literal::

    1


.. code:: ipython3

    table = page.tables[0]
    table.get_attribute_names()


.. parsed-literal::

    {'bbox',
     'cells',
     'columns',
     <TableType.html>,
     <TableType.item>,
     <TableType.max_col_span>,
     <TableType.max_row_span>,
     <TableType.number_of_columns>,
     <TableType.number_of_rows>,
     <Relationships.reading_order>,
     'rows',
     'text',
     'words'}



.. code:: ipython3

    table.number_of_rows, table.number_of_columns


.. parsed-literal::

    (8, 2)


.. code:: ipython3

    HTML(table.html)




.. raw:: html

    <table><tr><td>Jahresdurchschnitt der Mitarbeiterzahl</td><td>139</td></tr><tr><td>Gesamtvergütung ?</td><td>EUR 15.315.952</td></tr><tr><td>Fixe Vergütung</td><td>EUR 13.151.856</td></tr><tr><td>Variable Vergütung</td><td>EUR 2.164.096</td></tr><tr><td>davon: Carried Interest</td><td>EURO</td></tr><tr><td>Gesamtvergütung für Senior Management ®</td><td>EUR 1.468.434</td></tr><tr><td>Gesamtvergütung für sonstige Risikoträger</td><td>EUR 324.229</td></tr><tr><td>Gesamtvergütung für Mitarbeiter mit Kontrollfunktionen</td><td>EUR 554.046</td></tr></table>



Let’s go deeper into the rabbit hole. A ``Table`` has cells and we can
even get the text of one particular cell. Note that the output list is
not sorted by row or column. You have to do it yourself.

.. code:: ipython3

    cell = table.cells[0]
    cell.get_attribute_names()




.. parsed-literal::

    {'bbox',
     <CellType.body>,
     <CellType.column_number>,
     <CellType.column_span>,
     <CellType.header>,
     <Relationships.reading_order>,
     <CellType.row_number>,
     <CellType.row_span>,
     'text',
     'words'}



.. code:: ipython3

    cell.column_number, cell.row_number, cell.text, cell.annotation_id  # every object comes with a unique annotation_id




.. parsed-literal::

    (1,
     8,
     'Gesamtvergütung für Mitarbeiter mit Kontrollfunktionen',
     'afb3c667-5d58-3689-a82b-69a8a5f71cbd')



Still not down yet, we have a list of words that is responsible to
generate the text string.

.. code:: ipython3

    word = cell.words[0]
    word.get_attribute_names()




.. parsed-literal::

    {'bbox',
     <WordType.block>,
     <WordType.characters>,
     <Relationships.reading_order>,
     <WordType.tag>,
     <WordType.text_line>,
     <WordType.token_class>,
     <WordType.token_tag>}



The reading order determines the string position. OCR engines generally
provide a some heuristics to infer a reading order. This library,
however, follows the apporach to disentangle every processing step.

.. code:: ipython3

    word.characters, word.reading_order, word.token_class




.. parsed-literal::

    ('Gesamtvergütung', 1, None)



The ``Page`` object is read-only and even though you can change the
value it will not be persited.

.. code:: ipython3

    word.token_class = "ORG"

.. code:: ipython3

    word #  __repr__ of the base object does carry <WordType.token_class> information.  




.. parsed-literal::

    Word(active=True, _annotation_id='f35f5c53-8af3-3ed9-971a-4cd65c0a37ce', category_name=<LayoutType.word>, _category_name=<LayoutType.word>, category_id='1', score=0.91, sub_categories={<WordType.characters>: ContainerAnnotation(active=True, _annotation_id='fa28e8c0-5883-392f-b23b-92adb8537b8a', category_name=<WordType.characters>, _category_name=<WordType.characters>, category_id='None', score=0.91, sub_categories={}, relationships={}, value='Gesamtvergütung'), <WordType.block>: CategoryAnnotation(active=True, _annotation_id='8a40178f-1dff-3a02-81be-2b5f5b6d6250', category_name=<WordType.block>, _category_name=<WordType.block>, category_id='47', score=None, sub_categories={}, relationships={}), <WordType.text_line>: CategoryAnnotation(active=True, _annotation_id='34bd3cdf-0048-3647-af75-b43532688418', category_name=<WordType.text_line>, _category_name=<WordType.text_line>, category_id='1', score=None, sub_categories={}, relationships={}), <Relationships.reading_order>: CategoryAnnotation(active=True, _annotation_id='a266ac1d-2a35-3321-9f25-f5d05adef331', category_name=<Relationships.reading_order>, _category_name=<Relationships.reading_order>, category_id='1', score=None, sub_categories={}, relationships={})}, relationships={}, bounding_box=BoundingBox(absolute_coords=True, ulx=146, uly=1481, lrx=277, lry=1496, height=15, width=131))



You can save your result in a big ``.json`` file. The default ``save``
configuration will store the image as b64 string, so be aware: The
``.json`` file with that image has a size of 6,2 MB!

.. code:: ipython3

    page.save()

Having saved the results you can easily parse the file into the ``Page``
format.

.. code:: ipython3

    path = Path(dd.get_package_path()) / "notebooks/pics/samples/sample_2/sample_2.json"
    
    df = dd.SerializerJsonlines.load(path)
    page = dd.Page.from_dict(**next(iter(df)))
