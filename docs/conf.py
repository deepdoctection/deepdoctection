# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import mock


# Mock the following modules for building the docs in rtd
# Tensorflow, Tensorpack and everything that is related to
mod = sys.modules['tensorflow'] = mock.Mock(name='tensorflow')
mod.__version__ = mod.VERSION = '2.4.1'
mod.__spec__ = mock.Mock(name='tensorflow')

MOCK_MODULES = ['h5py','lmdb','tensorflow.python.training.monitored_session','tensorflow.python.training',
                'tensorflow.python.client','tensorflow.python.framework','tensorflow.python.platform',
                'tensorflow.python.tools','tensorflow.contrib.graph_editor']


# Pytorch
MOCK_MODULES.extend(['torch','torch.cuda'])

# Detectron2
MOCK_MODULES.extend(['detectron2',
                     'detectron2.structures',
                     'detectron2.layers',
                     'detectron2.config',
                     'detectron2.modeling',
                     'detectron2.checkpoint'])

# Transformers
MOCK_MODULES.extend(['transformers'])


for mod_name in MOCK_MODULES:
    mod = sys.modules[mod_name] = mock.Mock(name=mod_name)
    mod.__spec__ = mock.Mock(name=mod_name)

ON_RTD = (os.environ.get('READTHEDOCS') == 'True')

# Todo: Replace that HACK so that sphinx can find the package
import dataflow.dataflow

if ON_RTD:
    sys.path.insert(0, os.path.abspath('../'))
else:
    sys.path.insert(0, os.path.abspath('../deep_doctection'))

ROOT = os.path.dirname(os.path.realpath(os.path.join(os.path.dirname(__file__))))

about = {}
with open(os.path.join(ROOT, "__about__.py")) as about_file:
    exec(about_file.read(), about)

import deep_doctection


# -- Project information -----------------------------------------------------

project = about["__title__"]
copyright = about["__copyright__"]
author = about["__author__"]


# The full version, including alpha/beta/rc tags

release = about["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc','sphinx.ext.viewcode','sphinx.ext.inheritance_diagram','sphinx.ext.graphviz',
              'recommonmark','sphinx.ext.autosectionlabel']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autodoc_typehints = 'description'
autodoc_inherit_docstrings = False

# Both the class’ and the __init__ method’s docstring are concatenated and inserted.
autoclass_content= 'both'