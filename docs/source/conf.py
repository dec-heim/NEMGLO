# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))
sys.path.insert(0, os.path.abspath('../../examples/'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NEMGLO'
copyright = '2022, Declan Heim'
author = 'Declan Heim'
release = '0.3.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx_copybutton",
    "sphinx_external_toc",
    "sphinx_design",
#    "sphinx.ext.viewcode",
#    "myst_parser",
    "myst_nb",
]

autosectionlabel_prefix_document = True
templates_path = ['_templates']
exclude_patterns = []

todo_include_todos = True

# --  Napoleon options--------------------------------------------------------
napoleon_use_param = True
napoleon_custom_sections = ['Multiple Returns']

# -- Formats for MyST --------------------------------------------------------
source_suffix = [".rst", ".md"]
nb_custom_formats = {
    ".md": ["jupytext.reads", {"fmt": "mystnb"}],
}
# -- Autodoc options ---------------------------------------------------------

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "both"

# Only insert class docstring
autoclass_content = "class"

autodoc_mock_imports = ["pandas", "numpy", "datetime", "mip"]
autodoc_member_order = 'bysource'

# --  Intersphinx options-----------------------------------------------------
intersphinx_mapping = {"python": ("https://docs.python.org/3", None),
                       "pandas": ("https://pandas.pydata.org/docs/", None),}

# --  MyST options------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

nb_execution_mode = 'off'

# -- External TOC------------------------------------------------------------
external_toc_path = "_toc.yml"  # optional, default: _toc.yml
external_toc_exclude_missing = False  # optional, default: False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_show_sourcelink = False
html_theme_options = {
    "repository_url": "https://github.com/dec-heim/NEMGLO",
    "use_repository_button": True,
    "use_issues_button": True,
    }
