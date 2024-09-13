# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.abspath("../../python"))


# -- Project information -----------------------------------------------------

project = "stempy"
copyright = "2020, Kitware, INC"
author = "Kitware, INC"

# The full version, including alpha/beta/rc tags
release = "1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "sphinx_rtd_theme", "recommonmark"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# Unfortunately, if we have a class that inherits sphinx's mock
# object, and we use that class in a default argument, we run into
# attribute errors. Fix it by defining an explicit mock of the class.
class MockReader:
    class H5Format:
        Frame = 1


sys.modules["stempy._io"] = mock.Mock()
_io_mock = sys.modules["stempy._io"]

# We have to override these so we get don't get conflicting metaclasses
_io_mock._sector_reader = MockReader
_io_mock._threaded_reader = object
_io_mock._reader = object
_io_mock._pyreader = object
_io_mock._threaded_multi_pass_reader = object

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Get autodoc to mock these imports, because we don't actually need them
# for generating the docs
autodoc_mock_imports = ["stempy._image", "numpy", "h5py"]


# Modify this function to customize which classes/functions are skipped
def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    # Exclude any names that start with a string in this list
    exclude_startswith_list = ["_"]

    if any(name.startswith(x) for x in exclude_startswith_list):
        return True

    # Exclude any names that match a string in this list
    exclude_names = [
        "ReaderMixin",
        "PyReader",
        "SectorReader",
        "get_hdf5_reader",
        "SectorThreadedMultiPassReader",
    ]

    return name in exclude_names


# Automatically called by sphinx at startup
def setup(app):
    # Connect the autodoc-skip-member event from apidoc to the callback
    # This allows us to customize which classes/functions are skipped
    app.connect("autodoc-skip-member", autodoc_skip_member_handler)
