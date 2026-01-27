# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add repo root so `import circle_bundles` works
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "circle_bundles"
author = "Brad Turow, Jose A. Perea"
copyright = "2026, Brad Turow, Jose A. Perea"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# Autosummary / autodoc behavior
autosummary_generate = True
autodoc_member_order = "bysource"

# Napoleon (Google / NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
