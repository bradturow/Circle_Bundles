# docs/conf.py
from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

# -- Path setup --------------------------------------------------------------
# Make the package importable for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "circle_bundles"
author = "Brad Turow"
copyright = f"{date.today().year}, {author}"

# Prefer the package's __version__ when available
try:
    import circle_bundles  # noqa: F401

    release = getattr(circle_bundles, "__version__", "0+unknown")
except Exception:
    release = "0+unknown"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/Numpy docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    # Notebook support + notebook-native gallery
    "myst_nb",
    "myst_sphinx_gallery",
    "sphinxcontrib.bibtex", 
]

bibtex_bibfiles = ["references.bib"]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "**/.DS_Store",
    "**/.ipynb_checkpoints",
    "**/__pycache__",
]

# Autosummary: generate stub pages for autosummary directives
autosummary_generate = True

# Napoleon settings (good defaults for scientific Python projects)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc defaults (keeps pages readable)
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "exclude-members": "__init__",
}

# If you have optional deps (dash/plotly/etc), don't hard fail doc builds
autodoc_mock_imports = [
    "dash",
    "plotly",
    "sklearn",
    "scipy",
    "gudhi",
    "ripser",
    "dreimac",
]

# -- MyST-NB settings --------------------------------------------------------
# Treat notebooks/markdown as sources (useful if you also include .ipynb pages directly in toctrees)
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst": "myst-nb",
    ".ipynb": "myst-nb",
}

# Recommended: don't execute notebooks during doc builds (faster + stable).
# The saved outputs in your notebooks will be rendered.
nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# Read the Docs theme (Dreimac-style)
html_theme = "sphinx_rtd_theme"

# RTD theme options (nice defaults)
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
}

# Static files 
html_static_path = ["_static"]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- MyST Sphinx Gallery -----------------------------------------------------
from myst_sphinx_gallery import GalleryConfig  # noqa: E402

myst_sphinx_gallery_config = GalleryConfig(
    root_dir=Path(__file__).resolve().parent,
    examples_dirs=["../notebooks/tutorials"],
    gallery_dirs=["tutorials/auto_examples"],
    notebook_thumbnail_strategy="code",
)



myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]
