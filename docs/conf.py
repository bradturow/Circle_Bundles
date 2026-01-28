# docs/conf.py
from __future__ import annotations

import os
import sys
from datetime import date

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
    "sphinx.ext.napoleon",       # Google/Numpy docstrings
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**/.DS_Store"]

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

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["navbar-icon-links"],
    "show_nav_level": 2,
}

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# Autodoc defaults 
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "exclude-members": "__init__",
}
