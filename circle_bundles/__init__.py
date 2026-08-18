# circle_bundles/__init__.py
from __future__ import annotations

"""
circle_bundles: tools for detecting, visualizing, and classifying circle-bundle structure in data.

Recommended usage:
    import circle_bundles as cb

Public API:
    - Curated user-facing symbols are re-exported from :mod:`circle_bundles.api`.
    - Subpackages are also available as namespaces (``cb.synthetic``, ``cb.viz``,
      ``cb.optical_flow``).
    - Optional third-party integrations are imported only when their features are used.
"""

import importlib
from typing import Any

# ------------------------------------------------------------
# Version
# ------------------------------------------------------------
__version__ = "0.1.0"

# ------------------------------------------------------------
# Curated public API re-export (single source of truth)
# ------------------------------------------------------------
from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all

# ------------------------------------------------------------
# Subpackage namespaces
# ------------------------------------------------------------
_SUBPACKAGES = ("synthetic", "viz", "optical_flow")

# Top-level convenience exports are selected in api.py. Keeping that list stable
# prevents the caller's installed optional dependencies from changing cb.__all__.

__all__ = ["__version__", *_api_all, *_SUBPACKAGES]


def __getattr__(name: str) -> Any:
    # Resolve subpackages as namespaces if they have not already been imported by api.py.
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")

    # Otherwise, rely only on curated API symbols imported above.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Only expose stable/curated names + the subpackage namespaces
    names = set(__all__)
    return sorted(names)
