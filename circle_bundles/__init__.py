# circle_bundles/__init__.py
from __future__ import annotations

"""
circle_bundles: tools for detecting, visualizing, and classifying circle-bundle structure in data.

Recommended usage:
    import circle_bundles as cb

Public API:
    - Curated user-facing symbols are re-exported from :mod:`circle_bundles.api`.
    - Subpackages are available as namespaces (``cb.synthetic``, ``cb.viz``, ``cb.optical_flow``)
      and are imported lazily to avoid pulling in optional dependencies.
"""

import importlib
from typing import Any

# ------------------------------------------------------------
# Version
# ------------------------------------------------------------
try:
    from ._version import __version__  # type: ignore
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

# ------------------------------------------------------------
# Curated public API re-export (single source of truth)
# ------------------------------------------------------------
from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all

# ------------------------------------------------------------
# Lazy subpackage namespaces
# ------------------------------------------------------------
_SUBPACKAGES = ("synthetic", "viz", "optical_flow")

# IMPORTANT:
# We intentionally do NOT do convenience passthrough from subpackages
# (e.g. cb.sample_opt_flow_torus), because it makes the top-level namespace
# unstable and can break Sphinx autosummary/autodoc when optional deps are missing.
#
# If you want passthrough later, do it explicitly in api.py (curated + stable).

__all__ = ["__version__", *_api_all, *_SUBPACKAGES]


def __getattr__(name: str) -> Any:
    # Lazy-load subpackages as namespaces: cb.synthetic, cb.viz, cb.optical_flow
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")

    # Otherwise, rely only on curated API symbols imported above.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Only expose stable/curated names + the subpackage namespaces
    names = set(__all__)
    return sorted(names)
