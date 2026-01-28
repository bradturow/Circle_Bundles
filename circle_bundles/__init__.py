# circle_bundles/__init__.py
from __future__ import annotations

"""
circle_bundles: tools for detecting, visualizing, and classifying circle-bundle structure in data.

Recommended usage:
    import circle_bundles as cb

Public API:
    All curated user-facing symbols are re-exported from :mod:`circle_bundles.api`.
    For subpackages (synthetic, optical_flow, viz, etc.), import from those modules directly.
"""

# Optional version string
try:
    from ._version import __version__  # type: ignore
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

# Curated public API re-export
from .api import *  # noqa: F401,F403
from .api import __all__ as _api_all

__all__ = ["__version__", *_api_all]
