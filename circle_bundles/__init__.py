# circle_bundles/__init__.py
from __future__ import annotations

"""
circle_bundles package.

Recommended usage:
    import circle_bundles as cb

This re-exports the curated public API defined in circle_bundles.api.
"""

from .api import *  # noqa: F401,F403
from .api import __all__  # noqa: F401
