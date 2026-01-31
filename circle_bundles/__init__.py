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
    - Convenience passthrough: if an attribute is not found at top-level, we try to resolve it
      from ``synthetic``, then ``viz``, then ``optical_flow``.
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
# Curated public API re-export
# ------------------------------------------------------------
try:
    from .api import *  # noqa: F401,F403
    from .api import __all__ as _api_all
except Exception:  # pragma: no cover
    _api_all = []  # type: ignore[assignment]

# ------------------------------------------------------------
# Subpackages exposed as cb.synthetic / cb.viz / cb.optical_flow
# ------------------------------------------------------------
_SUBPACKAGES = ("synthetic", "viz", "optical_flow")

# Where to look for convenience passthrough attributes.
# Ordering matters: earlier modules win when names collide.
_PASSTHROUGH_MODULES = ("synthetic", "viz", "optical_flow")

__all__ = ["__version__", *_api_all, *_SUBPACKAGES]


def __getattr__(name: str) -> Any:
    # 1) Lazy-load subpackages as namespaces
    if name in _SUBPACKAGES:
        return importlib.import_module(f"{__name__}.{name}")

    # 2) Convenience passthrough: cb.<x> -> circle_bundles.<module>.<x>
    for mod in _PASSTHROUGH_MODULES:
        try:
            m = importlib.import_module(f"{__name__}.{mod}")
        except Exception:
            # If optional deps are missing inside a subpackage, just skip it.
            continue
        if hasattr(m, name):
            return getattr(m, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Include top-level names + subpackages + passthrough names (best-effort).
    names = set(globals().keys())
    names.update(_SUBPACKAGES)

    # Best-effort: add exports from subpackages so tab-completion is nice.
    for mod in _PASSTHROUGH_MODULES:
        try:
            m = importlib.import_module(f"{__name__}.{mod}")
            names.update(getattr(m, "__all__", []))
        except Exception:
            pass

    return sorted(names)
