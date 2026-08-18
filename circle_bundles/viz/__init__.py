"""
Visualization utilities for circle_bundles.

Notes
-----
Some visualizations depend on optional third-party libraries (plotly, dash, scikit-learn).
Those imports are guarded so that importing this module does not fail if optional deps
are missing.
"""

from __future__ import annotations

# Keep submodules importable as namespaces
from . import (
    angles,
    base_vis,
    circle_vis,
    fiber_vis,
    fiberwise_clustering_vis,
    gudhi_graph_utils,
    lattice_vis,
    nerve_circle,
    nerve_vis,
    pca_vis,
    thumb_grids,
    image_utils,
)

# ----------------------------
# Always-available
# ----------------------------
from .angles import (
    compare_trivs,
)

from .base_vis import (
    base_vis as base_vis,
)

from .fiber_vis import (
    fiber_vis,
)

from .lattice_vis import (
    lattice_vis,
    scatter_lattice_vis,
)

from .nerve_vis import (
    nerve_vis,
)

from .pca_vis import (
    show_pca,
)

from .thumb_grids import (
    show_data_vis,
)

# ----------------------------
# Interactive helpers (Plotly/Dash are imported when the helper is called)
# ----------------------------
from .bundle_dash import (
    show_bundle_vis,
)


__all__ = [
    # namespaces
    "angles",
    "base_vis",
    "circle_vis",
    "fiber_vis",
    "fiberwise_clustering_vis",
    "gudhi_graph_utils",
    "lattice_vis",
    "scatter_lattice_vis",
    "nerve_circle",
    "nerve_vis",
    "pca_vis",
    "thumb_grids",
    "image_utils",

    # always-available exports
    "compare_trivs",
    "base_vis",
    "fiber_vis",
    "lattice_vis",
    "nerve_vis",
    "show_pca",
    "show_data_vis",
    "show_bundle_vis",
]
