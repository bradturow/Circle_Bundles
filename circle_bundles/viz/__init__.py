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
    nerve_plotly,
    nerve_vis,
    pca_vis,
    thumb_grids,
    image_utils,
)

# ----------------------------
# Always-available 
# ----------------------------
from .angles import (
    compare_angle_pairs,
    compare_trivs,
)

from .base_vis import (
    base_vis as base_vis,
)

from .circle_vis import (
    circle_vis,
    circle_vis_grid,
)

from .fiber_vis import (
    fiber_vis,
)

from .fiberwise_clustering_vis import (
    make_patch_cluster_diagram,
    get_G_vertex_coords,
    plot_component_patch_diagram,
)

from .gudhi_graph_utils import (
    graph_to_st,
    create_st_dicts,
)

from .lattice_vis import (
    lattice_vis,
)

from .nerve_circle import (
    show_circle_nerve,
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
# Optional (plotly/dash/sklearn may be required)
# ----------------------------
_HAVE_PLOTLY_DASH = False
try:
    from .bundle_dash import (
        show_bundle_vis,
    )
    _HAVE_PLOTLY_DASH = True
except ImportError:
    # Optional dependency not installed
    show_bundle_vis = None  # type: ignore[assignment]


__all__ = [
    # namespaces
    "angles",
    "base_vis",
    "circle_vis",
    "fiber_vis",
    "fiberwise_clustering_vis",
    "gudhi_graph_utils",
    "lattice_vis",
    "nerve_circle",
    "nerve_plotly",
    "nerve_vis",
    "pca_vis",
    "thumb_grids",
    "image_utils",

    # always-available exports
    "compare_angle_pairs",
    "compare_trivs",
    "base_vis",
    "circle_vis",
    "circle_vis_grid",
    "fiber_vis",
    "make_patch_cluster_diagram",
    "get_G_vertex_coords",
    "plot_component_patch_diagram",
    "graph_to_st",
    "create_st_dicts",
    "lattice_vis",
    "show_circle_nerve",
    "nerve_vis",
    "show_pca",
    "show_data_vis",
]

if _HAVE_PLOTLY_DASH:
    __all__ += [
        "show_bundle_vis",
    ]
