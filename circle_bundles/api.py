from __future__ import annotations

"""
Public API for circle_bundles (curated).

Guiding principles
------------------
- Keep the public surface small and stable (JOSS-friendly).
- Expose theorem-level artifacts: bundle construction, bundle map, characteristic classes,
  and persistence.
- Avoid importing heavy/optional visualization dependencies at import time.
- Everything else remains available via submodules (e.g. circle_bundles.viz.*, circle_bundles.synthetic.*).
"""

# ----------------------------
# Core bundle construction / results
# ----------------------------
from .bundle import (
    BundleResult,
    BundleMapResult,
    GlobalTrivializationResult,
    MaxTrivialSubcomplex,
    build_bundle,
)

# ----------------------------
# Bundle map (theorem-level artifact)
# ----------------------------
from .trivializations.bundle_map import (
    BundleMapReport,
    ChartDisagreementStats,
    FrameDataset,
    FrameReducerConfig,
    FrameReductionReport,
    TrueFramesResult,
    build_bundle_map,
    get_bundle_map,
    show_bundle_map_summary,
)

from .reduction.frame_reduction import (
    reduction_curve_psc,
    reduction_curve_subspace_pca,
)

from .trivializations.gauge_canon import (
    GaugeCanonConfig,
)

# ----------------------------
# Characteristic classes
# ----------------------------
from .characteristic_class import (
    ClassResult,
    compute_classes,
    compute_twisted_euler_class,
)

# ----------------------------
# Characteristic class persistence
# ----------------------------
from .analysis.class_persistence import (
    CobirthResult,
    CodeathResult,
    PersistenceResult,
    compute_bundle_persistence,
    summarize_edge_driven_persistence,
    build_edge_weights_from_transition_report,
)


# ----------------------------
# Public __all__
# ----------------------------
__all__ = [
    # bundle
    "BundleResult",
    "BundleMapResult",
    "GlobalTrivializationResult",
    "MaxTrivialSubcomplex",
    "build_bundle",

    # bundle map
    "BundleMapReport",
    "ChartDisagreementStats",
    "FrameDataset",
    "FrameReducerConfig",
    "FrameReductionReport",
    "TrueFramesResult",
    "build_bundle_map",
    "get_bundle_map",
    "show_bundle_map_summary",
    "reduction_curve_psc",
    "reduction_curve_subspace_pca",
    "GaugeCanonConfig",

    # classes
    "ClassResult",
    "compute_classes",
    "compute_twisted_euler_class",

    # persistence
    "CobirthResult",
    "CodeathResult",
    "PersistenceResult",
    "compute_bundle_persistence",
    "summarize_edge_driven_persistence",
    "build_edge_weights_from_transition_report",
]
