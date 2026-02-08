from __future__ import annotations

"""
Public API for circle_bundles (curated).

Design
------
The primary entry point is :func:`build_bundle`, which returns a :class:`BundleResult`.
Downstream computations are exposed as cached methods on :class:`BundleResult`
(e.g. ``get_persistence()``, ``get_global_trivialization()``, ``get_bundle_map()``,
``get_pullback_data()``).

This module also exposes common *on-ramps* (covers + metrics) so typical users can
start workflows without deep imports.

Everything else remains accessible via submodules (covers, metrics, viz, synthetic, optical_flow).
"""

# =====
# New
# =====
from .bundle2 import Bundle
from .covers.covers import (
    CoverData,
    get_metric_ball_cover,
)

from .covers.fibonacci_covers import (
    get_s2_fibonacci_cover,
    get_rp2_fibonacci_cover,
)

# =============================================================================
# Core bundle construction + main result objects
# =============================================================================
from .bundle import (
    BundleResult,
    BundleMapResult,
    PullbackTotalSpaceResult,
    GlobalTrivializationResult,
    MaxTrivialSubcomplex,
    build_bundle,
)

# Additional dataclasses
from .trivializations.local_triv import LocalTrivResult, DreimacCCConfig
from .characteristic_class import ClassResult
from .o2_cocycle import TransitionReport, O2Cocycle
from .analysis.class_persistence import (
    CobirthResult,
    CodeathResult,
    PersistenceResult
)


# =============================================================================
# Covers 
# =============================================================================
from .base_covers import (
    MetricBallCover,
    TriangulationStarCover,
)

from .covers.metric_ball_cover_builders import (
    S2GeodesicMetric,
    RP2GeodesicMetric,
)

from .covers.triangle_cover_builders_fibonacci import (
    make_s2_fibonacci_star_cover,
    make_rp2_fibonacci_star_cover,
)

# =============================================================================
# Metrics 
# =============================================================================
from .metrics import (
    EuclideanMetric,
    S1AngleMetric,
    RP1AngleMetric,
    S1UnitVectorMetric,
    RP1UnitVectorMetric,
    RP2UnitVectorMetric,
    T2FlatMetric,
    Torus_DiagQuotientMetric_R4,
    Torus_KleinQuotientMetric_R4,    
    RP2_TrivialMetric,
    RP2_TwistMetric,
    RP2_FlipMetric,
    S3QuotientMetric,
)


# ----------------------------
# Fiberwise analysis utilities
# ----------------------------
from .analysis.fiberwise_clustering import (
    fiberwise_clustering,
    get_cluster_persistence,
    get_filtered_cluster_graph,
    plot_fiberwise_pca_grid,
    plot_fiberwise_summary_bars,
)

from .analysis.local_analysis import (
    get_local_pca,
    plot_local_pca,
    get_local_rips,
    plot_local_rips,
)

from .geometry.geometric_unwrapping import (
    get_cocycle_dict,
    lift_base_points,
)


__all__ = [
    # new
    "Bundle",
    "CoverData",
    "get_metric_ball_cover",
    "get_s2_fibonacci_cover",
    "get_rp2_fibonacci_cover",
    
    # core
    "build_bundle",
    "BundleResult",
    "BundleMapResult",
    "PullbackTotalSpaceResult",
    "GlobalTrivializationResult",
    "MaxTrivialSubcomplex",
    "DreimacCCConfig",
    "LocalTrivResult",
    "ClassResult",
    "TransitionReport",
    "O2Cocycle",
    "CobirthResult",
    "CodeathResult",
    "PersistenceResult",

    
    # covers
    "MetricBallCover",
    "TriangulationStarCover",
    "S2GeodesicMetric",
    "RP2GeodesicMetric",
    "make_s2_fibonacci_star_cover",
    "make_rp2_fibonacci_star_cover",

    # metrics
    "EuclideanMetric",
    "S1AngleMetric",
    "RP1AngleMetric",
    "S1UnitVectorMetric",
    "RP1UnitVectorMetric",
    "RP2UnitVectorMetric",
    "T2FlatMetric",
    "Torus_DiagQuotientMetric_R4",
    "Torus_KleinQuotientMetric_R4",
    "RP2_TrivialMetric",
    "RP2_TwistMetric",
    "RP2_FlipMetric",
    "S3QuotientMetric",

    # fiberwise analysis utilities
    "fiberwise_clustering",
    "get_cluster_persistence",
    "get_filtered_cluster_graph",
    "plot_fiberwise_pca_grid",
    "plot_fiberwise_summary_bars",
    "get_local_pca",
    "plot_local_pca",
    "get_local_rips",
    "plot_local_rips",
    "get_cocycle_dict",
    "lift_base_points",   
]


