# circle_bundles/api.py
"""
Public API for circle_bundles.

This module re-exports the *stable* user-facing surface of the package.
The objects listed in __all__ are considered part of the supported API.
"""

from __future__ import annotations

# =============================================================================
# Core workflow (recommended entry points)
# =============================================================================

# =============================================================================
# Core workflow (recommended entry points)
# =============================================================================

from .bundle2 import (
    Bundle,
    LocalTrivsResult,
    ClassesAndPersistence,
    BundleMapResult,
)


# =============================================================================
# Result + config types
# =============================================================================

from .trivializations.local_triv import LocalTrivResult, DreimacCCConfig
from .o2_cocycle import O2Cocycle
from .analysis.quality import BundleQualityReport
from .analysis.class_persistence import CobirthResult, CodeathResult, PersistenceResult
from .characteristic_class import ClassResult
from .trivializations.bundle_map import FramePacking
from .summaries.nerve_summary import NerveSummary
from .reduction.frame_reduction import FrameReducerConfig

# =============================================================================
# Covers (construction utilities)
# =============================================================================

from .covers.covers import CoverData, get_metric_ball_cover
from .covers.fibonacci_covers import get_s2_fibonacci_cover, get_rp2_fibonacci_cover

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
    ProductMetricConcat,
)

# =============================================================================
# Fiberwise analysis and clustering
# =============================================================================

from .analysis.fiberwise_clustering import (
    fiberwise_clustering,
)

from .analysis.local_analysis import (
    get_local_pca,
    get_local_rips,
    plot_local_rips,
)

from .geometry.geometric_unwrapping import get_cocycle_dict, lift_base_points

# ==========================
# Additional helpers (optional)
# ==========================

_syn_all: list[str] = []
_viz_all: list[str] = []
_of_all: list[str] = []

try:
    from .synthetic import *  # noqa: F401,F403
    from .synthetic import __all__ as _syn_all  # type: ignore
except Exception:
    pass

try:
    from .viz import *  # noqa: F401,F403
    from .viz import __all__ as _viz_all  # type: ignore
except Exception:
    pass

try:
    from .optical_flow import *  # noqa: F401,F403
    from .optical_flow import __all__ as _of_all  # type: ignore
except Exception:
    pass



from typing import Optional
import numpy as np

def sample_sphere(n: int, *, dim: int = 2, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Sample points uniformly from the unit sphere S^dim embedded in R^{dim+1}.
    Lazy-imported from circle_bundles.synthetic.
    """
    from .synthetic import sample_sphere as _sample_sphere
    return _sample_sphere(n, dim=dim, rng=rng)

def hopf_projection(x: np.ndarray) -> np.ndarray:
    """
    Hopf projection S^3 -> S^2 (expects points in R^4).
    Lazy-imported from circle_bundles.synthetic (or wherever your implementation lives).
    """
    from .synthetic import hopf_projection as _hopf_projection
    return _hopf_projection(x)



# =============================================================================
# Public export list
# =============================================================================

__all__ = [
    # Core workflow
    "Bundle",
    "LocalTrivsResult",
    "ClassesAndPersistence",
    "BundleMapResult",

    # Result + config types
    "LocalTrivResult",
    "DreimacCCConfig",
    "O2Cocycle",
    "BundleQualityReport",
    "CobirthResult",
    "CodeathResult",
    "PersistenceResult",
    "ClassResult",
    "FramePacking",
    "NerveSummary",
    "FrameReducerConfig",    

    # Covers
    "CoverData",
    "get_metric_ball_cover",
    "get_s2_fibonacci_cover",
    "get_rp2_fibonacci_cover",

    # Metrics
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
    "ProductMetricConcat",
    
    # Advanced utilities
    "fiberwise_clustering",
    "get_local_pca",
    "get_local_rips",
    "plot_local_rips",
    "get_cocycle_dict",
    "lift_base_points",

    "sample_sphere",
    "hopf_projection",
    
]

__all__ += [*_syn_all, *_viz_all, *_of_all]
