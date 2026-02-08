# circle_bundles/covers.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..metrics import EuclideanMetric, as_metric


@dataclass(frozen=True)
class CoverData:
    """Minimal cover payload for Bundle construction."""
    U: np.ndarray
    pou: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None


def get_metric_ball_cover(
    base_points: np.ndarray,
    landmarks: np.ndarray,
    *,
    radius: float,
    metric: Any = None,
) -> CoverData:
    """
    Metric-ball cover constructor.

    Returns
    -------
    CoverData with:
      - U: (n_sets, n_samples) bool membership
      - pou: (n_sets, n_samples) float partition of unity (linear hat)
      - landmarks: (n_sets, d) float
    """
    X = np.asarray(base_points, dtype=float)
    L = np.asarray(landmarks, dtype=float)

    if X.ndim != 2:
        raise ValueError(f"base_points must be (n_samples, d). Got {X.shape}.")
    if L.ndim != 2:
        raise ValueError(f"landmarks must be (n_sets, d). Got {L.shape}.")
    if X.shape[1] != L.shape[1]:
        raise ValueError(f"Dim mismatch: base_points d={X.shape[1]} vs landmarks d={L.shape[1]}.")

    M = EuclideanMetric() if metric is None else as_metric(metric)
    dist = M.pairwise(L, X)  # (n_sets, n_samples)

    U = dist < float(radius)

    # linear hat POU
    w = np.maximum(0.0, 1.0 - dist / float(radius))
    w *= U
    denom = w.sum(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    pou = w / denom

    return CoverData(
        U=U,
        pou=pou,
        landmarks=L,
        meta={"type": "metric_ball", "radius": float(radius), "metric": type(M).__name__},
    )
