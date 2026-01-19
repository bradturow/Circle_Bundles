# circle_bundles/metric_ball_cover_builders.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import numpy as np

from ..base_covers import MetricBallCover


__all__ = [
    # landmarks / reps
    "fibonacci_sphere",
    "canonical_hemisphere_reps",
    # metrics
    "S2GeodesicMetric",
    "RP2GeodesicMetric",
    # cover builders
    "make_s2_metric_ball_cover",
    "make_rp2_metric_ball_cover",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_rows(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm


def fibonacci_sphere(n: int, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Nearly-uniform landmarks on S^2 via a spherical Fibonacci lattice.

    Returns
    -------
    L : (n,3) float
        Unit vectors on S^2.

    Notes
    -----
    - Not antipodally symmetric by default.
    - Deterministic given n; rng unused (kept for API symmetry / future jitter).
    """
    if n <= 0:
        raise ValueError(f"n must be positive. Got {n}.")

    i = np.arange(n, dtype=float)
    phi = (1.0 + 5.0**0.5) / 2.0  # golden ratio
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    L = np.stack([x, y, z], axis=1)
    return _normalize_rows(L)


def canonical_hemisphere_reps(X: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """
    Choose canonical representatives for RP^2 points given by unit vectors on S^2.

    Rule:
      - Prefer z > 0
      - If z == 0, prefer y > 0
      - If z == 0 and y == 0, prefer x > 0
    Then flip sign if needed so each vector lands in the canonical closed hemisphere.

    Parameters
    ----------
    X : (n,3)
        Vectors (need not be normalized).

    Returns
    -------
    R : (n,3)
        Canonical hemisphere reps, normalized.

    Notes
    -----
    This ensures a deterministic representative for each projective class [x] = {x, -x}.
    """
    X = _normalize_rows(np.asarray(X, dtype=float))
    x, y, z = X[:, 0], X[:, 1], X[:, 2]

    # Decide whether to flip: keep if already canonical, else multiply by -1.
    keep = (z > atol) | ((np.abs(z) <= atol) & (y > atol)) | (
        (np.abs(z) <= atol) & (np.abs(y) <= atol) & (x >= -atol)
    )

    R = X.copy()
    R[~keep] *= -1.0
    return _normalize_rows(R)


# -----------------------------------------------------------------------------
# Metrics (vectorized .pairwise API)
# -----------------------------------------------------------------------------

@dataclass
class S2GeodesicMetric:
    """
    Geodesic distance on S^2: d(x,y) = arccos(<x,y>), with x,y unit vectors.
    """
    eps: float = 1e-12

    def pairwise(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
        A = _normalize_rows(np.asarray(A, dtype=float), eps=self.eps)
        if B is None:
            B = A
        else:
            B = _normalize_rows(np.asarray(B, dtype=float), eps=self.eps)

        G = A @ B.T
        G = np.clip(G, -1.0, 1.0)
        return np.arccos(G)


@dataclass
class RP2GeodesicMetric:
    """
    Projective geodesic distance on RP^2 via S^2 reps:
        d([x],[y]) = arccos(|<x,y>|), with x,y unit vectors.
    """
    eps: float = 1e-12

    def pairwise(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
        A = _normalize_rows(np.asarray(A, dtype=float), eps=self.eps)
        if B is None:
            B = A
        else:
            B = _normalize_rows(np.asarray(B, dtype=float), eps=self.eps)

        G = np.abs(A @ B.T)
        G = np.clip(G, 0.0, 1.0)
        return np.arccos(G)


# -----------------------------------------------------------------------------
# Radius heuristics / tuning
# -----------------------------------------------------------------------------

def _knn_radius_from_landmarks(
    dist_ll: np.ndarray,
    *,
    k: int = 7,
    radius_scale: float = 1.05,
) -> float:
    """
    Pick an initial radius from landmark-landmark distances.

    Uses the median kNN distance among landmarks, scaled by radius_scale.

    dist_ll : (m,m) distances (diagonal assumed 0)
    """
    m = dist_ll.shape[0]
    if m < 2:
        raise ValueError("Need at least 2 landmarks to set radius from kNN.")
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1. Got {k}.")
    k = min(k, m - 1)

    # sort each row; first entry is 0 (self)
    row_sorted = np.sort(dist_ll, axis=1)
    kth = row_sorted[:, k]  # k-th neighbor (since index 0 is self)
    r0 = float(np.median(kth) * float(radius_scale))
    return r0


def _max_overlap_order(U: np.ndarray) -> int:
    U = np.asarray(U, dtype=bool)
    if U.size == 0:
        return 0
    return int(U.sum(axis=0).max(initial=0))


def _tune_radius_to_avoid_5way_overlaps(
    cover: MetricBallCover,
    *,
    target_max_overlap: int = 4,
    max_iters: int = 30,
    shrink: float = 0.92,
    grow: float = 1.05,
    min_edges: int = 1,
) -> MetricBallCover:
    """
    Adjust cover.radius in-place (rebuild each time) to try to enforce:
      - max sample overlap <= target_max_overlap
    while keeping the nerve from becoming completely trivial.

    This is a pragmatic heuristic:
      - If max overlap too high: shrink radius.
      - Else if nerve too sparse (few edges): slightly grow radius.
      - Else accept.
    """
    if not (1 <= target_max_overlap <= 10):
        raise ValueError("target_max_overlap should be a small positive integer.")
    if not (0.0 < shrink < 1.0):
        raise ValueError("shrink must be in (0,1).")
    if not (grow > 1.0):
        raise ValueError("grow must be > 1.0.")

    r = float(cover.radius)
    for _ in range(int(max_iters)):
        cover.radius = float(r)
        cover.build()

        assert cover.U is not None
        max_ov = _max_overlap_order(cover.U)

        # Quick sparsity signal: do we have at least some overlaps?
        n_edges = len(cover.nerve_edges())

        if max_ov > target_max_overlap:
            r *= float(shrink)
            continue

        if n_edges < int(min_edges):
            r *= float(grow)
            continue

        # Good enough
        return cover

    # Final rebuild at last r
    cover.radius = float(r)
    cover.build()
    return cover


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------

def make_s2_metric_ball_cover(
    base_points: np.ndarray,
    *,
    n_landmarks: int = 128,
    radius: Optional[float] = None,
    radius_k: int = 7,
    radius_scale: float = 1.05,
    tune_radius: bool = True,
    target_max_overlap: int = 4,
    min_edges: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> MetricBallCover:
    """
    Build a MetricBallCover on S^2 using spherical Fibonacci landmarks and geodesic distance.

    Parameters
    ----------
    base_points : (n,3)
        Points on S^2 (or nonzero vectors; will be normalized).
    n_landmarks : int
        Number of Fibonacci landmarks.
    radius : float | None
        If None, choose via median kNN distance among landmarks (k=radius_k), scaled by radius_scale.
    tune_radius : bool
        If True, heuristically adjust radius to avoid 5-way overlaps (4-simplices in the nerve).
    target_max_overlap : int
        Enforce max sample overlap <= this (default 4).
    min_edges : int
        If cover becomes too sparse, we grow radius a bit.

    Returns
    -------
    cover : MetricBallCover
        cover.metric is S2GeodesicMetric
        cover.landmarks are Fibonacci points on S^2 (in R^3)
    """
    P = _normalize_rows(np.asarray(base_points, dtype=float))
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {P.shape}.")

    L = fibonacci_sphere(int(n_landmarks), rng=rng)
    metric = S2GeodesicMetric()

    if radius is None:
        dist_ll = metric.pairwise(L)  # (m,m)
        r0 = _knn_radius_from_landmarks(dist_ll, k=radius_k, radius_scale=radius_scale)
    else:
        r0 = float(radius)

    cover = MetricBallCover(base_points=P, landmarks=L, radius=r0, metric=metric)
    cover.build()

    if tune_radius:
        cover = _tune_radius_to_avoid_5way_overlaps(
            cover,
            target_max_overlap=target_max_overlap,
            min_edges=min_edges,
        )

    return cover


def make_rp2_metric_ball_cover(
    base_points: np.ndarray,
    *,
    n_landmarks: int = 128,
    radius: Optional[float] = None,
    radius_k: int = 7,
    radius_scale: float = 1.05,
    tune_radius: bool = True,
    target_max_overlap: int = 4,
    min_edges: int = 1,
    rng: Optional[np.random.Generator] = None,
    hemisphere_atol: float = 1e-12,
) -> MetricBallCover:
    """
    Build a MetricBallCover on RP^2 using hemisphere representatives + projective geodesic distance.

    Representation
    --------------
    - Input base_points are treated as S^2 reps; we map to canonical hemisphere reps.
    - Landmarks are Fibonacci points mapped to canonical hemisphere reps.
    - Metric is projective:
        d([x],[y]) = arccos(|<x,y>|)

    Parameters are analogous to make_s2_metric_ball_cover.

    Returns
    -------
    cover : MetricBallCover
        cover.metric is RP2GeodesicMetric
        cover.base_points are canonical hemisphere reps (R^3)
        cover.landmarks are canonical hemisphere reps (R^3)
    """
    P0 = _normalize_rows(np.asarray(base_points, dtype=float))
    if P0.ndim != 2 or P0.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {P0.shape}.")

    P = canonical_hemisphere_reps(P0, atol=hemisphere_atol)

    L0 = fibonacci_sphere(int(n_landmarks), rng=rng)
    L = canonical_hemisphere_reps(L0, atol=hemisphere_atol)

    metric = RP2GeodesicMetric()

    if radius is None:
        dist_ll = metric.pairwise(L)  # (m,m)
        r0 = _knn_radius_from_landmarks(dist_ll, k=radius_k, radius_scale=radius_scale)
    else:
        r0 = float(radius)

    cover = MetricBallCover(base_points=P, landmarks=L, radius=r0, metric=metric)
    cover.build()

    if tune_radius:
        cover = _tune_radius_to_avoid_5way_overlaps(
            cover,
            target_max_overlap=target_max_overlap,
            min_edges=min_edges,
        )

    return cover
