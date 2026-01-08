# quality.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .combinatorics import Edge, Tri


@dataclass
class BundleQualityReport:
    # --- paper-consistent surjectivity proxy (existing) ---
    delta: float  # NOTE: this is your "max over triangles of min over vertices of covering radius"

    # --- transition consistency ---
    cocycle_defect: float

    # --- edgewise transition fit summaries (from transitions_report) ---
    max_edge_rms: float
    mean_edge_rms: float

    # --- alignment error on overlaps (NEW naming) ---
    eps_align_geo: Optional[float]   # ε, geodesic distance on S^1, radians in [0, π]
    eps_align_euc: Optional[float]   # chordal version: 2 sin(ε/2) in [0, 2]

    # --- derived stability-ish ratio ---
    alpha: Optional[float]           # α = ε/(1-δ), +inf if δ>=1

    # --- bookkeeping ---
    n_edges_estimated: int
    n_edges_requested: int
    n_triangles: int


# ----------------------------
# Circle helpers
# ----------------------------

_TWO_PI = 2.0 * np.pi


def h_dist_S1(angle_array: np.ndarray) -> Tuple[float, float]:
    """
    Returns (h_euc, h_geo) where:
      h_geo = half the maximum gap on the circle (radians)
      h_euc = chordal version: 2 sin(h_geo/2)
    """
    a = np.asarray(angle_array, dtype=float).reshape(-1)
    if a.size == 0:
        return float("inf"), float("inf")

    a = np.sort(a % _TWO_PI)
    gaps = np.diff(a)
    gaps = np.append(gaps, _TWO_PI - (a[-1] - a[0]))
    h_geo = float(np.max(gaps) / 2.0)  # in [0, pi]
    h_euc = float(2.0 * np.sin(h_geo / 2.0))
    return h_euc, h_geo


def _s1_geodesic_from_unit_vectors(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """u,v: (...,2) unit vectors. Return geodesic distance (radians) in [0, pi]."""
    dot = np.sum(u * v, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)


def _eps_geo_to_euc(eps_geo: float) -> float:
    """Convert geodesic angle (radians) to chordal distance in R^2."""
    return float(2.0 * np.sin(float(eps_geo) / 2.0))


# ----------------------------
# Paper-consistent delta (existing, unchanged)
# ----------------------------

def compute_delta_from_triples(
    U: np.ndarray,
    f: np.ndarray,
    triangles: Iterable[Tri],
    *,
    min_points: int = 5,
    use_euclidean: bool = True,
    fail_fast: bool = True,
) -> float:
    """
    For each triangle (i,j,k):
      - compute h_dist_S1 on each vertex image set over the triple overlap
      - take MIN over vertices
    Then take MAX over triangles.

    This is the 'best-chart' triple-overlap surjectivity proxy used in your paper.
    """
    U = np.asarray(U, dtype=bool)
    f = np.asarray(f, dtype=float)

    triangles = [tuple(sorted(map(int, t))) for t in triangles]
    if not triangles:
        return 0.0

    vals: List[float] = []
    for (i, j, k) in triangles:
        idx = U[i] & U[j] & U[k]
        m = int(idx.sum())
        if m < min_points:
            if fail_fast:
                raise ValueError(
                    f"Triple intersection too small for triangle {(i,j,k)}: "
                    f"{m} points < min_points={min_points}."
                )
            continue

        per_vertex = []
        for v in (i, j, k):
            h_euc, h_geo = h_dist_S1(f[v, idx])
            per_vertex.append(h_euc if use_euclidean else h_geo)

        vals.append(float(np.min(per_vertex)))

    return float(np.max(vals)) if vals else 0.0


# ----------------------------
# Cocycle defect (existing)
# ----------------------------

def compute_O2_cocycle_defect(
    cocycle,
    triangles: Iterable[Tri],
    *,
    matrix_norm: str = "fro",
) -> float:
    triangles = [tuple(sorted(map(int, t))) for t in triangles]
    if not triangles:
        return 0.0

    coc = cocycle.complete_orientations()

    def norm(M: np.ndarray) -> float:
        if matrix_norm == "fro":
            return float(np.linalg.norm(M, ord="fro"))
        if matrix_norm == "op":
            return float(np.linalg.norm(M, ord=2))
        raise ValueError("matrix_norm must be 'fro' or 'op'.")

    worst = 0.0
    for (i, j, k) in triangles:
        Oij = coc.Omega[(i, j)]
        Ojk = coc.Omega[(j, k)]
        Oki = coc.Omega[(k, i)]
        M = Oij @ Ojk @ Oki
        worst = max(worst, norm(M - np.eye(2)))
    return worst


# ----------------------------
# NEW: epsilon alignment sup (geodesic), plus chordal conversion
# ----------------------------

def compute_eps_alignment_sup_geo(
    U: np.ndarray,
    f: np.ndarray,
    cocycle,
    edges: Iterable[Edge],
    *,
    min_points: int = 1,
) -> Optional[float]:
    """
    ε = sup_{(jk)} sup_{x in overlap} d_{S^1}( Ω_{jk} f_k(x), f_j(x) )
    where Ω_{jk} maps k -> j.

    Implemented using unit vectors and arccos(dot) ∈ [0, pi].
    Returns None if no usable overlaps.
    """
    U = np.asarray(U, dtype=bool)
    f = np.asarray(f, dtype=float)

    coc = cocycle.complete_orientations()
    edges = list(edges)

    worst = 0.0
    got_any = False

    for (j, k) in edges:
        if (j, k) not in coc.Omega:
            continue

        idx = np.where(U[j] & U[k])[0]
        if idx.size < min_points:
            continue

        O = coc.Omega[(j, k)]  # maps k -> j
        vj = np.stack([np.cos(f[j, idx]), np.sin(f[j, idx])], axis=1)
        vk = np.stack([np.cos(f[k, idx]), np.sin(f[k, idx])], axis=1)
        vk_trans = (O @ vk.T).T

        dgeo = _s1_geodesic_from_unit_vectors(vj, vk_trans)
        worst = max(worst, float(np.max(dgeo)))
        got_any = True

    return float(worst) if got_any else None


def compute_alpha_from_eps_delta(
    eps_geo: Optional[float],
    delta: float,
) -> Optional[float]:
    """
    α = ε/(1-δ) with α=+inf if δ>=1.
    If eps_geo is None, returns None.
    """
    if eps_geo is None:
        return None
    if float(delta) >= 1.0:
        return float("inf")
    return float(eps_geo) / (1.0 - float(delta))


# ----------------------------
# Main entry point
# ----------------------------

def compute_bundle_quality(
    cover,
    local_triv,
    cocycle,
    transitions_report,
    *,
    edges: Optional[Iterable[Edge]] = None,
    triangles: Optional[Iterable[Tri]] = None,
    # delta settings (paper delta)
    delta_min_points: int = 5,
    delta_use_euclidean: bool = True,
    delta_fail_fast: bool = True,
    # cocycle defect settings
    cocycle_matrix_norm: str = "fro",
    # epsilon settings
    eps_min_points: int = 1,
) -> BundleQualityReport:
    """
    Compute standard diagnostics for a bundle pipeline run.
    """
    if edges is None:
        edges_list = list(cover.nerve_edges())
    else:
        edges_list = list(edges)

    if triangles is None:
        triangles_list = list(cover.nerve_triangles())
    else:
        triangles_list = list(triangles)

    # paper delta (unchanged)
    delta = compute_delta_from_triples(
        cover.U,
        local_triv.f,
        triangles_list,
        min_points=delta_min_points,
        use_euclidean=delta_use_euclidean,
        fail_fast=delta_fail_fast,
    )

    # cocycle triangle defect (unchanged)
    cocycle_defect = compute_O2_cocycle_defect(
        cocycle,
        triangles_list,
        matrix_norm=cocycle_matrix_norm,
    )

    # edge RMS summaries (unchanged)
    max_edge_rms = float(getattr(transitions_report, "max_rms_angle_err", 0.0))
    mean_edge_rms = float(getattr(transitions_report, "mean_rms_angle_err", 0.0))

    # ε (geodesic, radians)
    eps_geo = compute_eps_alignment_sup_geo(
        cover.U,
        local_triv.f,
        cocycle,
        edges_list,
        min_points=eps_min_points,
    )

    # ε converted to chordal distance in R^2 (explicitly “witness-style”)
    eps_euc = None if eps_geo is None else _eps_geo_to_euc(eps_geo)

    # α based on *paper delta*
    alpha = compute_alpha_from_eps_delta(eps_geo, float(delta))

    n_edges_requested = int(getattr(transitions_report, "n_edges_requested", len(edges_list)))
    n_edges_estimated = int(getattr(transitions_report, "n_edges_estimated", len(getattr(cocycle, "Omega", {}))))

    return BundleQualityReport(
        delta=float(delta),
        cocycle_defect=float(cocycle_defect),
        max_edge_rms=max_edge_rms,
        mean_edge_rms=mean_edge_rms,
        eps_align_geo=eps_geo,
        eps_align_euc=eps_euc,
        alpha=alpha,
        n_edges_estimated=n_edges_estimated,
        n_edges_requested=n_edges_requested,
        n_triangles=int(len(triangles_list)),
    )
