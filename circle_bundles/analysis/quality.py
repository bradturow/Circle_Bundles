# quality.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np

from ..nerve.combinatorics import Edge, Tri


@dataclass
class BundleQualityReport:
    """
    Summary of diagnostic and quality metrics for a reconstructed bundle.

    This object is intended for *inspection and reporting*, not as an input to
    downstream algorithms. It aggregates several geometric, cohomological,
    and numerical indicators that help assess how well the data supports a
    coherent circle- (or O(2)-) bundle structure.

    Notes
    -----
    - Users typically obtain this object via a bundle method (e.g. ``get_quality()``),
      rather than constructing it directly.
    - Individual fields may be None if the corresponding diagnostic was not
      computed or not applicable to the current bundle model.

    Fields
    ------
    delta:
        Global transition inconsistency measure (smaller is better).
    cocycle_defect:
        Deviation of the recovered cocycle from being exactly closed.

    max_edge_rms, mean_edge_rms:
        Maximum and mean RMS angular transition error over edges.

    eps_align_geo, eps_align_euc:
        Worst-case alignment error (geodesic / Euclidean) for local trivializations.

    eps_align_geo_mean, eps_align_euc_mean:
        Mean alignment error (geodesic / Euclidean).

    alpha:
        Global scaling or regularization parameter used in alignment (if applicable).

    n_edges_estimated:
        Number of edges actually used for estimation.
    n_edges_requested:
        Number of edges originally requested.
    n_triangles:
        Number of triangles used in consistency checks.

    witness_err, witness_err_geo:
        Supremum witness error (chordal / geodesic).
    witness_err_mean, witness_err_geo_mean:
        Mean witness error (chordal / geodesic).

    cocycle_proj_dist:
        Distance from the cocycle to the nearest projected cocycle representative.
    """
    delta: float
    cocycle_defect: float

    max_edge_rms: float
    mean_edge_rms: float

    eps_align_geo: Optional[float]
    eps_align_euc: Optional[float]

    eps_align_geo_mean: Optional[float]
    eps_align_euc_mean: Optional[float]

    alpha: Optional[float]

    n_edges_estimated: int
    n_edges_requested: int
    n_triangles: int

    witness_err: Optional[float] = None          # chordal sup
    witness_err_geo: Optional[float] = None      # geodesic sup (radians)
    witness_err_mean: Optional[float] = None     # chordal mean
    witness_err_geo_mean: Optional[float] = None # geodesic mean (radians)

    cocycle_proj_dist: Optional[float] = None




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
# epsilon alignment sup (geodesic), plus chordal conversion
# ----------------------------

def compute_eps_alignment_stats(
    U: np.ndarray,
    f: np.ndarray,
    cocycle,
    edges: Iterable[Edge],
    *,
    min_points: int = 1,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Returns (eps_geo_sup, eps_geo_mean, eps_euc_sup, eps_euc_mean)

    geo: geodesic angle in radians in [0, pi]
    euc: chordal distance in R^2, 2 sin(geo/2) in [0, 2]
    """
    U = np.asarray(U, dtype=bool)
    f = np.asarray(f, dtype=float)

    coc = cocycle.complete_orientations()
    edges = list(edges)

    geo_worst = 0.0
    euc_worst = 0.0
    got_any = False

    geo_sum = 0.0
    euc_sum = 0.0
    n = 0

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

        dgeo = _s1_geodesic_from_unit_vectors(vj, vk_trans)          # [0, pi]
        deuc = 2.0 * np.sin(dgeo / 2.0)                              # [0, 2]

        geo_worst = max(geo_worst, float(np.max(dgeo)))
        euc_worst = max(euc_worst, float(np.max(deuc)))

        geo_sum += float(np.sum(dgeo))
        euc_sum += float(np.sum(deuc))
        n += int(dgeo.size)

        got_any = True

    if (not got_any) or (n == 0):
        return None, None, None, None

    return (
        float(geo_worst),
        float(geo_sum / n),
        float(euc_worst),
        float(euc_sum / n),
    )


def compute_alpha_from_eps_delta_euc(
    eps_euc: Optional[float],
    delta: float,
) -> Optional[float]:
    """
    α = ε/(1-δ) where ε is the *Euclidean/chordal* alignment error in [0,2].
    α=+inf if δ>=1. If eps_euc is None, returns None.
    """
    if eps_euc is None:
        return None
    if float(delta) >= 1.0:
        return float("inf")
    return float(eps_euc) / (1.0 - float(delta))



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
    compute_witness: bool = False,
) -> BundleQualityReport:
    """
    Compute standard diagnostics for a bundle pipeline run.

    Conventions
    ----------
    - delta is computed either in chordal (Euclidean in R^2) or geodesic (radians),
      depending on delta_use_euclidean (default True => chordal).
    - eps alignment is computed in BOTH:
        * eps_align_euc / eps_align_euc_mean : chordal in [0,2]
        * eps_align_geo / eps_align_geo_mean : radians in [0,pi]
    - alpha is computed using the Euclidean/chordal epsilon:
        alpha = eps_align_euc / (1 - delta)
      (so if you set delta_use_euclidean=True, numerator/denominator are in the same style)
    """
    if edges is None:
        edges_list = list(cover.nerve_edges())
    else:
        edges_list = list(edges)

    if triangles is None:
        triangles_list = list(cover.nerve_triangles())
    else:
        triangles_list = list(triangles)

    # paper delta 
    delta = compute_delta_from_triples(
        cover.U,
        local_triv.f,
        triangles_list,
        min_points=delta_min_points,
        use_euclidean=delta_use_euclidean,
        fail_fast=delta_fail_fast,
    )

    # cocycle triangle defect
    cocycle_defect = compute_O2_cocycle_defect(
        cocycle,
        triangles_list,
        matrix_norm=cocycle_matrix_norm,
    )

    # edge RMS summaries (from transitions_report)
    max_edge_rms = float(getattr(transitions_report, "max_rms_angle_err", 0.0))
    mean_edge_rms = float(getattr(transitions_report, "mean_rms_angle_err", 0.0))

    # epsilon alignment stats: (geo sup, geo mean, euc sup, euc mean)
    eps_geo, eps_geo_mean, eps_euc, eps_euc_mean = compute_eps_alignment_stats(
        cover.U,
        local_triv.f,
        cocycle,
        edges_list,
        min_points=eps_min_points,
    )

    # alpha computed using Euclidean epsilon
    alpha = compute_alpha_from_eps_delta_euc(eps_euc, float(delta))

    n_edges_requested = int(getattr(transitions_report, "n_edges_requested", len(edges_list)))
    n_edges_estimated = int(getattr(transitions_report, "n_edges_estimated", len(getattr(cocycle, "Omega", {}))))

    # ------------------------------------------
    # Optional witness diagnostics (Π(Ω) based)
    # ------------------------------------------
    witness_err = None
    witness_err_geo = None
    witness_err_mean = None
    witness_err_geo_mean = None
    cocycle_proj_dist = None

    if compute_witness:
        # local import to avoid any import-cycle surprises
        from ..trivializations.bundle_map import (
            build_true_frames,
            witness_error_stats,
            cocycle_projection_distance,
        )

        # Use the same edge set you used everywhere else
        # and the cocycle's Omega (simplicial) as input.
        Omega = getattr(cocycle, "Omega", None)
        if Omega is None:
            raise AttributeError("cocycle.Omega is missing; cannot compute witness diagnostics.")

        tf = build_true_frames(
            U=cover.U,
            pou=cover.pou,
            Omega=Omega,
            edges=edges_list,
        )

        # witness stats compare local triv angles using Π(Ω) (Omega_true from frames)
        w_geo, w_geo_mean, w_c, w_c_mean = witness_error_stats(
            U=cover.U,
            f=local_triv.f,
            Omega_true=tf.Omega_true,
            edges=edges_list,
        )

        witness_err_geo = w_geo
        witness_err_geo_mean = w_geo_mean
        witness_err = w_c
        witness_err_mean = w_c_mean

        # optionally record cocycle projection distance d_infty(Ω, Π(Ω))
        cocycle_proj_dist = cocycle_projection_distance(
            U=cover.U,
            Omega_simplicial=Omega,
            Omega_true=tf.Omega_true,
            edges=edges_list,
        )

    
    return BundleQualityReport(
        delta=float(delta),
        cocycle_defect=float(cocycle_defect),
        max_edge_rms=max_edge_rms,
        mean_edge_rms=mean_edge_rms,
        eps_align_geo=eps_geo,
        eps_align_euc=eps_euc,
        eps_align_geo_mean=eps_geo_mean,
        eps_align_euc_mean=eps_euc_mean,
        alpha=alpha,
        n_edges_estimated=n_edges_estimated,
        n_edges_requested=n_edges_requested,
        n_triangles=int(len(triangles_list)),
        witness_err=witness_err,
        witness_err_geo=witness_err_geo,
        witness_err_mean=witness_err_mean,
        witness_err_geo_mean=witness_err_geo_mean,
        cocycle_proj_dist=cocycle_proj_dist,
    )


