# circle_bundles/analysis/global_trivialization.py
"""
Global trivialization (Singer synchronization)

Design goals (Feb 2026)
----------------------
- Bundle-first, clean, and minimal.
- Singer's method ONLY (spectral synchronization on the 1-skeleton).
- normalize_degree=True ALWAYS.
- Partition of unity (pou) is REQUIRED (no internal generation).
- Supports orienting-gauge application (reflect charts with phi=-1) when needed.
- Provides helper to compute the "max trivial" subcomplex from persistence
  (earliest step where BOTH w1 and twisted Euler become coboundaries).

This module is intentionally separate from the legacy coordinatization.py
during the refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..nerve.combinatorics import Edge, Tri, canon_edge, canon_tri

Simp = Tuple[int, ...]
Tet = Tuple[int, int, int, int]

__all__ = [
    "MaxTrivialSubcomplex",
    "compute_max_trivial_subcomplex",
    "GlobalTrivializationResult",
    "build_global_trivialization_singer",
    "apply_orientation_gauge_to_f",
    "theta_dict_to_edge_vector_radians",
    "wrap_angle_rad",
]


# ============================================================
# Canonicalization helpers
# ============================================================

def canon_simplex(sig: Iterable[int]) -> Simp:
    return tuple(sorted(int(x) for x in sig))


def canon_edge_tuple(e: Edge) -> Edge:
    return canon_edge(int(e[0]), int(e[1]))


def canon_tri_tuple(t: Tri) -> Tri:
    return canon_tri(int(t[0]), int(t[1]), int(t[2]))


def canon_tet_tuple(t: Tet) -> Tet:
    a, b, c, d = (int(x) for x in t)
    tt = tuple(sorted((a, b, c, d)))
    return (tt[0], tt[1], tt[2], tt[3])


# ============================================================
# Persistence + max-trivial subcomplex bookkeeping
# ============================================================

@dataclass
class MaxTrivialSubcomplex:
    """
    Earliest step of the edge-removal filtration where BOTH:
      - w1 is a coboundary, and
      - (twisted) Euler is a coboundary.
    """
    k_removed: int
    removed_edges: List[Edge]
    kept_edges: List[Edge]
    kept_triangles: List[Tri]
    kept_tetrahedra: List[Tet]


def induced_triangles_from_edges(triangles: List[Tri], kept_edges_set: Set[Edge]) -> List[Tri]:
    kept: List[Tri] = []
    for t in triangles:
        i, j, k = canon_tri_tuple(t)
        eij = canon_edge(i, j)
        eik = canon_edge(i, k)
        ejk = canon_edge(j, k)
        if (eij in kept_edges_set) and (eik in kept_edges_set) and (ejk in kept_edges_set):
            kept.append((i, j, k))
    return kept


def induced_tetrahedra_from_edges(tets: List[Tet], kept_edges_set: Set[Edge]) -> List[Tet]:
    kept: List[Tet] = []
    for tt in tets:
        a, b, c, d = canon_tet_tuple(tt)
        edges6 = [
            canon_edge(a, b), canon_edge(a, c), canon_edge(a, d),
            canon_edge(b, c), canon_edge(b, d),
            canon_edge(c, d),
        ]
        if all(e in kept_edges_set for e in edges6):
            kept.append((a, b, c, d))
    return kept


def compute_max_trivial_subcomplex(
    *,
    persistence: Any,   # PersistenceResult from compute_bundle_persistence
    edges: List[Edge],
    triangles: List[Tri],
    tets: Optional[List[Tet]] = None,
) -> Optional[MaxTrivialSubcomplex]:
    """
    Pick the earliest step in the edge-removal filtration where BOTH:
      - w1 is a coboundary, and
      - twisted Euler is a coboundary.

    k = max(k_sw1_codeath, k_euler_codeath).

    Returns None if either class never becomes a coboundary (k_removed == -1).
    """
    sw1_cod = persistence.sw1["codeath"]
    te_cod = persistence.twisted_euler["codeath"]

    if sw1_cod.k_removed < 0 or te_cod.k_removed < 0:
        return None

    removal_order: List[Edge] = list(persistence.sw1["removal_order"])

    k = int(max(sw1_cod.k_removed, te_cod.k_removed))
    removed_edges = [canon_edge_tuple(e) for e in removal_order[:k]]
    removed_set = set(removed_edges)

    edges_all = sorted({canon_edge_tuple(e) for e in edges})
    kept_edges_set = set(edges_all) - removed_set
    kept_edges = sorted(kept_edges_set)

    triangles_all = [canon_tri_tuple(t) for t in triangles]
    kept_triangles = induced_triangles_from_edges(triangles_all, kept_edges_set)

    tets_all = [canon_tet_tuple(tt) for tt in (tets or [])]
    kept_tets = induced_tetrahedra_from_edges(tets_all, kept_edges_set) if tets_all else []

    return MaxTrivialSubcomplex(
        k_removed=k,
        removed_edges=removed_edges,
        kept_edges=kept_edges,
        kept_triangles=kept_triangles,
        kept_tetrahedra=kept_tets,
    )


# ============================================================
# Global trivialization (radians)
# ============================================================

def wrap_angle_rad(x: np.ndarray) -> np.ndarray:
    """Wrap radians to [0, 2π)."""
    return (x + 2 * np.pi) % (2 * np.pi)


def frechet_mean_circle(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Weighted circular (Fréchet) mean per column.

    angles:  (n,M) radians (any real; treated mod 2π)
    weights: (n,M) nonnegative
    returns: (M,) radians in [0, 2π)
    """
    sin_sum = np.sum(weights * np.sin(angles), axis=0)
    cos_sum = np.sum(weights * np.cos(angles), axis=0)
    return wrap_angle_rad(np.arctan2(sin_sum, cos_sum))


# ============================================================
# Orientation gauge helpers (optional, for O(2) -> SO(2) restriction)
# ============================================================

def reflect_angles_about_axis(f: np.ndarray, ref_angle: float) -> np.ndarray:
    """
    Reflect angles across the axis at angle ref_angle:
        alpha -> 2*ref_angle - alpha (mod 2π).
    """
    a = float(ref_angle)
    return wrap_angle_rad(2.0 * a - f)


def apply_orientation_gauge_to_f(
    *,
    f: np.ndarray,            # (n,M) radians
    phi_pm1: np.ndarray,      # (n,) entries ±1
    ref_angle: float,
    U: Optional[np.ndarray] = None,  # (n,M) mask; if provided, re-mask after transform
) -> np.ndarray:
    """
    Apply an orientation gauge that reflects charts with phi=-1.

    Convention:
      g_j = I if phi_j=+1
      g_j = r(ref_angle) if phi_j=-1

    So:
      f'_j = f_j                 if phi_j=+1
      f'_j = 2*ref_angle - f_j   if phi_j=-1
    """
    f = np.asarray(f, dtype=float)
    phi_pm1 = np.asarray(phi_pm1, dtype=int).reshape(-1)

    if f.ndim != 2:
        raise ValueError(f"f must be 2D (n,M); got shape {f.shape}")

    n, M = f.shape
    if phi_pm1.shape != (n,):
        raise ValueError(f"phi_pm1 must have shape {(n,)}, got {phi_pm1.shape}")

    out = f.copy()
    flip = (phi_pm1 == -1)
    if np.any(flip):
        out[flip, :] = reflect_angles_about_axis(out[flip, :], ref_angle=ref_angle)

    out = wrap_angle_rad(out)

    if U is not None:
        U = np.asarray(U)
        if U.shape != (n, M):
            raise ValueError(f"U must have shape {(n,M)}, got {U.shape}")
        out *= (U > 0)

    return out


# ============================================================
# Theta extraction (dict -> edge vector, radians)
# ============================================================

def theta_dict_to_edge_vector_radians(
    *,
    edges: List[Edge],
    theta: Dict[Edge, float],
) -> np.ndarray:
    """
    edges: list of canonical edges (j<k)
    theta: dict keyed by (j,k) canonical with values in radians [0,2π)

    returns theta_edge: (E,) radians in [0,2π)
    """
    edges_c = [canon_edge_tuple(e) for e in edges]
    theta_c: Dict[Edge, float] = {canon_edge_tuple(e): float(v) for e, v in theta.items()}

    out = np.zeros((len(edges_c),), dtype=float)
    for r, e in enumerate(edges_c):
        if e not in theta_c:
            raise KeyError(f"Missing theta on edge {e}")
        out[r] = float(theta_c[e]) % (2 * np.pi)
    return out


# ============================================================
# Singer synchronization (spectral) + global coordinate
# ============================================================

def _build_edge_lookup(edges: List[Edge]) -> Dict[Tuple[int, int], int]:
    """
    edges: canonical undirected edges (j<k)

    Returns:
      idx[(j,k)] = r and idx[(k,j)] = r
    """
    idx: Dict[Tuple[int, int], int] = {}
    for r, (a, b) in enumerate(edges):
        j, k = canon_edge(int(a), int(b))
        idx[(j, k)] = r
        idx[(k, j)] = r
    return idx


def _mu_vertices_from_singer_radians(
    *,
    edges: List[Edge],          # canonical edges j<k
    theta_edge: np.ndarray,     # (E,) radians
    n_vertices: int,
) -> np.ndarray:
    """
    Vertex gauge mu_v (radians) by Singer spectral synchronization.

    normalize_degree=True ALWAYS.
    """
    edges = [canon_edge_tuple(e) for e in edges]
    theta_edge = np.asarray(theta_edge, dtype=float).reshape(-1)
    if theta_edge.shape != (len(edges),):
        raise ValueError(f"theta_edge must have shape {(len(edges),)}, got {theta_edge.shape}")

    idx = _build_edge_lookup(edges)
    H = np.zeros((n_vertices, n_vertices), dtype=np.complex128)
    deg = np.zeros(n_vertices, dtype=float)

    for (j, k) in edges:
        th_jk = float(theta_edge[idx[(j, k)]])
        z = np.cos(th_jk) + 1j * np.sin(th_jk)  # exp(i th)
        H[j, k] = np.conjugate(z)  # exp(-i th)
        H[k, j] = z                # exp(+i th)
        deg[j] += 1.0
        deg[k] += 1.0

    # normalize_degree=True
    inv_sqrt = np.zeros_like(deg)
    mask = deg > 0
    inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
    H = (inv_sqrt[:, None] * H) * inv_sqrt[None, :]

    _, V = np.linalg.eigh(H)
    v = V[:, -1]
    v = v / (np.linalg.norm(v) + 1e-15)

    return wrap_angle_rad(np.angle(v))


def _global_from_mu(
    *,
    f: np.ndarray,        # (n,M) radians
    mu: np.ndarray,       # (n,M) radians
    U: np.ndarray,
    pou: np.ndarray,
) -> np.ndarray:
    """
    Build global angle F[m] as weighted Fréchet mean of gauged locals:
        g_j = f_j + mu_j (mod 2π).
    """
    g = wrap_angle_rad(f + mu)
    w = np.asarray(pou, dtype=float) * (np.asarray(U) > 0)
    return frechet_mean_circle(g, w)


def build_global_trivialization_singer(
    *,
    edges: List[Edge],
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    theta: Dict[Edge, float] | np.ndarray,
    n_vertices: Optional[int] = None,
) -> np.ndarray:
    """
    Compute a global circle coordinate F (M,) in radians using Singer's method.

    Parameters
    ----------
    edges:
        Canonical cover adjacency edges (j<k).
    U:
        Cover membership mask, shape (n_sets, n_samples).
    pou:
        Partition of unity, shape (n_sets, n_samples). REQUIRED.
    f:
        Local circular coordinates per chart, shape (n_sets, n_samples), in radians.
        (Typically already oriented/gauged if you’re restricting to det=+1 transitions.)
    theta:
        Either:
          - dict keyed by canonical edges with values theta_{jk} in radians, or
          - array theta_edge of shape (E,) aligned with `edges`.
    n_vertices:
        Optional. If not provided, inferred as U.shape[0].

    Returns
    -------
    F:
        Global angle array of shape (n_samples,), wrapped to [0, 2π).

    Notes
    -----
    - This is intentionally minimal: Singer only, normalize_degree always on.
    - No summaries/plots here.
    """
    U = np.asarray(U)
    pou = np.asarray(pou, dtype=float)
    f = np.asarray(f, dtype=float)

    if U.ndim != 2:
        raise ValueError(f"U must be 2D (n_sets,n_samples); got {U.shape}")
    n, M = U.shape

    if pou.shape != (n, M):
        raise ValueError(f"pou must have shape {(n,M)}; got {pou.shape}")
    if f.shape != (n, M):
        raise ValueError(f"f must have shape {(n,M)}; got {f.shape}")

    edges_c = [canon_edge_tuple(e) for e in edges]

    if n_vertices is None:
        n_vertices = n
    n_vertices = int(n_vertices)
    if n_vertices != n:
        # In this codebase, vertices are cover sets, so keep this strict.
        raise ValueError(f"n_vertices must equal n_sets={n}; got {n_vertices}")

    if isinstance(theta, dict):
        theta_edge = theta_dict_to_edge_vector_radians(edges=edges_c, theta=theta)
    else:
        theta_edge = np.asarray(theta, dtype=float).reshape(-1)
        if theta_edge.shape != (len(edges_c),):
            raise ValueError(f"theta_edge must have shape {(len(edges_c),)}, got {theta_edge.shape}")
        theta_edge = theta_edge % (2 * np.pi)

    mu_v = _mu_vertices_from_singer_radians(edges=edges_c, theta_edge=theta_edge, n_vertices=n_vertices)
    mu = mu_v[:, None] * np.ones_like(pou, dtype=float)
    mu *= (U > 0)

    return _global_from_mu(f=f, mu=mu, U=U, pou=pou)


# ============================================================
# Result container (optional convenience)
# ============================================================

@dataclass
class GlobalTrivializationResult:
    """
    Small result container for a computed global circle coordinate.
    """
    method: str
    edges_used: List[Edge]
    F: np.ndarray
    meta: Dict[str, Any]
