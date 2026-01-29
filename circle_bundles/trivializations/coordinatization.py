# coordinatization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from ..nerve.combinatorics import Edge, Tri, canon_edge, canon_tri

Simp = Tuple[int, ...]
Tet = Tuple[int, int, int, int]


__all__ = [
    "GlobalTrivializationResult",
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
    Pick the *earliest* step in the edge-removal filtration where BOTH:
      - SW1 is a coboundary, and
      - twisted Euler class is a coboundary.

    Earliest simultaneous step is k = max(k_sw1_codeath, k_euler_codeath).
    Returns None if either codeath is not achieved (k_removed == -1).

    Notes
    -----
    - Backwards-compatible: if you don't pass tets, kept_tetrahedra will be [].
    - We do NOT additionally require "Euler cobirth" (cocycle on tets) here;
      this helper is purely about finding a subcomplex where the *classes trivialize*
      (both are coboundaries).
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
# Global trivialization (RADIANS, single convention)
# ============================================================

def wrap_angle_rad(x: np.ndarray) -> np.ndarray:
    """Wrap radians to [0, 2π)."""
    return (x + 2 * np.pi) % (2 * np.pi)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radians to (-π, π]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


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
# Orientation gauge helpers
# ============================================================

def reflect_angles_about_axis(f: np.ndarray, ref_angle: float) -> np.ndarray:
    """
    Apply reflection r(ref_angle) to angles in radians.

    If an angle alpha represents the unit vector (cos alpha, sin alpha), then reflecting
    across the axis at angle 'ref_angle' sends:
        alpha  ->  2*ref_angle - alpha    (mod 2π).
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
    Given an orienting potential phi_pm1 with entries ±1, apply the gauge that sends
    charts with phi=-1 through the reflection r(ref_angle).

    Convention:
      g_j = I if phi_j=+1, and g_j = r(ref_angle) if phi_j=-1.

    For angles f_j, this means:
      f'_j = f_j                 if phi_j=+1
      f'_j = 2*ref_angle - f_j   (mod 2π)  if phi_j=-1
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
# Gauge solvers: Singer / tree / PU (radians)
# ============================================================

def build_edge_lookup(edges: List[Edge]) -> Dict[Tuple[int, int], int]:
    """
    edges: list of canonical undirected edges (j<k)

    Returns:
      idx[(j,k)] = r and idx[(k,j)] = r (same row index)
    """
    idx: Dict[Tuple[int, int], int] = {}
    for r, (a, b) in enumerate(edges):
        j, k = canon_edge(int(a), int(b))
        idx[(j, k)] = r
        idx[(k, j)] = r
    return idx


def theta_dir_from_canonical(theta_edge: np.ndarray, idx: Dict[Tuple[int, int], int], a: int, b: int) -> float:
    """
    Directed theta_{ab} in radians such that:

    We store canonical theta_{jk} for j<k as theta_edge[r] = theta_{jk}.
    Define directed theta:
        theta_{jk} = +theta_{jk}  if j<k
        theta_{kj} = -theta_{jk}  if k>j
    """
    r = idx[(a, b)]
    if a < b:
        return float(theta_edge[r])
    else:
        return -float(theta_edge[r])


def mu_vertices_from_singer_radians(
    *,
    edges: List[Edge],          # canonical edges j<k
    theta_edge: np.ndarray,     # (E,) radians
    n_vertices: int,
    normalize_degree: bool = True,
) -> np.ndarray:
    """
    Vertex gauge mu_v (radians) by spectral synchronization.
    """
    edges = [canon_edge_tuple(e) for e in edges]
    theta_edge = np.asarray(theta_edge, dtype=float).reshape(-1)
    if theta_edge.shape != (len(edges),):
        raise ValueError(f"theta_edge must have shape {(len(edges),)}, got {theta_edge.shape}")

    idx = build_edge_lookup(edges)
    H = np.zeros((n_vertices, n_vertices), dtype=np.complex128)
    deg = np.zeros(n_vertices, dtype=float)

    for (j, k) in edges:
        th_jk = float(theta_edge[idx[(j, k)]])
        z = np.cos(th_jk) + 1j * np.sin(th_jk)  # exp(i th)
        H[j, k] = np.conjugate(z)  # exp(-i th)
        H[k, j] = z                # exp(+i th)
        deg[j] += 1.0
        deg[k] += 1.0

    if normalize_degree:
        inv_sqrt = np.zeros_like(deg)
        mask = deg > 0
        inv_sqrt[mask] = 1.0 / np.sqrt(deg[mask])
        H = (inv_sqrt[:, None] * H) * inv_sqrt[None, :]

    _, V = np.linalg.eigh(H)
    v = V[:, -1]
    v = v / (np.linalg.norm(v) + 1e-15)

    return wrap_angle_rad(np.angle(v))


def mu_vertices_from_spanning_tree_radians(
    *,
    edges: List[Edge],
    theta_edge: np.ndarray,
    n_vertices: int,
    root: int = 0,
) -> np.ndarray:
    """
    Deterministic gauge solve on a spanning tree / DFS traversal.
    """
    edges = [canon_edge_tuple(e) for e in edges]
    theta_edge = np.asarray(theta_edge, dtype=float).reshape(-1)
    if theta_edge.shape != (len(edges),):
        raise ValueError(f"theta_edge must have shape {(len(edges),)}, got {theta_edge.shape}")

    idx = build_edge_lookup(edges)

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_vertices)]
    for (j, k) in edges:
        r = idx[(j, k)]
        th = float(theta_edge[r])
        adj[j].append((k, th))
        adj[k].append((j, -th))

    mu = np.full((n_vertices,), np.nan, dtype=float)
    mu[root] = 0.0
    stack = [root]

    while stack:
        u = stack.pop()
        for (v, delta_uv) in adj[u]:
            if np.isnan(mu[v]):
                mu[v] = mu[u] + delta_uv
                stack.append(v)

    mu = np.where(np.isnan(mu), 0.0, mu)
    return wrap_angle_rad(mu)


def mu_from_partition_unity_radians(
    *,
    edges: List[Edge],
    theta_edge: np.ndarray,
    U: np.ndarray,
    pou: np.ndarray,
    beta_edge: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pointwise gauge mu[j,m] using partition of unity.
    """
    edges = [canon_edge_tuple(e) for e in edges]
    theta_edge = np.asarray(theta_edge, dtype=float).reshape(-1)
    if theta_edge.shape != (len(edges),):
        raise ValueError(f"theta_edge must have shape {(len(edges),)}, got {theta_edge.shape}")

    n, M = pou.shape
    U = np.asarray(U)
    if U.shape != (n, M):
        raise ValueError(f"U must have shape {(n,M)}, got {U.shape}")

    if beta_edge is None:
        beta_edge = np.zeros_like(theta_edge)
    beta_edge = np.asarray(beta_edge, dtype=float).reshape(-1)
    if beta_edge.shape != (len(edges),):
        raise ValueError(f"beta_edge must have shape {(len(edges),)}, got {beta_edge.shape}")

    idx = build_edge_lookup(edges)
    mu = np.zeros((n, M), dtype=float)

    edge_index = -np.ones((n, n), dtype=int)
    for (a, b), r in idx.items():
        edge_index[a, b] = r

    theta_eff = theta_edge - beta_edge

    for j in range(n):
        for m in range(M):
            if U[j, m] == 0:
                continue
            acc = 0.0
            for k in range(n):
                r = edge_index[j, k]
                if r < 0:
                    continue
                th_jk = theta_dir_from_canonical(theta_eff, idx, j, k)
                acc += pou[k, m] * th_jk
            mu[j, m] = -acc

    mu *= (U > 0)
    return mu


# ============================================================
# Combine: apply gauge + Fréchet mean
# ============================================================

def global_from_mu(
    *,
    f: np.ndarray,        # (n,M) radians
    mu: np.ndarray,       # (n,M) radians
    U: np.ndarray,
    pou: np.ndarray,
) -> np.ndarray:
    """
    Build global angle F[m] as weighted Fréchet mean of gauged locals:
        g_j = f_j + mu_j  (mod 2π).
    """
    g = wrap_angle_rad(f + mu)
    w = pou * (U > 0)
    return frechet_mean_circle(g, w)


def build_global_trivialization(
    *,
    edges: List[Edge],
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    theta_edge: np.ndarray,
    method: str = "singer",                 # "singer" | "tree" | "pu"
    beta_edge: Optional[np.ndarray] = None, # (E,) radians optional (for "pu")
    singer_normalize_degree: bool = True,
    tree_root: int = 0,
) -> np.ndarray:
    """
    Main entrypoint: returns global trivialization F (M,) radians.

    IMPORTANT: assumes you have already oriented the cocycle if needed
    (det=+1 on the edge set used), and f has been gauge-reflected accordingly.
    """
    n, M = pou.shape
    U = np.asarray(U)
    if U.shape != (n, M):
        raise ValueError(f"U must have shape {(n,M)}, got {U.shape}")

    edges = [canon_edge_tuple(e) for e in edges]
    theta_edge = np.asarray(theta_edge, dtype=float).reshape(-1)
    if theta_edge.shape != (len(edges),):
        raise ValueError(f"theta_edge must have shape {(len(edges),)}, got {theta_edge.shape}")

    if method == "singer":
        mu_v = mu_vertices_from_singer_radians(
            edges=edges,
            theta_edge=theta_edge,
            n_vertices=n,
            normalize_degree=singer_normalize_degree,
        )
        mu = mu_v[:, None] * np.ones_like(pou, dtype=float)
        mu *= (U > 0)
        return global_from_mu(f=f, mu=mu, U=U, pou=pou)

    if method == "tree":
        mu_v = mu_vertices_from_spanning_tree_radians(
            edges=edges,
            theta_edge=theta_edge,
            n_vertices=n,
            root=int(tree_root),
        )
        mu = mu_v[:, None] * np.ones_like(pou, dtype=float)
        mu *= (U > 0)
        return global_from_mu(f=f, mu=mu, U=U, pou=pou)

    if method == "pu":
        mu = mu_from_partition_unity_radians(
            edges=edges,
            theta_edge=theta_edge,
            U=U,
            pou=pou,
            beta_edge=beta_edge,
        )
        return global_from_mu(f=f, mu=mu, U=U, pou=pou)

    raise ValueError("method must be 'singer', 'tree', or 'pu'")


# ============================================================
# Results containers
# ============================================================

@dataclass
class GlobalTrivializationResult:
    """
    Result container for a computed global circle coordinate (global trivialization).

    This object summarizes the output of a global gauge-fixing / synchronization step
    that combines local circular coordinates across a cover into a single global angle
    function on the dataset.

    Attributes
    ----------
    method : str
        Name of the method used to compute the gauge / trivialization. Typical values:
        - ``"singer"`` : spectral synchronization on the 1-skeleton
        - ``"tree"``   : spanning-tree / DFS propagation
        - ``"pu"``     : partition-of-unity pointwise gauge

    edges_used : list[Edge]
        The list of edges (cover-set adjacencies) used to define the gauge constraints.
        Edges are typically canonicalized (j < k) unless otherwise stated by the caller.

    F : ndarray of shape (n_samples,)
        The resulting global angle in radians, wrapped to the range [0, 2π).
        This is the global circle-valued coordinate defined on samples in the union
        of the cover.

    meta : dict[str, Any]
        Additional method-dependent metadata (e.g. solver options, diagnostics, or
        intermediate quantities used to construct ``F``).

    Notes
    -----
    - This result is primarily intended for inspection and downstream visualization.
    - In non-oriented settings (O(2)-bundles), callers typically first apply an
      orientation gauge so that the transitions used here have determinant +1.
    """
    method: str
    edges_used: List[Edge]
    F: np.ndarray
    meta: Dict[str, Any]
