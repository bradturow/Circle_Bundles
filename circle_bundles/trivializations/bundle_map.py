# circle_bundles/bundle_map.py
"""
bundle_map.py

Global bundle coordinatization pipeline.

This version includes:
- optional frame packing via vertex-coloring of the overlap graph ("coloring" packing)
- the PSC/subspace-PCA "frame reduction" hook (delegated to frame_reduction.py)
- the IMPORTANT gluing fix: anchored principal-branch lift for angle averaging
  (chart-equivariant; avoids π-flips caused by largest-gap branch cuts)

Core idea for gluing:
For each sample s and each active chart j, define
    F_j(s) = Phi_true[j,s] @ unit(mean_theta_j(s))
where mean_theta_j is computed by transporting local angles f_k(s) into chart j,
then taking an *anchored* weighted mean (strict semicircle by default).

If semicircle hypothesis holds, these chart formulas glue exactly on overlaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from ..analysis.class_persistence import (
    PersistenceResult,
    SubcomplexMode,
    _edges_for_subcomplex_from_persistence,
)
from ..nerve.combinatorics import Edge, canon_edge
from ..reduction.frame_reduction import (
    FrameReducerConfig,
    FrameReductionReport,
    reduce_frames_psc,
    reduce_frames_subspace_pca,
)
from ..o2_cocycle import angles_to_unit

Mat2 = np.ndarray
SummaryMode = Literal["auto", "text", "latex", "both"]

# packing mode for the frame-stacking construction
FramePacking = Literal["none", "coloring", "coloring2"]

__all__ = [
    # main pipeline
    "get_bundle_map",

    # reporting + results
    "FramePacking",
    "BundleMapReport",
    "TrueFramesResult",
    "CocycleProjectionResult",
    "ChartDisagreementStats",
    "chart_disagreement_stats",
    "FrameDataset",
    "get_frame_dataset",
    "show_bundle_map_summary",
    "get_bundle_map_v2",
]


# ============================================================
# Angle utilities (anchored strict semicircle averaging)
# ============================================================

def _wrap_to_pi(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


def _circle_dist(a: float, b: float) -> float:
    """Geodesic distance on S^1 between angles a,b (radians)."""
    return abs(_wrap_to_pi(a - b))


def _geo_to_chordal(d_geo: np.ndarray) -> np.ndarray:
    """Chordal distance in R^2 for S^1 geodesic angle: d_C = 2 sin(d_geo/2)."""
    return 2.0 * np.sin(d_geo / 2.0)


def _chordal_to_geo(d_c: float) -> float:
    """Inverse of d_C = 2 sin(d_geo/2), returning d_geo in [0, pi]."""
    x = float(np.clip(float(d_c) / 2.0, 0.0, 1.0))
    return float(2.0 * np.arcsin(x))


def weighted_angle_mean_anchored(
    angles: np.ndarray,
    weights: np.ndarray,
    *,
    anchor: float,
    tol: float = 1e-8,
) -> float:
    """
    Weighted mean of angles using a principal-branch lift anchored at `anchor`.

    Lift each angle to be within (-pi, pi] of anchor:
        diff_k = wrap_to_pi(a_k - anchor)
        a_lift_k = anchor + diff_k

    Anchored semicircle test:
        max_k |diff_k| < pi  (up to tol)

    Returns:
      mean angle in (-pi, pi]
    """
    angles = np.asarray(angles, dtype=float).reshape(-1) % (2.0 * np.pi)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if angles.shape != w.shape:
        raise ValueError("angles and weights must have the same shape.")
    if angles.size == 0:
        raise ValueError("Need at least one angle to average.")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative.")
    ws = float(w.sum())
    if ws <= 0:
        raise ValueError("weights must sum to a positive value.")
    w = w / ws

    anchor = float(anchor) % (2.0 * np.pi)

    diffs = ((angles - anchor + np.pi) % (2.0 * np.pi)) - np.pi  # (-pi,pi]
    max_abs = float(np.max(np.abs(diffs)))
    if max_abs >= np.pi - float(tol):
        raise ValueError(
            "Angles are not contained in an open semicircle around the anchor "
            f"(max_abs={max_abs:.6f} >= pi)."
        )

    lifted = anchor + diffs
    mean = float(np.sum(w * lifted))
    return _wrap_to_pi(mean)


# ============================================================
# Overlap-graph coloring (vertex coloring)
# ============================================================

def greedy_graph_coloring(
    *,
    n_sets: int,
    edges: Iterable[Edge],
) -> Tuple[np.ndarray, int]:
    """
    Greedy proper coloring of the overlap graph (1-skeleton).

    Vertices: charts 0..n_sets-1
    Edges: (j,k) if U_j overlaps U_k

    Returns:
      colors: (n_sets,) int array with values 0..K-1
      K: number of colors used

    Notes:
    - We do NOT attempt to compute the exact chromatic number (NP-hard).
    - We order vertices by descending degree to improve greedy performance.
    """
    n_sets = int(n_sets)
    if n_sets < 0:
        raise ValueError("n_sets must be nonnegative.")

    adj: List[Set[int]] = [set() for _ in range(n_sets)]
    for (a, b) in edges:
        a, b = canon_edge(int(a), int(b))
        if a == b:
            continue
        if not (0 <= a < n_sets and 0 <= b < n_sets):
            raise ValueError(f"Edge {(a,b)} out of range for n_sets={n_sets}.")
        adj[a].add(b)
        adj[b].add(a)

    deg = np.array([len(adj[i]) for i in range(n_sets)], dtype=int)
    order = list(np.argsort(-deg))  # descending degree

    colors = -np.ones(n_sets, dtype=int)
    for v in order:
        used = {colors[u] for u in adj[v] if colors[u] >= 0}
        c = 0
        while c in used:
            c += 1
        colors[v] = c

    K = int(colors.max()) + 1 if n_sets > 0 else 0
    return colors, K

def edges_power_k(
    *,
    n_sets: int,
    edges: Iterable[Edge],
    k: int,
) -> List[Edge]:
    """
    Return edge list for the k-th power graph G^k of the overlap graph G.

    G has vertices 0..n_sets-1 and undirected edges given by `edges`.
    In G^k, we connect (i,j) iff graph distance in G is <= k.

    For k=1: returns the original edges (canonized).
    For k=2: connects nodes with a common neighbor (distance 2), in addition to original edges.
    """
    n_sets = int(n_sets)
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")

    # adjacency in G
    adj: List[Set[int]] = [set() for _ in range(n_sets)]
    for (a, b) in edges:
        a, b = canon_edge(int(a), int(b))
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)

    out: Set[Edge] = set()

    # BFS from each node up to depth k; add edges to all reached nodes
    for src in range(n_sets):
        # frontier BFS
        seen = {src}
        frontier = {src}
        for _ in range(k):
            nxt: Set[int] = set()
            for u in frontier:
                nxt.update(adj[u])
            nxt -= seen
            seen |= nxt
            frontier = nxt
            if not frontier:
                break

        # seen now contains nodes within distance <=k (including src)
        for dst in seen:
            if dst == src:
                continue
            out.add(canon_edge(src, dst))

    return sorted(out)


# ============================================================
# Linear algebra helpers (rank-2 projector + Stiefel polar)
# ============================================================

def project_to_rank2_projection(A: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    As = (A + A^T)/2
    Take top-2 eigenspace of As, return P = V2 V2^T and dist = ||As - P||_F.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")
    As = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(As)  # ascending
    order = np.argsort(vals)[::-1]
    V2 = vecs[:, order[:2]]  # (d,2)
    P = V2 @ V2.T
    dist = float(np.linalg.norm(As - P, ord="fro"))
    return P, dist


def polar_stiefel_projection(P: np.ndarray, A: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Given projection P (d,d) and A (d,2), return polar factor of PA:
      U = (PA) ((PA)^T (PA))^{-1/2} ∈ V(2,d)
    """
    P = np.asarray(P, dtype=float)
    A = np.asarray(A, dtype=float)

    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square.")
    d = P.shape[0]
    if A.shape != (d, 2):
        raise ValueError(f"A must have shape {(d,2)}; got {A.shape}.")

    PA = P @ A
    AtA = PA.T @ PA
    eigvals, eigvecs = np.linalg.eigh(AtA)
    eigvals = np.clip(eigvals, eps, None)
    inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return PA @ inv_sqrt


# ============================================================
# Core construction pieces
# ============================================================

def infer_edges_from_U(U: np.ndarray) -> List[Edge]:
    """Edges (j,k) where U_j ∩ U_k contains at least one sample."""
    U = np.asarray(U, dtype=bool)
    n_sets, _ = U.shape
    E: List[Edge] = []
    for j in range(n_sets):
        for k in range(j + 1, n_sets):
            if np.any(U[j] & U[k]):
                E.append((j, k))
    return E


def get_local_frames(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Iterable[Edge],
    packing: FramePacking = "none",
    colors: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build approximate local frames Phi[j,s] ∈ R^{D×2}.

    packing="none":
        D = 2*n_sets, each chart i gets its own 2D block.

    packing="coloring":
        D = 2*K where K is the number of colors in a proper coloring of the overlap graph.
        Chart i is assigned slot colors[i] and its block is placed into that slot.

    Convention:
      Omega[(a,b)] for a<b maps b -> a.
      Directed Omega_{i<-j} is:
        - I if i==j
        - Omega[(i,j)] if i<j
        - Omega[(j,i)]^T if i>j
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou must have same shape as U.")

    edge_set = {canon_edge(a, b) for (a, b) in edges if a != b}
    sqrt_pou = np.sqrt(np.clip(pou, 0.0, None))

    # Directed Omega lookup: Omega_lookup[i,j] = Omega_{i<-j}
    Omega_lookup = np.zeros((n_sets, n_sets, 2, 2), dtype=float)
    for i in range(n_sets):
        Omega_lookup[i, i] = np.eye(2)

    for (a, b) in edge_set:
        if (a, b) not in Omega:
            raise ValueError(f"Omega missing edge {(a,b)} required by edges input.")
        Oab = np.asarray(Omega[(a, b)], dtype=float)
        if Oab.shape != (2, 2):
            raise ValueError(f"Omega[{(a,b)}] must be 2x2.")
        Omega_lookup[a, b] = Oab
        Omega_lookup[b, a] = Oab.T

    # Slot assignment
    if packing == "none":
        K = n_sets
        slot_of = np.arange(n_sets, dtype=int)
    elif packing in {"coloring", "coloring2"}:
        if colors is None:
            raise ValueError("colors must be provided when packing in {'coloring','coloring2'}.")
        colors = np.asarray(colors, dtype=int).reshape(-1)
        if colors.shape != (n_sets,):
            raise ValueError("colors must have shape (n_sets,).")
        if np.any(colors < 0):
            raise ValueError("colors must be nonnegative.")
        K = int(colors.max()) + 1 if n_sets > 0 else 0
        slot_of = colors
    else:
        raise ValueError(f"Unknown packing: {packing!r}")

    D = 2 * K
    Phi = np.zeros((n_sets, n_samples, D, 2), dtype=float)

    for j in range(n_sets):
        mask_j = U[j].astype(float)[:, None, None]  # (n_samples,1,1)
        for i in range(n_sets):
            slot = int(slot_of[i])
            r0 = 2 * slot
            r1 = r0 + 2

            Oij = Omega_lookup[i, j][None, :, :]          # (1,2,2)
            w = sqrt_pou[i, :, None, None]                # (n_samples,1,1)

            # Use += for robustness; with a proper coloring, there are no collisions at any b.
            Phi[j, :, r0:r1, :] += (w * Oij) * mask_j

    return Phi


def get_classifying_map(
    Phi: np.ndarray,
    pou: np.ndarray,
    *,
    project: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    cl[s] = sum_j pou[j,s] * Phi[j,s] Phi[j,s]^T  (D×D).

    If project=True, replace by nearest rank-2 projection via top-2 eigenspace.

    Returns:
      P: (n_samples, D, D) if project=True, else cl
      cl_proj_dist: sup_s || sym(cl[s]) - P[s] ||_F   (0 if project=False)
    """
    Phi = np.asarray(Phi, dtype=float)
    pou = np.asarray(pou, dtype=float)

    n_sets, n_samples, D, two = Phi.shape
    if two != 2:
        raise ValueError("Phi must have last dim = 2.")
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch.")

    cl = np.zeros((n_samples, D, D), dtype=float)
    for s in range(n_samples):
        for j in range(n_sets):
            w = float(pou[j, s])
            if w == 0.0:
                continue
            A = Phi[j, s]  # (D,2)
            cl[s] += w * (A @ A.T)

    if not project:
        return cl, 0.0

    P = np.zeros_like(cl)
    sup_dist = 0.0
    for s in range(n_samples):
        Ps, dist = project_to_rank2_projection(cl[s])
        P[s] = Ps
        sup_dist = max(sup_dist, dist)
    return P, float(sup_dist)

# ============================================================
# Steps 3–4 (paper): classifying map + cocycle projection
# ============================================================

@dataclass
class CocycleProjectionResult:
    """
    Cached outputs for the *projection* stage of the bundle-map pipeline (paper Steps 3–4).

    This container packages the intermediate objects produced by projecting stacked local
    frames onto a rank-2 plane (Grassmann) and then onto the Stiefel manifold, along with
    the resulting pointwise O(2) cocycle.

    Attributes
    ----------
    cl : np.ndarray
        Rank-2 projector field (Grassmann-valued classifying map), shape
        ``(n_samples, D, D)``. Each ``cl[s]`` is (approximately) a rank-2 orthogonal
        projection matrix in the ambient dimension ``D``.

    Phi_true : np.ndarray
        Projected Stiefel frames, shape ``(n_sets, n_samples, D, 2)``.
        For each chart ``j`` and sample ``s`` with ``U[j,s]=True``, ``Phi_true[j,s]``
        is an (approximately) orthonormal 2-frame spanning the projected plane ``cl[s]``.

    Omega_true : Dict[Edge, np.ndarray]
        Pointwise transition maps induced by ``Phi_true``, stored on canonical edges
        ``(j,k)`` with ``j<k``. Each entry has shape ``(n_samples, 2, 2)`` and satisfies
        (approximately)
        ``Omega_true[(j,k)][s] = Phi_true[j,s].T @ Phi_true[k,s]``.

        Interpretation:
        ``Omega_true[(j,k)][s]`` maps *k-coordinates to j-coordinates* at sample ``s``.

    cl_proj_dist : float
        Supremum (over samples) of the Frobenius error between the symmetrized raw
        classifying matrix and its nearest rank-2 projector:
        ``sup_s || sym(C(s)) - Π_Gr(C(s)) ||_F``.

    frame_proj_dist : float
        Supremum (over active chart/sample pairs) of the Frobenius distance from the raw
        frame to its Stiefel projection (polar factor), i.e.
        ``sup_{j,s:U[j,s]=1} || Phi_true[j,s] - Phi_raw[j,s] ||_F``.
    """
    cl: np.ndarray                     # (n_samples, D, D) rank-2 projectors
    Phi_true: np.ndarray               # (n_sets, n_samples, D, 2) projected Stiefel frames
    Omega_true: Dict[Edge, np.ndarray] # edges -> (n_samples,2,2)
    cl_proj_dist: float
    frame_proj_dist: float


def cocycle_project_from_raw_frames(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    Phi: np.ndarray,
    edges: Iterable[Edge],
) -> CocycleProjectionResult:
    """
    Input:
      Phi: (n_sets,n_samples,D,2) "raw" frames (your \hat{Psi}_j, possibly already reduced to d)

    Output:
      rank-2 projector field cl, projected Stiefel frames Phi_true, and cocycle Omega_true.
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    Phi = np.asarray(Phi, dtype=float)

    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch with U.")
    if Phi.shape[:2] != (n_sets, n_samples) or Phi.shape[3] != 2:
        raise ValueError("Phi must have shape (n_sets,n_samples,D,2).")

    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    # Step 3: classifying map (weighted projector sum) + rank-2 projection
    P, cl_proj_dist = get_classifying_map(Phi, pou, project=True)

    # Step 4: Stiefel projection + cocycle projection
    Phi_true, frame_proj_dist = project_frames_to_stiefel(U=U, Phi=Phi, P=P)
    Omega_true = cocycle_from_frames(Phi_true=Phi_true, edges=edges_list)

    return CocycleProjectionResult(
        cl=P,
        Phi_true=Phi_true,
        Omega_true=Omega_true,
        cl_proj_dist=float(cl_proj_dist),
        frame_proj_dist=float(frame_proj_dist),
    )


def project_frames_to_stiefel(
    *,
    U: np.ndarray,
    Phi: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Project Phi[j,s] onto V(2,D) over the plane P[s] via polar factor.

    Returns:
      Phi_true: (n_sets,n_samples,D,2)
      frame_proj_dist: sup_{j,s: U[j,s]=1} || Phi_true[j,s] - Phi[j,s] ||_F
    """
    U = np.asarray(U, dtype=bool)
    Phi = np.asarray(Phi, dtype=float)
    P = np.asarray(P, dtype=float)

    n_sets, n_samples, D, _ = Phi.shape
    if P.shape != (n_samples, D, D):
        raise ValueError("P shape mismatch with Phi.")

    out = np.zeros_like(Phi)
    sup_dist = 0.0
    for j in range(n_sets):
        for s in range(n_samples):
            if not U[j, s]:
                continue
            A = Phi[j, s]
            Uproj = polar_stiefel_projection(P[s], A)
            out[j, s] = Uproj
            sup_dist = max(sup_dist, float(np.linalg.norm(Uproj - A, ord="fro")))
    return out, float(sup_dist)


def cocycle_from_frames(
    *,
    Phi_true: np.ndarray,
    edges: Iterable[Edge],
) -> Dict[Edge, np.ndarray]:
    """
    Omega_true[(j,k)][s] = Phi_true[j,s]^T Phi_true[k,s] (2×2), for each undirected edge j<k.
    """
    Phi_true = np.asarray(Phi_true, dtype=float)
    _, n_samples, _, _ = Phi_true.shape

    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})
    Omega_true: Dict[Edge, np.ndarray] = {}

    for (j, k) in E:
        arr = np.zeros((n_samples, 2, 2), dtype=float)
        for s in range(n_samples):
            arr[s] = Phi_true[j, s].T @ Phi_true[k, s]
        Omega_true[(j, k)] = arr

    return Omega_true


def cocycle_projection_distance(
    *,
    U: np.ndarray,
    Omega_simplicial: Dict[Edge, Mat2],
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
) -> float:
    """
    d_\infty(Ω, Π(Ω)) := sup_{(jk)} sup_{b in U_j∩U_k} || Ω_{jk} - Π(Ω)_{jk}(b) ||_F.

    (Here Π(Ω) is realized as Omega_true[(j,k)][s] built from the projected Stiefel frames.)
    """
    U = np.asarray(U, dtype=bool)
    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    sup_diff = 0.0
    for (j, k) in E:
        if (j, k) not in Omega_simplicial or (j, k) not in Omega_true:
            continue
        O = np.asarray(Omega_simplicial[(j, k)], dtype=float)
        Ot = Omega_true[(j, k)]
        overlap = U[j] & U[k]
        idx = np.where(overlap)[0]
        for s in idx:
            sup_diff = max(sup_diff, float(np.linalg.norm(O - Ot[s], ord="fro")))
    return float(sup_diff)


def witness_error_stats(
    *,
    U: np.ndarray,
    f: np.ndarray,
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Trivialization error statistics computed using Π(Ω) (i.e., Omega_true):

    For each overlap (j,k) and each sample s with b_s in U_j ∩ U_k, compare:
      - unit(f_j(s))   vs   Π(Ω)_{j<-k}(b_s) unit(f_k(s)).

    Returns:
      (eps_geo_sup, eps_geo_mean, eps_C_sup, eps_C_mean)
    """
    U = np.asarray(U, dtype=bool)
    f = np.asarray(f, dtype=float)
    n_sets, n_samples = U.shape
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch with U.")

    E = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    geo_sup = 0.0
    c_sup = 0.0
    geo_sum = 0.0
    c_sum = 0.0
    n = 0
    got_any = False

    def omega_dir(j: int, k: int, s: int) -> Optional[np.ndarray]:
        """Π(Ω)_{j<-k}(b_s) from undirected store (a<b)."""
        if j == k:
            return np.eye(2)
        a, b = canon_edge(j, k)
        if (a, b) not in Omega_true:
            return None
        Oab = Omega_true[(a, b)][s]
        return Oab if (j == a and k == b) else Oab.T

    for (j, k) in E:
        overlap = U[j] & U[k]
        idx = np.where(overlap)[0]
        if idx.size == 0:
            continue

        for s in idx:
            O = omega_dir(j, k, int(s))
            if O is None:
                continue

            vj = angles_to_unit(np.array([f[j, s]], dtype=float))[0]  # (2,)
            vk = angles_to_unit(np.array([f[k, s]], dtype=float))[0]  # (2,)
            wk = O @ vk

            dc = float(np.linalg.norm(vj - wk))
            dgeo = _chordal_to_geo(dc)

            geo_sup = max(geo_sup, dgeo)
            c_sup = max(c_sup, dc)
            geo_sum += dgeo
            c_sum += dc
            n += 1
            got_any = True

    if (not got_any) or (n == 0):
        return None, None, None, None

    return float(geo_sup), float(geo_sum / n), float(c_sup), float(c_sum / n)


# ============================================================
# Bundle map construction
# ============================================================

def build_bundle_map(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    Phi_true: np.ndarray,
    Omega_true: Dict[Edge, np.ndarray],
    edges: Iterable[Edge],
    strict_semicircle: bool = True,
    semicircle_tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a global map F: samples -> R^D.

    Returns:
      F:     (n_samples, D)
      pre_F: (n_sets, n_samples, D) storing all F_j(s) for active charts (zeros otherwise)
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    f = np.asarray(f, dtype=float)
    Phi_true = np.asarray(Phi_true, dtype=float)

    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch.")
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch.")
    if Phi_true.shape[:2] != (n_sets, n_samples) or Phi_true.shape[3] != 2:
        raise ValueError("Phi_true must have shape (n_sets,n_samples,D,2).")

    D = int(Phi_true.shape[2])
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    def omega_dir(j: int, k: int, s: int) -> Optional[np.ndarray]:
        """Omega_{j<-k}(s) from undirected store (a<b)."""
        if j == k:
            return np.eye(2)
        a, b = canon_edge(j, k)
        if (a, b) not in Omega_true:
            return None
        Oab = Omega_true[(a, b)][s]
        return Oab if (j == a and k == b) else Oab.T

    pre_F = np.zeros((n_sets, n_samples, D), dtype=float)
    F = np.zeros((n_samples, D), dtype=float)

    for s in range(n_samples):
        active = np.where(U[:, s])[0]
        if active.size == 0:
            continue

        for j_ in active:
            j = int(j_)
            transported_angles: List[float] = []
            weights: List[float] = []

            for k in range(n_sets):
                w = float(pou[k, s])
                if w <= 0.0 or (not U[k, s]):
                    continue

                O = omega_dir(j, k, s)
                if O is None:
                    continue

                v = angles_to_unit(np.array([f[k, s]], dtype=float))[0]
                u = O @ v
                th = float(np.arctan2(u[1], u[0]))

                transported_angles.append(th)
                weights.append(w)

            if not weights:
                continue

            ang = np.asarray(transported_angles, dtype=float)
            wts = np.asarray(weights, dtype=float)

            ws = float(wts.sum())
            if ws <= 0.0:
                continue
            wts = wts / ws

            if strict_semicircle:
                idx_max = int(np.argmax(wts))
                anchor = float(ang[idx_max])
                mean_theta = weighted_angle_mean_anchored(ang, wts, anchor=anchor, tol=semicircle_tol)
            else:
                z = np.sum(wts * np.exp(1j * ang))
                mean_theta = float(np.angle(z))

            out = Phi_true[j, s] @ np.array([np.cos(mean_theta), np.sin(mean_theta)], dtype=float)
            pre_F[j, s] = out

        j0 = int(active[0])
        F[s] = pre_F[j0, s]

    return F, pre_F


# ============================================================
# Diagnostics: chart disagreement
# ============================================================

@dataclass
class ChartDisagreementStats:
    """
    Summary statistics for chart-to-chart disagreement in a glued bundle map.

    This is a lightweight diagnostic for the consistency of the per-chart formulas
    ``F_j(s)`` used during gluing. Ideally, on overlaps these agree exactly.

    Attributes
    ----------
    max : float
        The chart disagreement quantity

        ``Δ := sup_{(jk) in N(U)} sup_{s in U_j ∩ U_k} ||F_j(s) - F_k(s)||_2``,

        approximated using the discrete sample set and the computed ``pre_F`` array.
    """
    max: float

    def to_text(self, *, decimals: int = 3) -> str:
        r = int(decimals)
        return (
            "Chart disagreement:\n"
            "  Δ := sup_{(jk)∈N(U)} sup_{b∈U_j∩U_k} ||F_j(b) - F_k(b)||_2\n"
            f"  sup = {self.max:.{r}f}\n"
        )


def chart_disagreement_stats(*, U: np.ndarray, pre_F: np.ndarray) -> ChartDisagreementStats:
    """
    Compute Δ = sup_{(jk)∈N(U)} sup_{b∈U_j∩U_k} ||F_j(b) - F_k(b)||_2
    using the discrete samples / pre_F chart values.
    """
    U = np.asarray(U, dtype=bool)
    pre_F = np.asarray(pre_F, dtype=float)

    n_sets, n_samples = U.shape
    if pre_F.shape[0] != n_sets or pre_F.shape[1] != n_samples:
        raise ValueError("pre_F shape mismatch with U.")

    sup_delta = 0.0
    for s in range(n_samples):
        active = np.where(U[:, s])[0]
        if active.size <= 1:
            continue
        V = pre_F[active, s, :]
        mx = 0.0
        for a in range(V.shape[0]):
            for b in range(a + 1, V.shape[0]):
                mx = max(mx, float(np.linalg.norm(V[a] - V[b], ord=2)))
        sup_delta = max(sup_delta, mx)

    return ChartDisagreementStats(max=float(sup_delta))


# ============================================================
# Public pipeline: true frames + optional reduction
# ============================================================

@dataclass
class TrueFramesResult:
    """
    Step 2 result of the bundle-map pipeline: stacked local frames in an ambient space.

    This object stores the *raw* stacked frames ``Phi`` (paper Step 2), together with
    optional cached outputs of the subsequent projection stage (Steps 3–4). It is useful
    for debugging and for downstream tooling (frame reduction, diagnostics, etc.).

    Attributes
    ----------
    Phi : np.ndarray
        Raw stacked frames ``\\hat{\\Psi}_j(b_s)``, shape ``(n_sets, n_samples, D, 2)``.
        Entries are meaningful only where ``U[j,s]=True`` (inactive entries are typically
        zero).

    D : int
        Ambient dimension of the stacked-frame representation. For example:
        - packing="none": typically ``D = 2*n_sets``
        - packing="coloring*": typically ``D = 2*K`` where ``K`` is #colors

    packing : FramePacking
        Frame packing mode used to construct ``Phi`` (e.g. "none", "coloring", "coloring2").

    n_colors : Optional[int]
        If packing uses a graph coloring, this is the number of colors ``K`` used.
        Otherwise ``None``.

    colors : Optional[np.ndarray]
        If packing uses a graph coloring, an integer array of shape ``(n_sets,)`` giving
        each chart's assigned color/slot. Otherwise ``None``.

    Cached projection outputs (optional)
    ----------------------------
    These fields are ``None`` until the projection stage is computed (e.g. via
    ``ensure_projected_frames`` or indirectly via ``get_bundle_map``).

    Phi_true : Optional[np.ndarray]
        Projected Stiefel frames, shape ``(n_sets, n_samples, D, 2)``.

    Omega_true : Optional[Dict[Edge, np.ndarray]]
        Pointwise cocycle induced by ``Phi_true``, keyed by canonical edges ``(j,k)``
        with ``j<k`` and values of shape ``(n_samples,2,2)``.

    cl : Optional[np.ndarray]
        Rank-2 projector field (Grassmann classifying map), shape ``(n_samples, D, D)``.

    cl_proj_dist : Optional[float]
        Supremum Grassmann projection error (see ``CocycleProjectionResult.cl_proj_dist``).

    frame_proj_dist : Optional[float]
        Supremum Stiefel projection error (see ``CocycleProjectionResult.frame_proj_dist``).
    """
    Phi: np.ndarray               # (n_sets,n_samples,D,2) raw frames (hat-Psi)
    D: int                        # ambient dim
    packing: FramePacking = "none"
    n_colors: Optional[int] = None
    colors: Optional[np.ndarray] = None  # (n_sets,)

    # ---- cached Step 3–4 outputs (None until computed) ----
    Phi_true: Optional[np.ndarray] = None                 # (n_sets,n_samples,D,2)
    Omega_true: Optional[Dict[Edge, np.ndarray]] = None   # edges -> (n_samples,2,2)
    cl: Optional[np.ndarray] = None                       # (n_samples,D,D)
    cl_proj_dist: Optional[float] = None
    frame_proj_dist: Optional[float] = None


def build_true_frames(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Optional[Iterable[Edge]] = None,
    packing: FramePacking = "none",
) -> TrueFramesResult:
    """
    Step 2 (paper): build raw stacked frames \hat{Psi}_j over the sampled base points.

    Returns:
      TrueFramesResult with only Phi (= \hat{Psi}), plus packing metadata.

    IMPORTANT:
      This function does NOT compute:
        - classifying map / Grassmann projection
        - Stiefel projection (true frames)
        - Omega_true cocycle projection
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)

    if edges is None:
        edges = infer_edges_from_U(U)
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    colors: Optional[np.ndarray] = None
    n_colors: Optional[int] = None

    if packing in {"coloring", "coloring2"}:
        k = 1 if packing == "coloring" else 2
        edges_for_coloring = edges_list if k == 1 else edges_power_k(
            n_sets=int(U.shape[0]),
            edges=edges_list,
            k=k,
        )
        colors, n_colors = greedy_graph_coloring(n_sets=int(U.shape[0]), edges=edges_for_coloring)

    Phi = get_local_frames(
        U=U,
        pou=pou,
        Omega=Omega,
        edges=edges_list,
        packing=packing,
        colors=colors,
    )

    D = int(Phi.shape[2])

    return TrueFramesResult(
        Phi=Phi,
        D=D,
        packing=packing,
        n_colors=n_colors,
        colors=colors,
    )


def ensure_projected_frames(
    tf: TrueFramesResult,
    *,
    U: np.ndarray,
    pou: np.ndarray,
    edges: Iterable[Edge],
) -> TrueFramesResult:
    """
    Ensure that tf has cached Step 3–4 outputs (cl, Phi_true, Omega_true, projection diagnostics).

    If already computed, returns tf unchanged.
    Otherwise, runs cocycle_project_from_raw_frames on tf.Phi and stores results.
    """
    if tf.Phi_true is not None and tf.Omega_true is not None and tf.cl is not None:
        return tf

    proj = cocycle_project_from_raw_frames(U=U, pou=pou, Phi=tf.Phi, edges=edges)

    tf.Phi_true = proj.Phi_true
    tf.Omega_true = proj.Omega_true
    tf.cl = proj.cl
    tf.cl_proj_dist = float(proj.cl_proj_dist)
    tf.frame_proj_dist = float(proj.frame_proj_dist)
    return tf



def apply_frame_reduction(
    *,
    Phi_true: np.ndarray,
    U: np.ndarray,
    reducer: FrameReducerConfig,
) -> Tuple[np.ndarray, Optional[FrameReductionReport]]:
    """Delegate to frame_reduction.py."""
    if reducer.method == "none":
        return Phi_true, None

    if int(reducer.d) <= 0:
        raise ValueError("FrameReducerConfig.d must be set for reduction.")

    if reducer.method == "subspace_pca":
        Phi_red, rep, _B = reduce_frames_subspace_pca(
            Phi_true=Phi_true,
            U=U,
            d=int(reducer.d),
            max_frames=reducer.max_frames,
            rng_seed=reducer.rng_seed,
        )
        return Phi_red, rep

    if reducer.method == "psc":
        Phi_red, rep, _alpha = reduce_frames_psc(
            Phi_true=Phi_true,
            U=U,
            d=int(reducer.d),
            max_frames=reducer.max_frames,
            rng_seed=reducer.rng_seed,
            verbosity=int(getattr(reducer, "psc_verbosity", 0)),
        )
        return Phi_red, rep

    raise ValueError(f"Unknown reducer.method: {reducer.method}")


# ============================================================
# Frame dataset helper (for reducers / curve fitting / diagnostics)
# ============================================================

@dataclass
class FrameDataset:
    """
    Packed dataset of frames extracted from active chart/sample pairs.

    This is a utility container for downstream reducers, curve fitting, and diagnostics.
    It converts a sparse ``Phi[j,s]`` array (only meaningful where ``U[j,s]=True``) into a
    dense list of frames ``Y`` together with indices telling where each frame came from.

    Attributes
    ----------
    Y : np.ndarray
        Packed frames, shape ``(m, D, 2)``, where each row is a 2-frame in ambient
        dimension ``D``.

    idx : np.ndarray
        Integer index pairs, shape ``(m, 2)``. Row ``t`` is ``(j, s)`` meaning
        ``Y[t] = Phi[j, s]`` (at the chosen pipeline stage).

    stage : str
        Which pipeline stage the frames were taken from. Expected values include:
        - "pre_projection": raw stacked frames (Step 2)
        - "post_projection": projected Stiefel frames (Steps 3–4)
        - "post_reduction": after an optional dimensionality reduction

    D : int
        Ambient frame dimension of the stored frames.

    n_sets : int
        Number of cover sets/charts.

    n_samples : int
        Number of base samples.
    """
    Y: np.ndarray
    idx: np.ndarray
    stage: str
    D: int
    n_sets: int
    n_samples: int


def stack_active_frames(
    *,
    Phi: np.ndarray,
    U: np.ndarray,
    max_frames: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    allowed_vertices: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert Phi[j,s] (only meaningful where U[j,s]=True) into a packed dataset.

    allowed_vertices:
        If provided, only include frames whose chart index j is in this set.
        (Vertex-induced restriction coming from a chosen subcomplex.)
    """
    Phi = np.asarray(Phi, dtype=float)
    U = np.asarray(U, dtype=bool)

    n_sets, n_samples, D, two = Phi.shape
    if two != 2:
        raise ValueError("Phi must have shape (n_sets,n_samples,D,2).")
    if U.shape != (n_sets, n_samples):
        raise ValueError("U shape mismatch with Phi.")

    js, ss = np.where(U)

    if allowed_vertices is not None:
        keep_mask = np.array([int(j) in allowed_vertices for j in js], dtype=bool)
        js = js[keep_mask]
        ss = ss[keep_mask]

    idx = np.stack([js, ss], axis=1).astype(int)  # (m,2)

    if idx.shape[0] == 0:
        return np.zeros((0, D, 2), dtype=float), idx

    if (max_frames is not None) and (idx.shape[0] > int(max_frames)):
        rng = np.random.default_rng() if rng is None else rng
        keep = rng.choice(idx.shape[0], size=int(max_frames), replace=False)
        idx = idx[keep]

    Y = Phi[idx[:, 0], idx[:, 1], :, :]  # (m,D,2)
    return np.asarray(Y, dtype=float), np.asarray(idx, dtype=int)


def _vertices_from_edges(edges: Sequence[Tuple[int, int]]) -> Set[int]:
    V: Set[int] = set()
    for (a, b) in edges:
        V.add(int(a))
        V.add(int(b))
    return V


def get_frame_dataset(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Optional[Iterable[Edge]] = None,
    reducer: Optional[FrameReducerConfig] = None,
    stage: str = "post_projection",
    max_frames: Optional[int] = None,
    rng_seed: Optional[int] = None,
    subcomplex: SubcomplexMode = "full",
    persistence: Optional[PersistenceResult] = None,
    packing: FramePacking = "none",
) -> FrameDataset:
    """
    Build and return a packed frame dataset at a specified pipeline stage.

    stage ∈ {
      "pre_projection",   # raw hat-Psi frames tf.Phi
      "post_projection",  # projected Stiefel frames tf.Phi_true  (Steps 3–4)
      "post_reduction"    # reduction applied to tf.Phi_true (if reducer provided)
    }

    Notes
    -----
    - "post_projection" and "post_reduction" require running Steps 3–4 once.
      We do that via ensure_projected_frames(tf, ...), which caches results on tf.
    """
    stage = str(stage)

    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)

    if edges is None:
        edges = infer_edges_from_U(U)
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    tf = build_true_frames(U=U, pou=pou, Omega=Omega, edges=edges_list, packing=packing)

    # ------------------------------------------------------------
    # Choose Phi_stage according to stage
    # ------------------------------------------------------------
    if stage == "pre_projection":
        Phi_stage = tf.Phi

    elif stage == "post_projection":
        tf = ensure_projected_frames(tf, U=U, pou=pou, edges=edges_list)
        assert tf.Phi_true is not None
        Phi_stage = tf.Phi_true

    elif stage == "post_reduction":
        tf = ensure_projected_frames(tf, U=U, pou=pou, edges=edges_list)
        assert tf.Phi_true is not None

        if reducer is None or getattr(reducer, "method", "none") == "none":
            Phi_stage = tf.Phi_true
        else:
            Phi_stage, _rep = apply_frame_reduction(Phi_true=tf.Phi_true, U=U, reducer=reducer)

    else:
        raise ValueError(
            "stage must be one of: 'pre_projection', 'post_projection', 'post_reduction'. "
            f"Got: {stage!r}"
        )

    # ------------------------------------------------------------
    # Optional vertex-induced restriction from persistence/subcomplex
    # ------------------------------------------------------------
    allowed_vertices: Optional[Set[int]] = None
    if persistence is not None and subcomplex != "full":
        kept_edges = _edges_for_subcomplex_from_persistence(persistence, subcomplex)
        allowed_vertices = _vertices_from_edges(kept_edges)

    # ------------------------------------------------------------
    # Pack active frames into (m, D, 2)
    # ------------------------------------------------------------
    rng = None if rng_seed is None else np.random.default_rng(int(rng_seed))
    Y, idx = stack_active_frames(
        Phi=Phi_stage,
        U=U,
        max_frames=max_frames,
        rng=rng,
        allowed_vertices=allowed_vertices,
    )

    n_sets, n_samples = U.shape
    D = int(Phi_stage.shape[2])

    return FrameDataset(
        Y=Y,
        idx=idx,
        stage=stage,
        D=D,
        n_sets=int(n_sets),
        n_samples=int(n_samples),
    )


# ============================================================
# Reporting
# ============================================================

@dataclass
class BundleMapReport:
    """
    Diagnostics and summary statistics for the bundle-map coordinatization pipeline.

    This report aggregates sup/mean errors computed across overlaps and across the
    projection/reduction steps. It is intended to be printed (text) or displayed (LaTeX)
    via :func:`show_bundle_map_summary`.

    Attributes
    ----------
    cl_proj_dist : float
        Supremum Grassmann projection error from the classifying map to a rank-2 projector.

    frame_proj_dist : float
        Supremum Stiefel projection error (raw frames to projected orthonormal frames).

    cocycle_proj_dist : float
        Supremum Frobenius discrepancy between the simplicial cocycle ``Omega`` and its
        pointwise projected cocycle ``Π(Omega)`` computed from projected frames:
        ``sup_{(jk)} sup_{s in U_j ∩ U_k} ||Omega_{jk} - Π(Omega)_{jk}(s)||_F``.

    eps_triv : Optional[float]
        Supremum *chordal* trivialization error computed using ``Π(Omega)`` on overlaps.
        ``None`` if no overlap samples were available.

    eps_triv_geo : Optional[float]
        Supremum *geodesic* trivialization error in radians, derived from chordal errors.
        ``None`` if unavailable.

    eps_triv_mean : Optional[float]
        Mean chordal trivialization error over all overlap samples considered.

    eps_triv_geo_mean : Optional[float]
        Mean geodesic trivialization error (radians) over all overlap samples considered.

    chart_disagreement : Optional[float]
        Optional disagreement statistic Δ computed from the glued per-chart values
        ``pre_F``. ``None`` if not requested.

    reduction_method : Optional[str]
        Name of the dimensionality reduction method used (if any), e.g. "psc" or
        "subspace_pca".

    reduction_D_in : Optional[int]
        Input ambient dimension for reduction (if any).

    reduction_d_out : Optional[int]
        Output dimension after reduction (if any).

    eps_red : Optional[float]
        Supremum reduction projection error (if any).

    eps_red_mean : Optional[float]
        Mean reduction projection error (if any).

    packing : Optional[str]
        Frame packing mode used (if any), e.g. "none", "coloring", "coloring2".

    n_colors : Optional[int]
        If packing used a graph coloring, the number of colors/slots used.
    """
    # core diagnostics
    cl_proj_dist: float
    frame_proj_dist: float
    cocycle_proj_dist: float

    # trivialization error (computed using Π(Ω))
    eps_triv: Optional[float]
    eps_triv_geo: Optional[float]
    eps_triv_mean: Optional[float]
    eps_triv_geo_mean: Optional[float]

    # optional extras
    chart_disagreement: Optional[float] = None

    # dimensionality reduction (optional)
    reduction_method: Optional[str] = None
    reduction_D_in: Optional[int] = None
    reduction_d_out: Optional[int] = None
    eps_red: Optional[float] = None
    eps_red_mean: Optional[float] = None

    # packing info (optional)
    packing: Optional[str] = None
    n_colors: Optional[int] = None

    def has_reduction(self) -> bool:
        return self.eps_red_mean is not None or self.eps_red is not None

    def to_text(self, r: int = 3) -> str:
        IND = "  "
        LABEL_W = 30

        def _tline(label: str, content: str) -> str:
            return f"{IND}{label:<{LABEL_W}} {content}"

        def _fmt_eps(eps_c: Optional[float], eps_geo: Optional[float]) -> str:
            if eps_c is None:
                return "—"
            if eps_geo is None:
                return f"{float(eps_c):.{r}f}"
            return f"{float(eps_c):.{r}f} (geo = {float(eps_geo)/np.pi:.{r}f}π rad)"

        lines: List[str] = []
        lines.append("=== Diagnostics ===")
        lines.append(_tline("cocycle proj dist:", f"{float(self.cocycle_proj_dist):.{r}f}"))
        lines.append(_tline("trivialization error:", _fmt_eps(self.eps_triv, self.eps_triv_geo)))
        lines.append(_tline("mean triv error:", _fmt_eps(self.eps_triv_mean, self.eps_triv_geo_mean)))
        lines.append(_tline("Grassmann proj dist:", f"{float(self.cl_proj_dist):.{r}f}"))
        lines.append(_tline("Stiefel proj dist:", f"{float(self.frame_proj_dist):.{r}f}"))

        if self.chart_disagreement is not None:
            lines.append(_tline("chart disagreement:", f"{float(self.chart_disagreement):.{r}f}"))

        if self.packing is not None:
            pack = str(self.packing)
            extra = ""
            if self.n_colors is not None:
                extra = f" (colors={int(self.n_colors)})"
            lines.append(_tline("frame packing:", pack + extra))

        if self.has_reduction():
            meth = self.reduction_method or "reduction"
            Din = self.reduction_D_in
            dout = self.reduction_d_out

            lines.append("")
            lines.append("=== Dimensionality Reduction ===")
            if (Din is not None) and (dout is not None):
                lines.append(_tline("method:", f"{meth} (D={int(Din)} → d={int(dout)})"))
            elif dout is not None:
                lines.append(_tline("method:", f"{meth} (d={int(dout)})"))
            else:
                lines.append(_tline("method:", f"{meth}"))

            if self.eps_red is not None:
                lines.append(_tline("proj err (sup):", f"{float(self.eps_red):.{r}f}"))
            if self.eps_red_mean is not None:
                lines.append(_tline("proj err (mean):", f"{float(self.eps_red_mean):.{r}f}"))

        return "\n".join(lines)


def show_bundle_map_summary(
    report: BundleMapReport,
    *,
    show: bool = True,
    mode: SummaryMode = "auto",
    rounding: int = 3,
    extra_rows: Optional[List[Tuple[str, str]]] = None,  
) -> str:

    """
    Pretty summary, matching characteristic-class `show_summary`.
    """
    text = report.to_text(r=rounding)

    if not show:
        return text

    did_latex = False
    if mode in {"latex", "auto", "both"}:
        did_latex = _display_bundle_map_summary_latex(
            report,
            rounding=rounding,
            extra_rows=extra_rows,
        )
        
    if mode == "both" or mode == "text" or (mode == "auto" and not did_latex) or (mode == "latex" and not did_latex):
        print("\n" + text + "\n")

    return text



def _display_bundle_map_summary_latex(
    report: BundleMapReport,
    *,
    rounding: int = 3,
    extra_rows: Optional[List[Tuple[str, str]]] = None,  
) -> bool:
    try:
        from IPython.display import Math, display  # type: ignore
    except Exception:
        return False

    r = int(rounding)

    coord_rows: List[Tuple[str, str]] = []

    coord_rows.append(
        (
            r"\text{Cocycle projection dist.}",
            r"d_{\infty}(\Omega,\Pi(\Omega))"
            r":=\sup_{(jk)\in\mathcal{N}(\mathcal{U})}\sup_{b\in U_j\cap U_k}"
            r"\ \|\Omega_{jk}-\Pi(\Omega)_{jk}(b)\|_{F}"
            + r" = " + f"{float(report.cocycle_proj_dist):.{r}f}",
        )
    )
    
    if report.eps_triv is not None:
        geo_pi = None if report.eps_triv_geo is None else float(report.eps_triv_geo) / np.pi
        rhs = f"{float(report.eps_triv):.{r}f}"
        if geo_pi is not None:
            rhs += rf"\ \ (\varepsilon_{{\mathrm{{triv}}}}^{{\mathrm{{geo}}}} = {geo_pi:.{r}f}\pi)"
        coord_rows.append(
            (
                r"\text{Trivialization error}",
                r"\varepsilon_{\mathrm{triv}}"
                r":=\sup_{(jk)\in\mathcal{N}(\mathcal{U})}\sup_{b\in U_j\cap U_k}"
                r"\ d_{\mathbb{C}}\!\left(f_j(x),\ \Pi(\Omega)_{jk}(b)f_k(x)\right)"
                + r" = " + rhs,
            )
        )

    if report.eps_triv_mean is not None:
        geo_pi_m = None if report.eps_triv_geo_mean is None else float(report.eps_triv_geo_mean) / np.pi
        rhs = f"{float(report.eps_triv_mean):.{r}f}"
        if geo_pi_m is not None:
            rhs += rf"\ \ (\bar{{\varepsilon}}_{{\mathrm{{triv}}}}^{{\mathrm{{geo}}}} = {geo_pi_m:.{r}f}\pi)"
        coord_rows.append((r"\text{Mean triv error}", r"\bar{\varepsilon}_{\mathrm{triv}} = " + rhs))

    coord_rows.append(
        (
            r"\text{Grassmann proj. dist.}",
            r"d_{\mathrm{Gr}}"
            r":=\sup_{b\in \cup_{U_j\in\mathcal{U}}U_j}\ \left\|\operatorname{sym}(C(b))-\Pi_{\mathrm{Gr}}(C(b))\right\|_F"
            + r" = " + f"{float(report.cl_proj_dist):.{r}f}",
        )
    )
    coord_rows.append(
        (
            r"\text{Stiefel proj. dist.}",
            r"d_{\mathrm{St}}"
            r":=\sup_{U_j\in\mathcal{U}}\sup_{b\in U_j}\ \left\|\Pi_{\mathrm{St}}\!\big(\Phi_j(b)\big)-\Phi_j(b)\right\|_F"
            + r" = " + f"{float(report.frame_proj_dist):.{r}f}",
        )
    )

    if report.chart_disagreement is not None:
        coord_rows.append(
            (
                r"\text{Chart disagreement}",
                r"\Delta"
                r":=\sup_{(jk)\in\mathcal{N}(\mathcal{U})}\sup_{b\in U_j\cap U_k}\ \|F_j(b)-F_k(b)\|_2"
                + r" = " + f"{float(report.chart_disagreement):.{r}f}",
            )
        )

    if report.packing not in {None, "none"}:
        pack = str(report.packing)
        if report.n_colors is not None:
            # use math for K, keep the mode in monospace
            coord_rows.append(
                (
                    r"\text{Frame packing}",
                    rf"\texttt{{{pack}}}\quad (K={int(report.n_colors)})",
                )
            )
        else:
            coord_rows.append((r"\text{Frame packing}", rf"\texttt{{{pack}}}"))

    if extra_rows:
        coord_rows.extend(extra_rows)    
            
    red_rows: List[Tuple[str, str]] = []
    if report.has_reduction():
        meth = report.reduction_method or "reduction"
        Din = report.reduction_D_in
        dout = report.reduction_d_out

        if (Din is not None) and (dout is not None):
            red_rows.append((r"\text{Method}", rf"\texttt{{{meth}}},\ D={int(Din)}\to d={int(dout)}"))
        elif dout is not None:
            red_rows.append((r"\text{Method}", rf"\texttt{{{meth}}},\ d={int(dout)}"))
        else:
            red_rows.append((r"\text{Method}", rf"\texttt{{{meth}}}"))

        if report.eps_red is not None:
            red_rows.append(
                (
                    r"\text{Projection err. (sup)}",
                    r"\varepsilon_{\mathrm{red}}"
                    r":=\sup_{U_j\in\mathcal{U}}\sup_{b\in U_j}\ \|\Phi_j(b)-\Pi(\Phi_j(b))\|_{F}"
                    + r" = " + f"{float(report.eps_red):.{r}f}",
                )
            )
        if report.eps_red_mean is not None:
            red_rows.append((r"\text{Projection err. (mean)}", r"\bar{\varepsilon}_{\mathrm{red}} = " + f"{float(report.eps_red_mean):.{r}f}"))

    def _rows_to_aligned(rows: List[Tuple[str, str]]) -> str:
        return r"\\[3pt]".join(r"\quad " + lab + r" &:\quad " + expr for lab, expr in rows)

    latex = (
        r"\begin{aligned}"
        r"\textbf{Bundle Map} & \\[6pt]"
        + _rows_to_aligned(coord_rows)
    )

    if red_rows:
        latex += (
            r"\\[14pt]"
            r"\textbf{Dimensionality Reduction} & \\[6pt]"
            + _rows_to_aligned(red_rows)
        )

    latex += r"\end{aligned}"

    try:
        display(Math(latex))
        return True
    except Exception:
        return False



# ============================================================
# End-to-end coordinatization
# ============================================================

def get_bundle_map(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Optional[Iterable[Edge]] = None,
    strict_semicircle: bool = True,
    semicircle_tol: float = 1e-8,
    reducer: Optional[FrameReducerConfig] = None,
    show_summary: bool = True,
    compute_chart_disagreement: bool = True,
    packing: FramePacking = "none",
) -> Tuple[np.ndarray, np.ndarray, Dict[Edge, np.ndarray], np.ndarray, BundleMapReport]:
    """
    End-to-end coordinatization.

    Paper-aligned stages (default when reducer.stage == "pre_classifying"):
      Step 2: build raw frames \hat{Psi} in ambient dimension D (optionally packed)
              (optionally reduce to d via PSC/subspace-PCA)
      Step 3: classifying map + Grassmann projection
      Step 4: Stiefel projection + cocycle projection (Omega_used)
      Step 5: bundle map gluing (anchored averaging)

    If reducer.stage == "post_stiefel":
      project first in ambient D (Steps 3–4), then reduce frames and rebuild Omega.
    """
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    f = np.asarray(f, dtype=float)

    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch with U.")
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch with U.")

    if edges is None:
        edges = infer_edges_from_U(U)
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    # ------------------------------------------------------------
    # Step 2: build raw frames (and optionally pack)
    # ------------------------------------------------------------
    tf = build_true_frames(U=U, pou=pou, Omega=Omega, edges=edges_list, packing=packing)

    # Keep Step-2 metadata for reporting (independent of later branches)
    packing_used = str(tf.packing) if tf.packing is not None else None
    n_colors_used = tf.n_colors

    
    # ------------------------------------------------------------
    # Optional reduction bookkeeping
    # ------------------------------------------------------------
    reduction_report: Optional[FrameReductionReport] = None

    reducer_stage = "pre_classifying"
    if reducer is not None:
        reducer_stage = str(getattr(reducer, "stage", "pre_classifying"))

    # ------------------------------------------------------------
    # Steps 3–4: cocycle projection, with reduction either pre or post
    # ------------------------------------------------------------
    if reducer is not None and reducer.method != "none" and reducer_stage == "pre_classifying":
        # PAPER-FAITHFUL: reduce raw frames BEFORE classifying map / Stiefel projection
        Phi_red, reduction_report = apply_frame_reduction(Phi_true=tf.Phi, U=U, reducer=reducer)

        proj = cocycle_project_from_raw_frames(U=U, pou=pou, Phi=Phi_red, edges=edges_list)
        Phi_used = proj.Phi_true
        Omega_used = proj.Omega_true

        cl_proj_dist = float(proj.cl_proj_dist)
        frame_proj_dist = float(proj.frame_proj_dist)

    else:
        # Default: project first (Steps 3–4) in ambient dimension, cached into tf
        tf = ensure_projected_frames(tf, U=U, pou=pou, edges=edges_list)
        assert tf.Phi_true is not None and tf.Omega_true is not None
        Phi_used = tf.Phi_true
        Omega_used = tf.Omega_true

        cl_proj_dist = float(tf.cl_proj_dist) if tf.cl_proj_dist is not None else 0.0
        frame_proj_dist = float(tf.frame_proj_dist) if tf.frame_proj_dist is not None else 0.0

        # LEGACY/EXPERIMENTAL: optional post-stiefel reduction
        if reducer is not None and reducer.method != "none" and reducer_stage == "post_stiefel":
            Phi_used, reduction_report = apply_frame_reduction(Phi_true=Phi_used, U=U, reducer=reducer)
            Omega_used = cocycle_from_frames(Phi_true=Phi_used, edges=edges_list)

    # ------------------------------------------------------------
    # Diagnostics (computed using Omega_used)
    # ------------------------------------------------------------
    cocycle_proj_dist = cocycle_projection_distance(
        U=U, Omega_simplicial=Omega, Omega_true=Omega_used, edges=edges_list
    )

    # witness_error_stats returns:
    #   (eps_geo_sup, eps_geo_mean, eps_C_sup, eps_C_mean)
    eps_geo_sup, eps_geo_mean, eps_c_sup, eps_c_mean = witness_error_stats(
        U=U, f=f, Omega_true=Omega_used, edges=edges_list
    )

    # ------------------------------------------------------------
    # Step 5: bundle map (anchored averaging gluing)
    # ------------------------------------------------------------
    F, pre_F = build_bundle_map(
        U=U,
        pou=pou,
        f=f,
        Phi_true=Phi_used,
        Omega_true=Omega_used,
        edges=edges_list,
        strict_semicircle=bool(strict_semicircle),
        semicircle_tol=float(semicircle_tol),
    )

    # Chart disagreement (optional)
    cd_val: Optional[float] = None
    if compute_chart_disagreement:
        cd_val = float(chart_disagreement_stats(U=U, pre_F=pre_F).max)

    # ------------------------------------------------------------
    # Reduction summary fields (optional)
    # ------------------------------------------------------------
    red_method: Optional[str] = None
    red_D_in: Optional[int] = None
    red_d: Optional[int] = None
    eps_red: Optional[float] = None
    eps_red_mean: Optional[float] = None

    if reduction_report is not None:
        red_method = getattr(reduction_report, "method", None) or getattr(reduction_report, "name", None)
        red_method = None if red_method is None else str(red_method)

        red_D_in = getattr(reduction_report, "D_in", None)
        red_D_in = None if red_D_in is None else int(red_D_in)

        red_d = getattr(reduction_report, "d_out", None)
        if red_d is None:
            red_d = getattr(reduction_report, "d", None)
        red_d = None if red_d is None else int(red_d)

        eps_red_mean = getattr(reduction_report, "mean_proj_err", None)
        if eps_red_mean is None:
            eps_red_mean = getattr(reduction_report, "mean_projection_error", None)
        eps_red_mean = None if eps_red_mean is None else float(eps_red_mean)

        eps_red = getattr(reduction_report, "sup_proj_err", None)
        if eps_red is None:
            eps_red = getattr(reduction_report, "sup_projection_error", None)
        eps_red = None if eps_red is None else float(eps_red)

    # ------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------
    report = BundleMapReport(
        cl_proj_dist=float(cl_proj_dist),
        frame_proj_dist=float(frame_proj_dist),
        cocycle_proj_dist=float(cocycle_proj_dist),

        # witness stats (IMPORTANT: consistent with witness_error_stats return order)
        eps_triv=eps_c_sup,
        eps_triv_geo=eps_geo_sup,
        eps_triv_mean=eps_c_mean,
        eps_triv_geo_mean=eps_geo_mean,

        chart_disagreement=cd_val,

        reduction_method=red_method,
        reduction_D_in=red_D_in,
        reduction_d_out=red_d,
        eps_red=eps_red,
        eps_red_mean=eps_red_mean,

        packing=packing_used,
        n_colors=n_colors_used,
    )

    if show_summary:
        show_bundle_map_summary(report, show=True, mode="auto", rounding=3)

    return F, pre_F, Omega_used, Phi_used, report






def get_bundle_map_v2(
    *,
    U: np.ndarray,
    pou: np.ndarray,
    f: np.ndarray,
    Omega: Dict[Edge, Mat2],
    edges: Optional[Iterable[Edge]] = None,
    # --- new persistence interface ---
    weight: Optional[float] = None,
    # --- reduction / packing ---
    reducer: Optional[FrameReducerConfig] = None,
    packing: FramePacking = "none",
    # --- UX / diagnostics ---
    show_summary: bool = True,
    compute_chart_disagreement: bool = True,
    strict_semicircle: bool = True,  
    # --- REQUIRED: info from get_classes() ---
    # These are expected to be precomputed & stored by get_classes().
    # Your Bundle.method wrapper will supply them.
    persistence: Optional[PersistenceResult] = None,
    cocycle_subcomplex_edges: Optional[List[Edge]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[Edge, np.ndarray], np.ndarray, BundleMapReport]:
    """
    End-to-end coordinatization with simplified persistence controls.

    Persistence / restriction
    -------------------------
    The only persistence-related knob is `weight`.

    - If weight is None:
        Restrict to the maximal subcomplex on which the characteristic-class
        representatives are *cocycles*. This restriction must already be known
        from `get_classes()` and passed in via `cocycle_subcomplex_edges`.

    - If weight is not None:
        Restrict to edges with persistence weight <= weight.
        This is only allowed if the characteristic-class representatives are
        still cocycles on that subcomplex; otherwise, raise ValueError.

    Notes
    -----
    - We *recompute always* (no caching).
    - Rounding is fixed to 3 in summaries.
    - Semicircle tolerance is fixed internally.
    - Frame reduction (if any) is applied in the existing pipeline options.
    """

    # ----------------------------
    # Basic validation
    # ----------------------------
    U = np.asarray(U, dtype=bool)
    pou = np.asarray(pou, dtype=float)
    f = np.asarray(f, dtype=float)

    n_sets, n_samples = U.shape
    if pou.shape != (n_sets, n_samples):
        raise ValueError("pou shape mismatch with U.")
    if f.shape != (n_sets, n_samples):
        raise ValueError("f shape mismatch with U.")

    if edges is None:
        edges = infer_edges_from_U(U)
    edges_list = sorted({canon_edge(a, b) for (a, b) in edges if a != b})

    # ----------------------------
    # Decide subcomplex restriction (edges)
    # ----------------------------
    # We restrict *both*:
    #   - which edges are used for cocycle projection / diagnostics / gluing
    #   - which vertices/frames participate (by restricting U/pou/f to induced vertices if you want)
    #
    # For now we keep U/pou/f unchanged and only restrict the *nerve edges* used in
    # computations. (This is consistent with how your persistence helpers work.)
    #
    # `cocycle_subcomplex_edges` is the maximal-edge set where classes are cocycles.
    if cocycle_subcomplex_edges is None:
        if weight is None:
            raise ValueError(
                "cocycle_subcomplex_edges must be provided when weight=None "
                "(expected to come from get_classes())."
            )
        # If user passes a weight, we *can* derive edges from persistence,
        # but we still need the cocycle threshold check below.
    cocycle_edges_max = None if cocycle_subcomplex_edges is None else sorted(
        {canon_edge(a, b) for (a, b) in cocycle_subcomplex_edges if a != b}
    )

    if weight is None:
        # default: maximal cocycle subcomplex
        edges_used = cocycle_edges_max
        assert edges_used is not None
    else:
        if persistence is None:
            raise ValueError("persistence must be provided when weight is not None.")

        # edges under weight threshold
        w = float(weight)
        edges_under_w = _edges_for_subcomplex_from_persistence(
            persistence,
            subcomplex="edge_weight",   # you may implement/alias this mode; see note below
            weight=w,
        )

        # ---- cocycle check: weight must not exceed cocycle-valid threshold ----
        # simplest: require edges_under_w to be subset of cocycle_edges_max
        # (equivalently: w <= cocycle_threshold)
        if cocycle_edges_max is None:
            raise ValueError("cocycle_subcomplex_edges must be provided to validate weight.")
        cocycle_set = set(cocycle_edges_max)
        for e in edges_under_w:
            if canon_edge(*e) not in cocycle_set:
                raise ValueError(
                    "Invalid weight: at this threshold the characteristic classes are not cocycles "
                    "(requested subcomplex exceeds cocycle-valid subcomplex)."
                )

        edges_used = sorted({canon_edge(a, b) for (a, b) in edges_under_w if a != b})

    # Sanity: edges_used must be subset of inferred edges_list if you expect that
    # (not strictly necessary, but usually true)
    edges_used = sorted({e for e in edges_used if e in set(edges_list)})

    # ----------------------------
    # Step 2: build raw frames (packing happens here)
    # ----------------------------
    tf = build_true_frames(U=U, pou=pou, Omega=Omega, edges=edges_list, packing=packing)

    packing_used = str(tf.packing) if tf.packing is not None else None
    n_colors_used = tf.n_colors

    # ----------------------------
    # Reduction stage logic (keep as-is, but simpler defaults)
    # ----------------------------
    reduction_report: Optional[FrameReductionReport] = None

    reducer_stage = "pre_classifying"
    if reducer is not None:
        reducer_stage = str(getattr(reducer, "stage", "pre_classifying"))
        if getattr(reducer, "method", "none") == "none":
            reducer = None

    # Fixed semicircle tolerance
    _SEMICIRCLE_TOL = 1e-8

    # ----------------------------
    # Steps 3–4 (projection)
    # Use *edges_used* consistently for cocycle projection & overlap diagnostics.
    # ----------------------------
    if reducer is not None and reducer_stage == "pre_classifying":
        # reduce raw frames BEFORE classifying map / Stiefel projection
        Phi_red, reduction_report = apply_frame_reduction(Phi_true=tf.Phi, U=U, reducer=reducer)
        proj = cocycle_project_from_raw_frames(U=U, pou=pou, Phi=Phi_red, edges=edges_used)
        Phi_used = proj.Phi_true
        Omega_used = proj.Omega_true
        cl_proj_dist = float(proj.cl_proj_dist)
        frame_proj_dist = float(proj.frame_proj_dist)

    else:
        # project in ambient D
        proj = cocycle_project_from_raw_frames(U=U, pou=pou, Phi=tf.Phi, edges=edges_used)
        Phi_used = proj.Phi_true
        Omega_used = proj.Omega_true
        cl_proj_dist = float(proj.cl_proj_dist)
        frame_proj_dist = float(proj.frame_proj_dist)

        # optional post-stiefel reduction (legacy/experimental)
        if reducer is not None and reducer_stage == "post_stiefel":
            Phi_used, reduction_report = apply_frame_reduction(Phi_true=Phi_used, U=U, reducer=reducer)
            Omega_used = cocycle_from_frames(Phi_true=Phi_used, edges=edges_used)

    # ----------------------------
    # Diagnostics using edges_used
    # ----------------------------
    cocycle_proj_dist = cocycle_projection_distance(
        U=U, Omega_simplicial=Omega, Omega_true=Omega_used, edges=edges_used
    )

    eps_geo_sup, eps_geo_mean, eps_c_sup, eps_c_mean = witness_error_stats(
        U=U, f=f, Omega_true=Omega_used, edges=edges_used
    )

    # ----------------------------
    # Step 5: gluing (NO averaging across charts; pick one chart value)
    # Your build_bundle_map already picks first active chart for F[s].
    # ----------------------------
    F, pre_F = build_bundle_map(
        U=U,
        pou=pou,
        f=f,
        Phi_true=Phi_used,
        Omega_true=Omega_used,
        edges=edges_used,
        strict_semicircle=bool(strict_semicircle),
        semicircle_tol=_SEMICIRCLE_TOL,
    )

    cd_val: Optional[float] = None
    if compute_chart_disagreement:
        cd_val = float(chart_disagreement_stats(U=U, pre_F=pre_F).max)

    # ----------------------------
    # Reduction summary
    # ----------------------------
    red_method: Optional[str] = None
    red_D_in: Optional[int] = None
    red_d: Optional[int] = None
    eps_red: Optional[float] = None
    eps_red_mean: Optional[float] = None

    if reduction_report is not None:
        red_method = getattr(reduction_report, "method", None) or getattr(reduction_report, "name", None)
        red_method = None if red_method is None else str(red_method)

        red_D_in = getattr(reduction_report, "D_in", None)
        red_D_in = None if red_D_in is None else int(red_D_in)

        red_d = getattr(reduction_report, "d_out", None)
        if red_d is None:
            red_d = getattr(reduction_report, "d", None)
        red_d = None if red_d is None else int(red_d)

        eps_red_mean = getattr(reduction_report, "mean_proj_err", None)
        if eps_red_mean is None:
            eps_red_mean = getattr(reduction_report, "mean_projection_error", None)
        eps_red_mean = None if eps_red_mean is None else float(eps_red_mean)

        eps_red = getattr(reduction_report, "sup_proj_err", None)
        if eps_red is None:
            eps_red = getattr(reduction_report, "sup_projection_error", None)
        eps_red = None if eps_red is None else float(eps_red)

    report = BundleMapReport(
        cl_proj_dist=float(cl_proj_dist),
        frame_proj_dist=float(frame_proj_dist),
        cocycle_proj_dist=float(cocycle_proj_dist),

        eps_triv=eps_c_sup,
        eps_triv_geo=eps_geo_sup,
        eps_triv_mean=eps_c_mean,
        eps_triv_geo_mean=eps_geo_mean,

        chart_disagreement=cd_val,

        reduction_method=red_method,
        reduction_D_in=red_D_in,
        reduction_d_out=red_d,
        eps_red=eps_red,
        eps_red_mean=eps_red_mean,

        packing=packing_used,
        n_colors=n_colors_used,
    )

    if show_summary:
        # fixed auto + fixed rounding=3
        show_bundle_map_summary(report, show=True, mode="auto", rounding=3)

    return F, pre_F, Omega_used, Phi_used, report
