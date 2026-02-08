# circle_bundles/covers/fibonacci_covers.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from .covers import CoverData
from ..geometry.geometry import get_bary_coords

__all__ = [
    # helpers
    "fibonacci_sphere",
    "hemisphere_rep",
    # public-facing builders 
    "get_s2_fibonacci_cover",
    "get_rp2_fibonacci_cover",
]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def _normalize_rows(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array. Got shape {X.shape}.")
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm


def fibonacci_sphere(n: int) -> np.ndarray:
    """
    Nearly-uniform points on S^2 via a spherical Fibonacci lattice.

    Returns
    -------
    (n,3) float unit vectors on S^2. Deterministic for a given n.
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


def hemisphere_rep(x: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """
    Canonical representative for RP^2: pick the unit vector in the closed upper hemisphere.
    Deterministic tie-break near the equator.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError(f"Expected (3,) vector. Got {x.shape}.")
    if x[2] > atol:
        return x
    if x[2] < -atol:
        return -x
    for k in (0, 1):
        if x[k] > atol:
            return x
        if x[k] < -atol:
            return -x
    return x


# -----------------------------------------------------------------------------
# Hull triangulation + ray projection
# -----------------------------------------------------------------------------

def _build_hull(points_s2: np.ndarray) -> Tuple[ConvexHull, np.ndarray]:
    P = np.asarray(points_s2, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"points_s2 must be (n,3). Got {P.shape}.")
    if P.shape[0] < 4:
        raise ValueError("Need at least 4 non-coplanar points for a 3D convex hull.")
    hull = ConvexHull(P)
    faces = np.asarray(hull.simplices, dtype=int)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Unexpected hull.simplices shape {faces.shape}; expected (F,3).")
    return hull, faces


def _project_rays_to_hull(
    directions_s2: np.ndarray,
    hull: ConvexHull,
    *,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each unit direction x, intersect ray t x with convex hull.
    Returns hit points and facet indices.

    hull.equations rows: [n_x, n_y, n_z, d] for planes n·p + d = 0
    with outward normals and inside satisfying n·p + d <= 0.

    For direction x:
        t = min_{facets with n·x > 0} (-d)/(n·x).
    """
    X = _normalize_rows(np.asarray(directions_s2, dtype=float), eps=eps)
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"directions_s2 must be (N,3). Got {X.shape}.")

    eq = np.asarray(hull.equations, dtype=float)  # (F,4)
    n = eq[:, :3]                                 # (F,3)
    d = eq[:, 3]                                  # (F,)

    denom = n @ X.T                               # (F,N)
    num = (-d)[:, None]                           # (F,1)
    t = np.where(denom > eps, num / denom, np.inf)

    face_idx = np.argmin(t, axis=0)  # (N,)
    t_min = t[face_idx, np.arange(X.shape[0])]
    if not np.all(np.isfinite(t_min)):
        raise RuntimeError(
            "Ray projection failed for some points (no forward-intersecting facet). "
            "Hull may not contain origin or eps too large."
        )

    P_hit = X * t_min[:, None]
    return P_hit, face_idx


# -----------------------------------------------------------------------------
# S^2 Fibonacci star cover -> CoverData
# -----------------------------------------------------------------------------

def get_s2_fibonacci_cover(
    base_points: np.ndarray,
    *,
    n_vertices: int = 256,
    eps: float = 1e-12,
) -> CoverData:
    r"""
    Build a fast star cover of S^2 using Fibonacci landmarks and a convex-hull triangulation.

    Returns CoverData with:
      - U: (n_vertices, n_samples) bool
      - pou: (n_vertices, n_samples) float (barycentric weights on hit face)
      - landmarks: (n_vertices, 3) float

    Notes
    -----
    Each sample is assigned to exactly one hull face, so overlap order is ≤ 3.
    """
    P = _normalize_rows(np.asarray(base_points, dtype=float), eps=eps)
    if P.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {P.shape}.")

    m = int(n_vertices)
    V = fibonacci_sphere(m)  # (m,3)

    hull, faces = _build_hull(V)                         # faces: (F,3) vertex ids
    P_hit, hit_face_idx = _project_rays_to_hull(P, hull, eps=eps)

    N = P.shape[0]
    U = np.zeros((m, N), dtype=bool)
    pou = np.zeros((m, N), dtype=float)

    # Compute barycentric coords per hit face in batches (by face id)
    hit_face_idx = np.asarray(hit_face_idx, dtype=int)
    for f in np.unique(hit_face_idx):
        mask = hit_face_idx == f
        a, b, c = (int(x) for x in faces[int(f)])
        tri_xyz = V[[a, b, c]]                     # (3,3)
        bc = get_bary_coords(P_hit[mask], tri_xyz) # (k,3)

        # keep nonnegative and renormalize (numerical safety)
        bc = np.maximum(0.0, bc)
        s = bc.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        bc = bc / s

        idx = np.flatnonzero(mask)
        U[a, idx] = True
        U[b, idx] = True
        U[c, idx] = True
        pou[a, idx] = bc[:, 0]
        pou[b, idx] = bc[:, 1]
        pou[c, idx] = bc[:, 2]

    return CoverData(
        U=U,
        pou=pou,
        landmarks=V,
        meta={"type": "s2_fibonacci_star", "n_vertices": m},
    )


# -----------------------------------------------------------------------------
# RP^2 Fibonacci star cover -> CoverData
# -----------------------------------------------------------------------------

def _rp2_dist_chordal(a: np.ndarray, b: np.ndarray) -> float:
    """
    Chordal distance on RP^2 induced by antipodal quotient:
        d([a],[b]) = min(||a-b||, ||a+b||)
    for unit vectors a,b in R^3.
    """
    d1 = a - b
    d2 = a + b
    return float(min(np.sqrt(np.dot(d1, d1)), np.sqrt(np.dot(d2, d2))))


def _fps_indices_rp2(X: np.ndarray, k: int, *, seed: Optional[int] = None) -> np.ndarray:
    """
    Farthest point sampling on RP^2 using chordal distance.
    X should be (n,3) unit hemisphere reps.
    """
    X = _normalize_rows(np.asarray(X, float))
    n = X.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"k must be in 1..{n}. Got {k}.")
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))

    chosen = np.empty(k, dtype=int)
    chosen[0] = first

    min_d = np.array([_rp2_dist_chordal(X[i], X[first]) for i in range(n)], dtype=float)
    min_d[first] = -np.inf

    for t in range(1, k):
        nxt = int(np.argmax(min_d))
        chosen[t] = nxt
        dn = np.array([_rp2_dist_chordal(X[i], X[nxt]) for i in range(n)], dtype=float)
        min_d = np.minimum(min_d, dn)
        min_d[nxt] = -np.inf

    return chosen


def _pick_rp2_landmarks(
    k: int,
    *,
    oversample: int = 20,
    seed: Optional[int] = None,
    atol: float = 1e-12,
) -> np.ndarray:
    """
    Make k well-spread RP^2 landmark reps via:
        Fibonacci(M=k*oversample) -> hemisphere reps -> FPS in RP^2 metric.
    Returns (k,3) hemisphere reps.
    """
    M = int(max(k * oversample, k))
    cand = fibonacci_sphere(M)
    cand = np.stack([hemisphere_rep(cand[i], atol=atol) for i in range(M)], axis=0)
    cand = _normalize_rows(cand)
    idx = _fps_indices_rp2(cand, k, seed=seed)
    return cand[idx]


def get_rp2_fibonacci_cover(
    base_points: np.ndarray,
    *,
    n_pairs: int = 256,
    landmark_oversample: int = 20,
    landmark_seed: Optional[int] = 0,
    eps: float = 1e-12,
    atol: float = 1e-12,
) -> CoverData:
    r"""
    Build a star cover of RP^2 using Fibonacci landmarks, working upstairs on S^2
    and pushing down through the antipodal quotient.

    Returns CoverData with:
      - U: (n_pairs, n_samples) bool
      - pou: (n_pairs, n_samples) float
      - landmarks: (n_pairs, 3) float (hemisphere reps)

    Construction: for each sample [x], accumulate barycentric weights from BOTH lifts
    x and -x on an antipodal-symmetric hull, push vertex contributions down by pairing,
    then keep top-3 vertices per sample and renormalize.
    """
    # 1) Samples as RP^2 hemisphere reps
    P = _normalize_rows(np.asarray(base_points, float), eps=eps)
    if P.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {P.shape}.")
    P_rep = np.stack([hemisphere_rep(P[i], atol=atol) for i in range(P.shape[0])], axis=0)
    P_other = -P_rep

    # 2) RP^2 landmarks (hemisphere reps)
    n = int(n_pairs)
    L_rp2 = _pick_rp2_landmarks(
        n,
        oversample=int(landmark_oversample),
        seed=landmark_seed,
        atol=atol,
    )  # (n,3)

    # 3) Lift upstairs
    V_full = np.vstack([L_rp2, -L_rp2])  # (2n,3)

    # 4) Hull triangulation upstairs
    hull, faces_full = _build_hull(V_full)

    # 5) Ray hits for both lifts
    P_hit_a, hit_face_a = _project_rays_to_hull(P_rep, hull, eps=eps)
    P_hit_b, hit_face_b = _project_rays_to_hull(P_other, hull, eps=eps)

    # 6) Quotient map old_vid -> rp2_vid (pair i <-> i+n)
    old_to_new: Dict[int, int] = {i: i for i in range(n)}
    old_to_new.update({i + n: i for i in range(n)})

    N = P_rep.shape[0]
    U = np.zeros((n, N), dtype=bool)
    pou = np.zeros((n, N), dtype=float)

    def _accumulate_from_hit(
        acc: Dict[int, float], *, face_idx: int, p_hit_row: np.ndarray
    ) -> None:
        a, b, c = (int(x) for x in faces_full[int(face_idx)])
        tri_xyz = V_full[[a, b, c]]  # (3,3)
        bc = get_bary_coords(p_hit_row[None, :], tri_xyz)[0]  # (3,)
        bc = np.maximum(0.0, bc)

        vids = [old_to_new[a], old_to_new[b], old_to_new[c]]
        for vid, w in zip(vids, bc):
            vv = int(vid)
            acc[vv] = acc.get(vv, 0.0) + float(w)

    for s in range(N):
        acc: Dict[int, float] = {}
        _accumulate_from_hit(acc, face_idx=int(hit_face_a[s]), p_hit_row=P_hit_a[s])  # x
        _accumulate_from_hit(acc, face_idx=int(hit_face_b[s]), p_hit_row=P_hit_b[s])  # -x

        # take top-3 weights
        items = sorted(acc.items(), key=lambda t: -t[1])
        if len(items) == 0:
            items = [(0, 1.0)]
        while len(items) < 3:
            items.append((items[-1][0], 0.0))

        (v0, w0), (v1, w1), (v2, w2) = items[:3]
        ws = np.array([w0, w1, w2], dtype=float)
        ssum = float(ws.sum())
        ws = ws / ssum if ssum > 0 else np.array([1.0, 0.0, 0.0], dtype=float)

        verts = (int(v0), int(v1), int(v2))
        weights = (float(ws[0]), float(ws[1]), float(ws[2]))

        for v, w in zip(verts, weights):
            if w <= 0.0:
                continue
            U[v, s] = True
            pou[v, s] += w

        # pou already normalized by construction, but keep a safe normalization:
        colsum = float(pou[:, s].sum())
        if colsum > 0:
            pou[:, s] /= colsum
        else:
            pou[0, s] = 1.0
            U[0, s] = True

    return CoverData(
        U=U,
        pou=pou,
        landmarks=L_rp2,
        meta={"type": "rp2_fibonacci_star", "n_pairs": n},
    )
