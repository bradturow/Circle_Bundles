# circle_bundles/triangle_cover_builders_fibonacci.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from ..base_covers import CoverBase, TriangulationStarCover
from ..geometry.geometry import get_bary_coords
from ..nerve.combinatorics import Edge, Tri, canon_edge, canon_tri
from .triangle_covers import _require_gudhi  # lazy gudhi import helper

from ..metrics import RP2UnitVectorMetric

__all__ = [
    "fibonacci_sphere",
    "hemisphere_rep",
    "make_s2_fibonacci_star_cover",
    "make_rp2_fibonacci_star_cover",
    "RP2StarCover",
]


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def _normalize_rows(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return X / nrm


def fibonacci_sphere(n: int) -> np.ndarray:
    """
    Nearly-uniform points on S^2 via a spherical Fibonacci lattice.

    Returns
    -------
    L : (n,3) float
        Unit vectors on S^2. Deterministic for a given n.
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
    Tie-break deterministically near the equator.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (3,):
        raise ValueError(f"Expected (3,) vector. Got {x.shape}.")
    if x[2] > atol:
        return x
    if x[2] < -atol:
        return -x
    # near equator: deterministic tie-break by first nonzero coord
    for k in (0, 1):
        if x[k] > atol:
            return x
        if x[k] < -atol:
            return -x
    return x


# -----------------------------------------------------------------------------
# Hull triangulation + ray projection (shared)
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
    n = eq[:, :3]  # (F,3)
    d = eq[:, 3]  # (F,)

    denom = n @ X.T  # (F,N)
    num = (-d)[:, None]  # (F,1)
    t = np.where(denom > eps, num / denom, np.inf)  # (F,N)

    face_idx = np.argmin(t, axis=0)  # (N,)
    t_min = t[face_idx, np.arange(X.shape[0])]  # (N,)
    if not np.all(np.isfinite(t_min)):
        raise RuntimeError(
            "Ray projection failed for some points (no forward-intersecting facet). "
            "Hull may not contain origin or eps too large."
        )

    P_hit = X * t_min[:, None]
    return P_hit, face_idx


def _make_gudhi_triangle_complex(faces: np.ndarray):
    gd = _require_gudhi()
    K = gd.SimplexTree()
    for (a, b, c) in faces:
        K.insert([int(a), int(b), int(c)])
    return K


# -----------------------------------------------------------------------------
# S^2 star cover from Fibonacci hull (TriangulationStarCover)
# -----------------------------------------------------------------------------


def make_s2_fibonacci_star_cover(
    base_points: np.ndarray,
    *,
    n_vertices: int = 256,
    eps: float = 1e-12,
) -> TriangulationStarCover:
    r"""
    Build a fast star cover of :math:`\mathbb{S}^2` using Fibonacci landmarks and a hull triangulation.

    This constructs a triangulation of a set of nearly-uniform landmark points on the sphere
    (via a spherical Fibonacci lattice), then assigns each sample to **the unique** hull face hit
    by the ray in its direction. The resulting cover is the **open star cover** of the triangulation
    vertices: each sample lies in exactly the three vertex stars of its containing triangle
    (so overlap order is at most 3).

    The returned cover is a :class:`~circle_bundles.base_covers.TriangulationStarCover` that is
    already built (its ``U`` and ``pou`` are populated).

    Parameters
    ----------
    base_points :
        Array of shape ``(n_samples, 3)`` representing directions in :math:`\mathbb{R}^3`.
        Rows are normalized internally and interpreted as points on :math:`\mathbb{S}^2`.
    n_vertices :
        Number of Fibonacci landmark vertices used to build the hull triangulation.
        Larger values yield a finer cover (more sets, smaller stars). Default is ``256``.
    eps :
        Numerical stability parameter used for normalization and ray–facet intersection tests.

    Returns
    -------
    cover :
        A built :class:`~circle_bundles.base_covers.TriangulationStarCover` representing the star cover.
        Key fields include:

        - ``cover.base_points``: normalized samples on :math:`\mathbb{S}^2` (shape ``(n,3)``)
        - ``cover.landmarks``: landmark vertex coordinates (shape ``(n_vertices,3)``)
        - ``cover.U``: membership matrix (shape ``(n_vertices, n_samples)``)
        - ``cover.pou``: barycentric partition of unity weights (same shape as ``U``)

    Notes
    -----
    - **Overlap order** is at most 3, since each sample is assigned to one triangle and hence
      belongs to exactly three vertex stars.
    - This builder uses a “hit facet index” fast path (no searching over all triangles),
      so it scales well to large sample sizes.

    Examples
    --------
    >>> import circle_bundles as cb
    >>> cover = cb.make_s2_fibonacci_star_cover(base_points, n_vertices=512)
    >>> summ = cover.summarize(plot=False)
    >>> bundle = cb.build_bundle(total_space_data, cover, show=True)
    """
    P = _normalize_rows(np.asarray(base_points, dtype=float), eps=eps)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {P.shape}.")

    # 1) Fibonacci vertices on S^2
    V = fibonacci_sphere(int(n_vertices))  # (m,3)

    # 2) Hull triangulation
    hull, faces = _build_hull(V)

    # 3) Project each base direction to hull surface
    K_preimages, hit_face_idx = _project_rays_to_hull(P, hull, eps=eps)  # (N,3), (N,)

    # 4) Gudhi triangle complex + coords dict
    K = _make_gudhi_triangle_complex(faces)
    vertex_coords_dict: Dict[int, np.ndarray] = {int(i): V[i].copy() for i in range(V.shape[0])}

    # 5) Build the cover object
    cover = TriangulationStarCover(
        base_points=P,  # keep base points on S^2 for bookkeeping/viz
        K_preimages=K_preimages,  # points on hull surface
        K=K,
        vertex_coords_dict=vertex_coords_dict,
    )

    # --- Fast build path: we already know which triangle each sample lies in ---
    cover._relabel_vertices()  # sets vid maps (identity here) + vertex_coords + landmarks

    tris = [canon_tri(int(a), int(b), int(c)) for (a, b, c) in faces]
    cover.triangles = tris
    tri_to_idx = {t: idx for idx, t in enumerate(tris)}

    hit_tris = [canon_tri(*faces[int(f)]) for f in hit_face_idx]
    cover.sample_tri = np.array([tri_to_idx[t] for t in hit_tris], dtype=int)

    # barycentric coords in hit triangle
    N = P.shape[0]
    cover.sample_bary = np.zeros((N, 3), dtype=float)
    sample_tri = cover.sample_tri
    for t_idx in np.unique(sample_tri):
        mask = sample_tri == t_idx
        i, j, k = cover.triangles[int(t_idx)]
        tri_xyz = cover.vertex_coords[[i, j, k]]  # (3,3)
        pts = cover.K_preimages[mask]  # (m,3)
        bc = get_bary_coords(pts, tri_xyz)  # (m,3)
        s = bc.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        cover.sample_bary[mask] = bc / s

    cover._build_star_sets_U()
    cover._build_pou_from_barycentric()
    cover.ensure_metric()
    return cover


# -----------------------------------------------------------------------------
# RP^2: landmark selection (Fibonacci oversample + RP^2 FPS)
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
    Farthest point sampling on RP^2 using _rp2_dist_chordal.
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


# -----------------------------------------------------------------------------
# RP^2 star cover (witnessed Čech nerve, BOTH lifts)
# -----------------------------------------------------------------------------


class RP2StarCover(CoverBase):
    """
    RP^2 cover built from barycentric weights on hull faces, pushed down by antipodal quotient.

    Key point:
      For each RP^2 sample [x], we must consider BOTH lifts x and -x upstairs, then push down.

    Nerve:
      Witnessed Čech nerve computed from U itself (no ghost simplices).
    """

    def __init__(
        self,
        base_points: np.ndarray,  # (N,3) hemisphere reps of RP^2 samples
        landmarks: np.ndarray,  # (M,3) hemisphere reps for RP^2 vertices
        sample_vertex_triples: np.ndarray,  # (N,3) ints: top-3 RP^2 vertex ids
        sample_weights: np.ndarray,  # (N,3) weights (sum=1)
    ):
        super().__init__(base_points=np.asarray(base_points, float))
        self.landmarks = np.asarray(landmarks, float)
        self.sample_vertex_triples = np.asarray(sample_vertex_triples, dtype=int)
        self.sample_weights = np.asarray(sample_weights, dtype=float)

    def build(self) -> "RP2StarCover":
        N = self.base_points.shape[0]
        M = self.landmarks.shape[0]

        U = np.zeros((M, N), dtype=bool)
        pou = np.zeros((M, N), dtype=float)

        for s in range(N):
            vs = self.sample_vertex_triples[s]
            ws = self.sample_weights[s]
            for a, w in zip(vs, ws):
                if w <= 0.0:
                    continue
                a = int(a)
                U[a, s] = True
                pou[a, s] += float(w)

        denom = pou.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        self.U = U
        self.pou = pou / denom
        self.metric = RP2UnitVectorMetric()
        self.ensure_metric()
        return self

    def nerve_edges(self) -> List[Edge]:
        self.ensure_built()
        U = self.U
        m = U.shape[0]
        edges: set[Edge] = set()
        for i in range(m):
            Ui = U[i]
            for j in range(i + 1, m):
                if np.any(Ui & U[j]):
                    edges.add(canon_edge(i, j))
        return sorted(edges)

    def nerve_triangles(self) -> List[Tri]:
        self.ensure_built()
        U = self.U
        m = U.shape[0]
        tris: set[Tri] = set()
        for i in range(m):
            Ui = U[i]
            for j in range(i + 1, m):
                Uij = Ui & U[j]
                if not np.any(Uij):
                    continue
                for k in range(j + 1, m):
                    if np.any(Uij & U[k]):
                        tris.add(canon_tri(i, j, k))
        return sorted(tris)


def make_rp2_fibonacci_star_cover(
    base_points: np.ndarray,
    *,
    n_pairs: int = 256,
    landmark_oversample: int = 20,
    landmark_seed: Optional[int] = 0,
    eps: float = 1e-12,
    atol: float = 1e-12,
) -> RP2StarCover:
    r"""
    Build a star cover of :math:`\mathbb{RP}^2` using Fibonacci landmarks and an antipodal-symmetric hull.

    This constructs a cover downstairs on :math:`\mathbb{RP}^2` by working upstairs on :math:`\mathbb{S}^2`
    and carefully pushing the result down through the antipodal quotient.

    Pipeline (high-level)
    ---------------------
    1. Canonicalize each sample ``x`` to a *hemisphere representative* ``x_rep`` of its class ``[x]``.
    2. Choose ``n_pairs`` well-spread landmark representatives in :math:`\mathbb{RP}^2` via:
       Fibonacci oversampling + farthest point sampling (FPS) in the chordal RP² metric.
    3. Lift landmarks to :math:`\mathbb{S}^2` by adding antipodes, then triangulate via the convex hull.
    4. For each sample class ``[x]``, compute hull-face hits and barycentric weights for both lifts
       ``x_rep`` and ``-x_rep``.
    5. Push vertex contributions down to RP² vertex ids (pairing antipodes), sum weights, and keep the
       top-3 vertices per sample (renormalized).
    6. Build the cover membership matrix ``U`` and partition of unity ``pou``. The nerve is “witnessed”
       from ``U`` (no ghost simplices).

    The returned object is an :class:`~circle_bundles.triangle_cover_builders_fibonacci.RP2StarCover`
    that is already built (its ``U`` and ``pou`` are populated).

    Parameters
    ----------
    base_points :
        Array of shape ``(n_samples, 3)`` representing (approximate) unit vectors in :math:`\mathbb{R}^3`.
        Each row represents a projective class ``[x]`` with the antipodal identification ``x ~ -x``.
        Rows are normalized internally.
    n_pairs :
        Number of *projective* landmark vertices (i.e. the number of antipodal pairs).
        The upstairs hull uses ``2*n_pairs`` vertices. Default is ``256``.
    landmark_oversample :
        Oversampling factor used before RP² farthest point sampling.
        Candidates are generated with ``M = n_pairs * landmark_oversample`` Fibonacci points.
    landmark_seed :
        Random seed used only to choose the initial FPS index. FPS is otherwise deterministic.
        Set to ``None`` for nondeterministic behavior.
    eps :
        Numerical stability parameter used for normalization and ray–facet intersection tests.
    atol :
        Tolerance used for deterministic hemisphere representative tie-breaking near the equator.

    Returns
    -------
    cover :
        A built :class:`~circle_bundles.triangle_cover_builders_fibonacci.RP2StarCover`.

    Notes
    -----
    - Correct handling of :math:`\mathbb{RP}^2` requires considering **both lifts** of each sample
      upstairs (``x`` and ``-x``) before pushing weights down.

    Examples
    --------
    >>> import circle_bundles as cb
    >>> cover = cb.make_rp2_fibonacci_star_cover(base_points, n_pairs=256)
    >>> cover.summarize(plot=False)
    >>> bundle = cb.build_bundle(total_space_data, cover, show=True)
    """
    # 1) Samples as RP^2 hemisphere reps
    P = _normalize_rows(np.asarray(base_points, float), eps=eps)
    P_rep = np.stack([hemisphere_rep(P[i], atol=atol) for i in range(P.shape[0])], axis=0)
    P_other = -P_rep

    # 2) Well-spread RP^2 landmarks
    n = int(n_pairs)
    L_rp2 = _pick_rp2_landmarks(
        n,
        oversample=int(landmark_oversample),
        seed=landmark_seed,
        atol=atol,
    )  # (n,3), hemisphere reps

    # 3) Lift upstairs
    V_full = np.vstack([L_rp2, -L_rp2])  # (2n,3)

    # 4) Hull triangulation upstairs
    hull, faces_full = _build_hull(V_full)

    # 5) Ray hits for both lifts
    P_hit_a, hit_face_a = _project_rays_to_hull(P_rep, hull, eps=eps)
    P_hit_b, hit_face_b = _project_rays_to_hull(P_other, hull, eps=eps)

    # 6) Quotient map old_vid -> rp2_vid (exact pairing i <-> i+n)
    old_to_new: Dict[int, int] = {i: i for i in range(n)}
    old_to_new.update({i + n: i for i in range(n)})

    # 7) Accumulate weights from both lifts; take top-3
    N = P_rep.shape[0]
    sample_triples = np.zeros((N, 3), dtype=int)
    sample_weights = np.zeros((N, 3), dtype=float)

    def _accumulate_from_hit(
        acc: Dict[int, float], *, face_idx: int, p_hit_row: np.ndarray
    ) -> None:
        a, b, c = (int(x) for x in faces_full[face_idx])
        tri_xyz = V_full[[a, b, c]]  # (3,3)
        bc = get_bary_coords(p_hit_row[None, :], tri_xyz)[0]  # (3,)
        vids = [old_to_new[a], old_to_new[b], old_to_new[c]]
        wts = [float(bc[0]), float(bc[1]), float(bc[2])]
        for vid, w in zip(vids, wts):
            vv = int(vid)
            acc[vv] = acc.get(vv, 0.0) + max(0.0, float(w))

    for s in range(N):
        acc: Dict[int, float] = {}
        _accumulate_from_hit(acc, face_idx=int(hit_face_a[s]), p_hit_row=P_hit_a[s])  # x
        _accumulate_from_hit(acc, face_idx=int(hit_face_b[s]), p_hit_row=P_hit_b[s])  # -x

        items = sorted(acc.items(), key=lambda t: -t[1])
        if len(items) == 0:
            items = [(0, 1.0)]
        while len(items) < 3:
            items.append((items[-1][0], 0.0))

        (v0, w0), (v1, w1), (v2, w2) = items[:3]
        ws = np.array([w0, w1, w2], dtype=float)
        ssum = float(ws.sum())
        ws = ws / ssum if ssum > 0 else np.array([1.0, 0.0, 0.0], dtype=float)

        sample_triples[s] = np.array([int(v0), int(v1), int(v2)], dtype=int)
        sample_weights[s] = ws

    cover = RP2StarCover(
        base_points=P_rep,
        landmarks=L_rp2,
        sample_vertex_triples=sample_triples,
        sample_weights=sample_weights,
    ).build()
    return cover
