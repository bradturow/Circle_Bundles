# circle_bundles/characteristic_class.py
"""
Characteristic classes for O(2)-bundles on a nerve (supports up to 3-simplices).

REFRACTOR (Feb 2026)
--------------------
We split the old monolithic `compute_classes(...)` into two layers so Bundle.get_classes()
can do:

  (1) compute reps only (fast, cochain-level): w1 (edge Z2 cochain) + twisted Euler rep on triangles
  (2) run persistence on reps (weights filtration)
  (3) restrict to a subcomplex where reps are cocycles
  (4) compute derived class data (coboundary tests, pairings, trivial/spin flags) on that subcomplex

New public helpers:
  - compute_class_representatives_from_nerve(...)
  - compute_class_data_on_complex(...)

Back-compat:
  - compute_classes(...) still returns a full ClassResult on the full complex (old behavior),
    implemented by calling the new helpers internally.

Notes
-----
- We keep the exact same conventions as your old file.
- We do NOT change the mathematics; this is purely a refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import sympy as sp

from .nerve.combinatorics import Edge, Tri, canon_edge, canon_tri
from .analysis.class_persistence import (
    build_delta_C0_to_C1_Z2,
    build_delta_C1_to_C2_Z_twisted,
    in_image_mod2,
    in_image_Z_fast_pipeline,
)

__all__ = [
    # main old entrypoint
    "compute_classes",
    # new refactor entrypoints
    "compute_class_representatives_from_nerve",
    "compute_class_data_on_complex",
    # results
    "ClassReps",
    "ClassResult",
    # summary helper (kept)
    "show_summary",
]

Tet = Tuple[int, int, int, int]


# ============================================================
# Canonicalization
# ============================================================

def canon_tet(a: int, b: int, c: int, d: int) -> Tet:
    return tuple(sorted((int(a), int(b), int(c), int(d))))


# ============================================================
# R/Z lift helper
# ============================================================

def principal_lift_RZ(x: float) -> float:
    """
    x in R/Z represented as float (mod 1).
    Return the principal lift in (-1/2, 1/2].
    """
    y = float(x) % 1.0
    if y > 0.5:
        y -= 1.0
    return y


# ============================================================
# O(1) cochain canonicalization
# ============================================================

def canonicalize_o1_cochain(omega_O1: Dict[Tuple[int, int], int]) -> Dict[Edge, int]:
    """
    Force an O(1) 1-cochain (±1) to use canonical edge keys (min,max).
    If both orientations are present, require consistency.
    """
    out: Dict[Edge, int] = {}
    for (a, b), v in omega_O1.items():
        e = canon_edge(int(a), int(b))
        vv = int(v)
        if vv not in (+1, -1):
            raise ValueError(f"omega_O1 values must be ±1; got {vv} on edge {(a, b)}")
        if e in out and out[e] != vv:
            raise ValueError(f"Inconsistent omega_O1 values on edge {e}: {out[e]} vs {vv}")
        out[e] = vv
    return out


# ============================================================
# Twisted coboundaries (2 and 3)
# ============================================================

def twisted_delta_theta_real(tri: Tri, theta_norm: Dict[Edge, float], omega_O1: Dict[Edge, int]) -> float:
    """
    δ_ω θ(i,j,k) = ω(i,j) θ(j,k) - θ(i,k) + θ(i,j)
    with tri = (i<j<k), edges canonical.

    theta_norm values are in R/Z as floats mod 1.
    We principal-lift each to (-1/2, 1/2] before forming the coboundary.
    """
    i, j, k = tri
    e01 = canon_edge(i, j)
    e12 = canon_edge(j, k)
    e02 = canon_edge(i, k)

    w01 = int(omega_O1[e01])

    t01 = principal_lift_RZ(theta_norm[e01])
    t12 = principal_lift_RZ(theta_norm[e12])
    t02 = principal_lift_RZ(theta_norm[e02])

    return float(w01 * t12 - t02 + t01)


def twisted_delta_euler_on_tet(tet: Tet, euler: Dict[Tri, int], omega_O1: Dict[Edge, int]) -> int:
    """
    3D twisted coboundary of an integer 2-cochain e on a tetrahedron (i<j<k<l):

      (δ_ω e)(i,j,k,l) = ω(i,j) e(j,k,l) - e(i,k,l) + e(i,j,l) - e(i,j,k)
    """
    i, j, k, l = tet
    w01 = int(omega_O1[canon_edge(i, j)])

    t_jkl = canon_tri(j, k, l)
    t_ikl = canon_tri(i, k, l)
    t_ijl = canon_tri(i, j, l)
    t_ijk = canon_tri(i, j, k)

    return int(
        w01 * int(euler.get(t_jkl, 0))
        - int(euler.get(t_ikl, 0))
        + int(euler.get(t_ijl, 0))
        - int(euler.get(t_ijk, 0))
    )


def compute_twisted_euler_class(cocycle: Any, triangles: Iterable[Tri]):
    """
    Compute the twisted Euler representative by rounding δ_ω θ on triangles.

    Returns
    -------
    e_rep: dict tri -> int
    rounding_dist: float (max |delta - round(delta)|)
    e_real: dict tri -> float
    omega_O1_used: dict edge -> ±1 (canonical keys)
    """
    theta_norm_raw = cocycle.theta_normalized()
    theta_norm: Dict[Edge, float] = {canon_edge(a, b): float(v) for (a, b), v in theta_norm_raw.items()}

    omega_O1_raw = cocycle.omega_O1()
    omega_O1_used: Dict[Edge, int] = canonicalize_o1_cochain(omega_O1_raw)

    e_real: Dict[Tri, float] = {}
    e_rep: Dict[Tri, int] = {}
    max_dev = 0.0

    for t in triangles:
        tri = canon_tri(*t)
        val = twisted_delta_theta_real(tri, theta_norm, omega_O1_used)
        r = int(np.round(val))
        e_real[tri] = float(val)
        e_rep[tri] = r
        max_dev = max(max_dev, abs(val - r))

    return e_rep, float(max_dev), e_real, omega_O1_used


# ============================================================
# Linear algebra helpers (Q and mod2)
# ============================================================

def rank_over_Q(A: np.ndarray) -> int:
    if A.size == 0:
        return 0
    return int(sp.Matrix(A).rank())


def in_colspace_over_Q(M: sp.Matrix, v: sp.Matrix) -> bool:
    if M.cols == 0:
        return False
    if v.cols != 1:
        v = sp.Matrix(v).reshape(v.rows, 1)
    return int(M.row_join(v).rank()) == int(M.rank())


def rref_mod2(A_in: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    A = (np.asarray(A_in, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    row = 0
    pivots: List[int] = []

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue

        if pivot != row:
            A[[row, pivot], :] = A[[pivot, row], :]

        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r, :] ^= A[row, :]

        pivots.append(col)
        row += 1
        if row == m:
            break

    return A, pivots


def nullspace_basis_mod2(A_in: np.ndarray) -> List[np.ndarray]:
    A = (np.asarray(A_in, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    R, pivots = rref_mod2(A)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]

    basis: List[np.ndarray] = []
    for free in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[free] = 1
        for i in range(len(pivots) - 1, -1, -1):
            p = pivots[i]
            ones = np.flatnonzero(R[i, :])
            s = 0
            for j in ones:
                if j == p:
                    continue
                s ^= int(x[j])
            x[p] = s & 1
        basis.append(x)
    return basis


# ============================================================
# H2 dims and fundamental classes (same as old)
# ============================================================

def build_twisted_boundary_C2_to_C1(
    edges: List[Edge],
    triangles: List[Tri],
    omega_O1: Dict[Edge, int],
) -> np.ndarray:
    edges = [canon_edge(*e) for e in edges]
    triangles = [canon_tri(*t) for t in triangles]
    edge_index = {e: r for r, e in enumerate(edges)}

    omega_O1 = canonicalize_o1_cochain(omega_O1)
    for e in edges:
        if e not in omega_O1:
            raise KeyError(f"omega_O1 missing required edge {e}")

    m = len(edges)
    n = len(triangles)
    D = np.zeros((m, n), dtype=int)

    for c, (i, j, k) in enumerate(triangles):
        eij = canon_edge(i, j)
        ejk = canon_edge(j, k)
        eik = canon_edge(i, k)

        w01 = int(omega_O1[eij])
        D[edge_index[ejk], c] += w01
        D[edge_index[eik], c] += -1
        D[edge_index[eij], c] += +1

    return D


def build_twisted_boundary_C3_to_C2(
    triangles: List[Tri],
    tets: List[Tet],
    omega_O1: Dict[Edge, int],
) -> np.ndarray:
    triangles = [canon_tri(*t) for t in triangles]
    tets = [canon_tet(*tt) for tt in tets]
    tri_index = {t: r for r, t in enumerate(triangles)}

    omega_O1 = canonicalize_o1_cochain(omega_O1)

    m = len(triangles)
    n = len(tets)
    D = np.zeros((m, n), dtype=int)

    for c, (i, j, k, l) in enumerate(tets):
        w01 = int(omega_O1[canon_edge(i, j)])

        t_jkl = canon_tri(j, k, l)
        t_ikl = canon_tri(i, k, l)
        t_ijl = canon_tri(i, j, l)
        t_ijk = canon_tri(i, j, k)

        for face in (t_jkl, t_ikl, t_ijl, t_ijk):
            if face not in tri_index:
                raise KeyError(f"Tetrahedron face {face} not found in triangles list; include all faces.")

        D[tri_index[t_jkl], c] += w01
        D[tri_index[t_ikl], c] += -1
        D[tri_index[t_ijl], c] += +1
        D[tri_index[t_ijk], c] += -1

    return D


def build_boundary_mod2_C2_to_C1(edges: List[Edge], triangles: List[Tri]) -> np.ndarray:
    edges = [canon_edge(*e) for e in edges]
    triangles = [canon_tri(*t) for t in triangles]
    edge_index = {e: r for r, e in enumerate(edges)}

    D = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for c, (i, j, k) in enumerate(triangles):
        D[edge_index[canon_edge(j, k)], c] ^= 1
        D[edge_index[canon_edge(i, k)], c] ^= 1
        D[edge_index[canon_edge(i, j)], c] ^= 1
    return D


def build_boundary_mod2_C3_to_C2(triangles: List[Tri], tets: List[Tet]) -> np.ndarray:
    triangles = [canon_tri(*t) for t in triangles]
    tets = [canon_tet(*tt) for tt in tets]
    tri_index = {t: r for r, t in enumerate(triangles)}

    D = np.zeros((len(triangles), len(tets)), dtype=np.uint8)
    for c, (i, j, k, l) in enumerate(tets):
        for face in (
            canon_tri(j, k, l),
            canon_tri(i, k, l),
            canon_tri(i, j, l),
            canon_tri(i, j, k),
        ):
            if face not in tri_index:
                raise KeyError(f"Tet face {face} not found in triangles list; include all faces.")
            D[tri_index[face], c] ^= 1
    return D


def normalize_cycle_Z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=int).copy()
    if z.size == 0 or np.all(z == 0):
        return z
    g = int(np.gcd.reduce(np.abs(z[z != 0])))
    if g > 1:
        z //= g
    first = int(np.flatnonzero(z)[0])
    if z[first] < 0:
        z = -z
    return z


def H2_dimensions(
    edges: List[Edge],
    triangles: List[Tri],
    tets: List[Tet],
    omega_O1: Dict[Edge, int],
) -> Dict[str, int]:
    edges = sorted({canon_edge(*e) for e in edges})
    triangles = sorted({canon_tri(*t) for t in triangles})
    tets = sorted({canon_tet(*tt) for tt in tets})

    if len(triangles) == 0:
        return {"dim_Q": 0, "dim_Z2": 0}

    D2w = build_twisted_boundary_C2_to_C1(edges, triangles, omega_O1=omega_O1)
    r2 = rank_over_Q(D2w)
    n2 = len(triangles)
    ker2_dim = n2 - r2

    if len(tets) == 0:
        im3_dim = 0
    else:
        D3w = build_twisted_boundary_C3_to_C2(triangles, tets, omega_O1=omega_O1)
        im3_dim = rank_over_Q(D3w)

    dim_Q = int(max(ker2_dim - im3_dim, 0))

    D2 = build_boundary_mod2_C2_to_C1(edges, triangles)
    _, piv2 = rref_mod2(D2)
    ker2_dim2 = n2 - len(piv2)

    if len(tets) == 0:
        im3_dim2 = 0
    else:
        D3 = build_boundary_mod2_C3_to_C2(triangles, tets)
        _, piv3 = rref_mod2(D3)
        im3_dim2 = len(piv3)

    dim_Z2 = int(max(ker2_dim2 - im3_dim2, 0))
    return {"dim_Q": dim_Q, "dim_Z2": dim_Z2}


def fundamental_class_Z_rank1(
    edges: List[Edge],
    triangles: List[Tri],
    tets: List[Tet],
    omega_O1: Dict[Edge, int],
) -> Optional[np.ndarray]:
    edges = sorted({canon_edge(*e) for e in edges})
    triangles = sorted({canon_tri(*t) for t in triangles})
    tets = sorted({canon_tet(*tt) for tt in tets})

    if len(triangles) == 0:
        return None

    D2w = sp.Matrix(build_twisted_boundary_C2_to_C1(edges, triangles, omega_O1=omega_O1))
    ns = D2w.nullspace()
    if len(ns) == 0:
        return None

    if len(tets) == 0:
        if len(ns) != 1:
            return None
        v = ns[0]
    else:
        D3w = sp.Matrix(build_twisted_boundary_C3_to_C2(triangles, tets, omega_O1=omega_O1))
        candidates = []
        for v0 in ns:
            if not in_colspace_over_Q(D3w, v0):
                candidates.append(v0)
        if len(candidates) != 1:
            return None
        v = candidates[0]

    lcm = 1
    for x in v:
        xr = sp.Rational(x)
        lcm = int(sp.ilcm(lcm, int(xr.q)))

    z = np.array([int(sp.Integer(sp.Rational(x) * lcm)) for x in v], dtype=int)
    return normalize_cycle_Z(z)


def fundamental_class_Z2_rank1(
    edges: List[Edge],
    triangles: List[Tri],
    tets: List[Tet],
) -> Optional[np.ndarray]:
    edges = sorted({canon_edge(*e) for e in edges})
    triangles = sorted({canon_tri(*t) for t in triangles})
    tets = sorted({canon_tet(*tt) for tt in tets})

    if len(triangles) == 0:
        return None

    D2 = build_boundary_mod2_C2_to_C1(edges, triangles)
    ker_basis = nullspace_basis_mod2(D2)
    if len(ker_basis) == 0:
        return None

    if len(tets) == 0:
        if len(ker_basis) != 1:
            return None
        return ker_basis[0]

    D3 = build_boundary_mod2_C3_to_C2(triangles, tets)

    # column space membership over GF2
    def _in_image_mod2_cols(A: np.ndarray, b: np.ndarray) -> bool:
        return in_image_mod2(A, b)

    candidates = []
    for v in ker_basis:
        if not _in_image_mod2_cols(D3, v):
            candidates.append(v)

    if len(candidates) != 1:
        return None
    return candidates[0]


def euler_pairing_Z(e_class: Dict[Tri, int], triangles: List[Tri], z_fund_Z: np.ndarray) -> int:
    total = 0
    for idx, tri in enumerate(triangles):
        total += int(e_class.get(tri, 0)) * int(z_fund_Z[idx])
    return int(total)


def euler_pairing_Z2(e_class: Dict[Tri, int], triangles: List[Tri], z_fund_Z2: np.ndarray) -> int:
    acc = 0
    for idx, tri in enumerate(triangles):
        acc ^= (int(e_class.get(tri, 0)) & 1) & int(z_fund_Z2[idx])
    return int(acc & 1)


# ============================================================
# Result containers
# ============================================================

@dataclass
class ClassReps:
    """
    Lightweight cochain-level representatives (no coboundary tests / pairings).
    This is what Bundle.get_classes() wants first.

    Required downstream fields for persistence:
      - sw1_Z2
      - euler_class
      - omega_O1_used
    """
    n_vertices: int
    n_edges: int
    n_triangles: int
    n_tets: int

    sw1_Z2: Dict[Edge, int]
    sw1_O1: Dict[Edge, int]
    sw1_is_cocycle: Optional[bool]

    orientable: bool
    phi_pm1: Optional[np.ndarray]

    cocycle_used: Any

    euler_class: Dict[Tri, int]
    euler_class_real: Dict[Tri, float]
    rounding_dist: float
    omega_O1_used: Dict[Edge, int]


@dataclass
class ClassResult:
    """
    Full class report (old behavior), potentially after restricting to a subcomplex.
    """
    n_vertices: int
    n_edges: int
    n_triangles: int
    n_tets: int

    sw1_Z2: Dict[Edge, int]
    sw1_O1: Dict[Edge, int]
    sw1_is_cocycle: Optional[bool]
    sw1_is_coboundary_Z2: Optional[bool]

    orientable: bool
    phi_pm1: Optional[np.ndarray]
    cocycle_used: Any

    euler_class: Dict[Tri, int]
    euler_class_real: Dict[Tri, float]
    rounding_dist: float
    omega_O1_used: Dict[Edge, int]

    euler_is_cocycle: Optional[bool]
    max_abs_delta_euler_on_tets: Optional[int]

    euler_is_coboundary_Z: Optional[bool]

    H2_dim_Q: Optional[int]
    H2_dim_Z2: Optional[int]

    twisted_euler_number_Z: Optional[int]
    euler_number_mod2: Optional[int]

    sw2_Z2: Dict[Tri, int]

    bundle_trivial_on_this_complex: Optional[bool]
    spin_on_this_complex: Optional[bool]


# ============================================================
# Internal checks
# ============================================================

def _sw1_cocycle_check_on_triangles(triangles: List[Tri], omega_Z2: Dict[Edge, int]) -> bool:
    for (i, j, k) in triangles:
        eij = canon_edge(i, j)
        eik = canon_edge(i, k)
        ejk = canon_edge(j, k)
        val = (int(omega_Z2.get(ejk, 0)) - int(omega_Z2.get(eik, 0)) + int(omega_Z2.get(eij, 0))) & 1
        if val != 0:
            return False
    return True


def _infer_n_vertices_from_simplices(
    edges: List[Edge],
    triangles: List[Tri],
    tets: List[Tet],
    *,
    fallback: Optional[int] = None,
) -> int:
    mx = -1
    for (a, b) in edges:
        mx = max(mx, int(a), int(b))
    for (a, b, c) in triangles:
        mx = max(mx, int(a), int(b), int(c))
    for (a, b, c, d) in tets:
        mx = max(mx, int(a), int(b), int(c), int(d))
    if mx >= 0:
        return int(mx + 1)
    if fallback is None:
        raise ValueError("Could not infer n_vertices from empty simplices and no fallback provided.")
    return int(fallback)


# ============================================================
# NEW: reps-only computation
# ============================================================

def compute_class_representatives_from_nerve(
    *,
    cocycle: Any,
    edges: Iterable[Edge],
    triangles: Iterable[Tri],
    tets: Optional[Iterable[Tet]] = None,
    n_vertices: Optional[int] = None,
    try_orient: bool = True,
) -> ClassReps:
    """
    Compute only the cochain-level class representatives:
      - w1 as a Z2 1-cochain on edges
      - twisted Euler rep on triangles (rounded δ_ω θ)
    plus the orient_if_possible attempt (if requested).

    Does NOT compute:
      - sw1 coboundary test
      - Euler cocycle/coboundary tests
      - H2 dims / Euler numbers / trivial/spin flags
    """
    edges_list = sorted({canon_edge(*e) for e in edges})
    tris_list = sorted({canon_tri(*t) for t in triangles})
    tets_list = sorted({canon_tet(*tt) for tt in (tets or [])})

    if n_vertices is None:
        n_vertices = _infer_n_vertices_from_simplices(edges_list, tris_list, tets_list, fallback=None)

    # Restrict cocycle to this 1-skeleton
    coc_res = cocycle.restrict(edges_list)

    sw1_Z2 = {canon_edge(*e): int(v) & 1 for e, v in coc_res.omega_Z2().items()}
    sw1_O1 = canonicalize_o1_cochain(coc_res.omega_O1())

    sw1_is_cocycle: Optional[bool] = None
    if len(tris_list) > 0:
        sw1_is_cocycle = bool(_sw1_cocycle_check_on_triangles(tris_list, sw1_Z2))

    orientable = False
    phi_pm1 = None
    coc_used = coc_res

    if try_orient and len(edges_list) > 0:
        ok, coc_oriented, phi_pm1 = coc_res.orient_if_possible(edges_list, n_vertices=n_vertices)
        orientable = bool(ok)
        coc_used = coc_oriented if orientable else coc_res

    if len(tris_list) == 0:
        euler_rep: Dict[Tri, int] = {}
        euler_real: Dict[Tri, float] = {}
        rounding_dist = 0.0
        omega_O1_used = canonicalize_o1_cochain(coc_used.omega_O1())
    else:
        euler_rep, rounding_dist, euler_real, omega_O1_used = compute_twisted_euler_class(coc_used, tris_list)
        # restrict ω to exactly edges in this complex
        omega_O1_used = {e: int(omega_O1_used.get(e, 1)) for e in edges_list}

    return ClassReps(
        n_vertices=int(n_vertices),
        n_edges=int(len(edges_list)),
        n_triangles=int(len(tris_list)),
        n_tets=int(len(tets_list)),
        sw1_Z2=sw1_Z2,
        sw1_O1=sw1_O1,
        sw1_is_cocycle=sw1_is_cocycle,
        orientable=orientable,
        phi_pm1=phi_pm1,
        cocycle_used=coc_used,
        euler_class=euler_rep,
        euler_class_real=euler_real,
        rounding_dist=float(rounding_dist),
        omega_O1_used=omega_O1_used,
    )


# ============================================================
# NEW: derived class data on a chosen complex
# ============================================================

def compute_class_data_on_complex(
    *,
    reps: ClassReps,
    edges: Iterable[Edge],
    triangles: Iterable[Tri],
    tets: Optional[Iterable[Tet]] = None,
    n_vertices: Optional[int] = None,
    compute_euler_num: bool = True,
) -> ClassResult:
    """
    Given already-computed reps (w1 cochain + Euler rep), compute derived class data on
    the provided (possibly restricted) complex:

      - sw1 coboundary test on edges
      - euler cocycle check on tets (if any)
      - euler coboundary test (only meaningful if e is a cocycle in 3D)
      - H2 dims and Euler numbers (when available)
      - w2 (e mod 2) and spin flag (orientable case)
      - bundle_trivial_on_this_complex

    This is the “after restriction” piece for Bundle.get_classes().
    """
    edges_list = sorted({canon_edge(*e) for e in edges})
    tris_list = sorted({canon_tri(*t) for t in triangles})
    tets_list = sorted({canon_tet(*tt) for tt in (tets or [])})

    if n_vertices is None:
        n_vertices = _infer_n_vertices_from_simplices(edges_list, tris_list, tets_list, fallback=reps.n_vertices)

    # --- sw1 coboundary on this complex ---
    sw1_is_coboundary_Z2: Optional[bool] = None
    if len(edges_list) > 0:
        V0 = [(i,) for i in range(int(n_vertices))]
        A01 = build_delta_C0_to_C1_Z2(vertices=V0, edges=edges_list)
        b1 = np.array([int(reps.sw1_Z2.get(e, 0)) & 1 for e in edges_list], dtype=np.uint8)
        sw1_is_coboundary_Z2 = bool(in_image_mod2(A01, b1))

    # --- Euler rep restricted to current triangles ---
    euler_rep = {canon_tri(*t): int(reps.euler_class.get(canon_tri(*t), 0)) for t in tris_list}
    omega_O1_used = {canon_edge(*e): int(reps.omega_O1_used.get(canon_edge(*e), 1)) for e in edges_list}

    # sw2 is e mod2 on triangles present
    sw2 = {tri: (int(euler_rep.get(tri, 0)) & 1) for tri in tris_list}

    # --- spin check: w2 coboundary over Z2 (only meaningful if orientable and tris exist) ---
    sw2_is_coboundary_Z2: Optional[bool] = None
    if reps.orientable and (len(tris_list) > 0) and (len(edges_list) > 0):
        D2_mod2 = build_boundary_mod2_C2_to_C1(edges_list, tris_list)  # (E x T)
        A12_mod2 = (D2_mod2.T).astype(np.uint8)                        # (T x E)
        b2 = np.array([sw2.get(t, 0) & 1 for t in tris_list], dtype=np.uint8)
        sw2_is_coboundary_Z2 = bool(in_image_mod2(A12_mod2, b2))

    # --- euler cocycle check on tets (3D) ---
    if len(tets_list) == 0:
        euler_is_cocycle = None
        max_abs_delta = None
    else:
        bad = False
        max_abs = 0
        for tt in tets_list:
            dv = twisted_delta_euler_on_tet(tt, euler_rep, omega_O1_used)
            max_abs = max(max_abs, abs(int(dv)))
            if dv != 0:
                bad = True
        euler_is_cocycle = (not bad)
        max_abs_delta = int(max_abs)

    # --- Euler coboundary test (twisted) ---
    euler_is_coboundary_Z: Optional[bool] = None
    if len(tris_list) > 0 and len(edges_list) > 0:
        ok_e = True if (euler_is_cocycle is None) else bool(euler_is_cocycle)
        if ok_e:
            A12 = build_delta_C1_to_C2_Z_twisted(edges=edges_list, triangles=tris_list, omega_O1=omega_O1_used).astype(
                np.int64
            )
            b2Z = np.array([int(euler_rep.get(t, 0)) for t in tris_list], dtype=np.int64)
            euler_is_coboundary_Z = bool(in_image_Z_fast_pipeline(A12, b2Z))
        else:
            euler_is_coboundary_Z = None

    # --- H2 dims and pairings (Euler numbers) ---
    H2_dim_Q = None
    H2_dim_Z2 = None
    eZ = None
    eZ2 = None

    if compute_euler_num:
        ok_e = True if (euler_is_cocycle is None) else bool(euler_is_cocycle)
        if ok_e:
            dims = H2_dimensions(edges_list, tris_list, tets_list, omega_O1=omega_O1_used)
            H2_dim_Q = int(dims["dim_Q"])
            H2_dim_Z2 = int(dims["dim_Z2"])

            if H2_dim_Q == 0:
                eZ = None
            elif H2_dim_Q == 1:
                z = fundamental_class_Z_rank1(edges_list, tris_list, tets_list, omega_O1=omega_O1_used)
                eZ = None if z is None else euler_pairing_Z(euler_rep, tris_list, z)
            else:
                eZ = None

            if H2_dim_Z2 == 0:
                eZ2 = 0
            elif H2_dim_Z2 == 1:
                z2 = fundamental_class_Z2_rank1(edges_list, tris_list, tets_list)
                eZ2 = None if z2 is None else euler_pairing_Z2(euler_rep, tris_list, z2)
            else:
                eZ2 = None

    # --- bundle trivial flag on this complex (same logic as old file) ---
    if len(tets_list) == 0:
        bundle_trivial = bool(reps.orientable and (euler_is_coboundary_Z is True))
    else:
        bundle_trivial = bool(reps.orientable and (euler_is_cocycle is True) and (euler_is_coboundary_Z is True))

    spin = bool(sw2_is_coboundary_Z2) if sw2_is_coboundary_Z2 is not None else None

    # Preserve reps fields + attach derived fields
    # (We keep euler_class_real and rounding_dist from reps, but note: euler_class_real
    # is computed on the full triangle list used when building reps. That’s fine for summaries,
    # and it keeps this function lightweight.)
    return ClassResult(
        n_vertices=int(n_vertices),
        n_edges=int(len(edges_list)),
        n_triangles=int(len(tris_list)),
        n_tets=int(len(tets_list)),
        sw1_Z2=reps.sw1_Z2,
        sw1_O1=reps.sw1_O1,
        sw1_is_cocycle=reps.sw1_is_cocycle,
        sw1_is_coboundary_Z2=sw1_is_coboundary_Z2,
        orientable=bool(reps.orientable),
        phi_pm1=reps.phi_pm1,
        cocycle_used=reps.cocycle_used,
        euler_class=euler_rep,
        euler_class_real=reps.euler_class_real,
        rounding_dist=float(reps.rounding_dist),
        omega_O1_used=omega_O1_used,
        euler_is_cocycle=euler_is_cocycle,
        max_abs_delta_euler_on_tets=max_abs_delta,
        euler_is_coboundary_Z=euler_is_coboundary_Z,
        H2_dim_Q=H2_dim_Q,
        H2_dim_Z2=H2_dim_Z2,
        twisted_euler_number_Z=eZ,
        euler_number_mod2=eZ2,
        sw2_Z2=sw2,
        bundle_trivial_on_this_complex=bundle_trivial,
        spin_on_this_complex=spin,
    )


# ============================================================
# Back-compat: old compute_classes
# ============================================================

def compute_classes(
    cover: Any,
    cocycle: Any,
    *,
    edges: Optional[Iterable[Edge]] = None,
    triangles: Optional[Iterable[Tri]] = None,
    tets: Optional[Iterable[Tet]] = None,
    n_vertices: Optional[int] = None,
    try_orient: bool = True,
    compute_euler_num: bool = True,
) -> ClassResult:
    """
    Backwards-compatible full class computation on the provided complex (defaults to cover nerve).

    Implemented via:
      reps = compute_class_representatives_from_nerve(...)
      full = compute_class_data_on_complex(reps, ...)
    """
    if edges is None:
        edges = cover.nerve_edges()
    if triangles is None:
        triangles = cover.nerve_triangles()
    if tets is None:
        # try common names; else empty
        if hasattr(cover, "nerve_tetrahedra"):
            tets = cover.nerve_tetrahedra()
        elif hasattr(cover, "nerve_tets"):
            tets = cover.nerve_tets()
        else:
            tets = []

    edges_list = sorted({canon_edge(*e) for e in edges})
    tris_list = sorted({canon_tri(*t) for t in triangles})
    tets_list = sorted({canon_tet(*tt) for tt in tets})

    if n_vertices is None:
        n_vertices = _infer_n_vertices_from_simplices(edges_list, tris_list, tets_list, fallback=getattr(cover, "U", np.zeros((0, 0))).shape[0] or 0)

    reps = compute_class_representatives_from_nerve(
        cocycle=cocycle,
        edges=edges_list,
        triangles=tris_list,
        tets=tets_list,
        n_vertices=int(n_vertices),
        try_orient=bool(try_orient),
    )

    return compute_class_data_on_complex(
        reps=reps,
        edges=edges_list,
        triangles=tris_list,
        tets=tets_list,
        n_vertices=int(n_vertices),
        compute_euler_num=bool(compute_euler_num),
    )


# ============================================================
# Pretty summary (UNCHANGED from your version)
# ============================================================

def _euc_to_geo_rad(d: Optional[float]) -> Optional[float]:
    if d is None:
        return None
    d = float(d)
    if not np.isfinite(d):
        return None
    x = np.clip(d / 2.0, 0.0, 1.0)
    return float(2.0 * np.arcsin(x))


def _fmt_euc_with_geo_pi(d_euc: Optional[float], *, decimals: int = 3) -> str:
    if d_euc is None or not np.isfinite(float(d_euc)):
        return "—"
    d_euc = float(d_euc)
    theta = _euc_to_geo_rad(d_euc)
    if theta is None:
        return f"{d_euc:.{decimals}f}"
    return f"{d_euc:.{decimals}f} (\\varepsilon_{{\\text{{triv}}}}^{{\\text{{geo}}}}={theta/np.pi:.{decimals}f}\\pi)"


def _fmt_mean_euc_with_geo_pi(d_euc_mean: Optional[float], *, decimals: int = 3) -> str:
    if d_euc_mean is None or not np.isfinite(float(d_euc_mean)):
        return "—"
    d_euc_mean = float(d_euc_mean)
    theta = _euc_to_geo_rad(d_euc_mean)
    if theta is None:
        return f"{d_euc_mean:.{decimals}f}"
    return f"{d_euc_mean:.{decimals}f} (\\bar{{\\varepsilon}}_{{\\text{{triv}}}}^{{\\text{{geo}}}}={theta/np.pi:.{decimals}f}\\pi)"


def show_summary(classes, *, quality=None, show: bool = True, mode: str = "auto") -> str:
    """
    Pretty summary of (1) diagnostics and (2) characteristic classes.

    This is kept verbatim from your previous version (so existing notebooks stay stable).
    """
    eps_triv = getattr(quality, "eps_align_euc", None) if quality is not None else None
    eps_triv_mean = getattr(quality, "eps_align_euc_mean", None) if quality is not None else None

    delta = getattr(quality, "delta", None) if quality is not None else None
    alpha = getattr(quality, "alpha", None) if quality is not None else None
    eps_coc = getattr(quality, "cocycle_defect", None) if quality is not None else None

    rounding_dist = float(getattr(classes, "rounding_dist", 0.0))
    orientable = bool(getattr(classes, "orientable", False))
    n_tri = int(getattr(classes, "n_triangles", 0))

    w1_is_cob = getattr(classes, "sw1_is_coboundary_Z2", None)
    if w1_is_cob is None:
        w1_is_cob = bool(orientable)

    e_is_cob = getattr(classes, "euler_is_coboundary_Z", None)
    euler_rep = getattr(classes, "euler_class", {}) or {}
    e_trivial_cochain = all(int(v) == 0 for v in euler_rep.values()) if euler_rep else True
    e_zero_for_print = bool(e_is_cob) if e_is_cob is not None else bool(e_trivial_cochain)

    eZ = getattr(classes, "twisted_euler_number_Z", None)

    def _infer_w2_is_zero() -> Optional[bool]:
        spn = getattr(classes, "spin_on_this_complex", None)
        if spn is not None:
            return bool(spn)
        sw2 = getattr(classes, "sw2_Z2", None)
        if isinstance(sw2, dict) and len(sw2) > 0:
            return all((int(v) & 1) == 0 for v in sw2.values())
        return None

    IND = "  "
    LABEL_W = 28

    def _tline(label: str, content: str) -> str:
        return f"{IND}{label:<{LABEL_W}} {content}"

    lines: List[str] = []
    lines.append("=== Diagnostics ===")

    if quality is None:
        lines.append(f"{IND}(no quality report provided)")
    else:
        if eps_triv is not None:
            lines.append(
                _tline(
                    "trivialization error:",
                    "ε_triv := sup_{(j k)∈N(U)} sup_{x∈π^{-1}(U_j∩U_k)} d_𝕮(Ω_{jk} f_k(x), f_j(x))"
                    f" = {_fmt_euc_with_geo_pi(eps_triv)}",
                )
            )
        if eps_triv_mean is not None:
            lines.append(_tline("mean triv error:", f"\\bar{{ε}}_triv = {_fmt_mean_euc_with_geo_pi(eps_triv_mean)}"))
        if delta is not None:
            lines.append(
                _tline(
                    "surjectivity defect:",
                    "δ := sup_{(i j k)∈N(U)} min_{v∈{i,j,k}} d_H(f_v(π^{-1}(U_i∩U_j∩U_k)), S^1)"
                    f" = {float(delta):.3f}",
                )
            )
        if alpha is not None:
            if alpha == float("inf"):
                lines.append(_tline("stability ratio:", "α := ε_triv/(1-δ) = ∞  (since δ ≥ 1)"))
            else:
                lines.append(_tline("stability ratio:", f"α := ε_triv/(1-δ) = {float(alpha):.3f}"))
        if eps_coc is not None:
            lines.append(
                _tline(
                    "cocycle error:",
                    "ε_coc := sup_{(i j k)∈N(U)} ‖Ω_{ij}Ω_{jk}Ω_{ki} - I‖_F" f" = {float(eps_coc):.3f}",
                )
            )
        lines.append(_tline("Euler rounding diag:", f"d_∞(δ_ω θ, ẽ) = {rounding_dist:.6g}"))

    lines.append("")
    lines.append("=== Characteristic Classes ===")
    lines.append(
        _tline(
            "Stiefel–Whitney:",
            "w₁ = 0 (orientable)" if bool(w1_is_cob) else "w₁ ≠ 0 (non-orientable)",
        )
    )

    if n_tri == 0:
        lines.append(_tline("Euler class:", "0 (no 2-simplices)"))
        text = "\n".join(lines)
        if show:
            did_latex = False
            if mode in {"latex", "auto", "both"}:
                did_latex = _display_summary_latex(
                    classes,
                    quality=quality,
                    rounding_dist=rounding_dist,
                    e_zero_for_print=e_zero_for_print,
                    w1_is_zero=bool(w1_is_cob),
                )
            if mode == "both" or (mode == "text") or (mode == "auto" and not did_latex):
                print("\n" + text + "\n")
        return text

    if e_zero_for_print:
        bt = getattr(classes, "bundle_trivial_on_this_complex", None)
        if orientable:
            lines.append(_tline("Euler class:", "e = 0 (trivial)"))
        else:
            lines.append(_tline("(twisted) Euler class:", "ẽ = 0"))
        if bt is not None:
            lines.append(_tline("bundle trivial:", f"{bool(bt)}"))

        text = "\n".join(lines)
        if show:
            did_latex = False
            if mode in {"latex", "auto", "both"}:
                did_latex = _display_summary_latex(
                    classes,
                    quality=quality,
                    rounding_dist=rounding_dist,
                    e_zero_for_print=e_zero_for_print,
                    w1_is_zero=bool(w1_is_cob),
                )
            if mode == "both" or (mode == "text") or (mode == "auto" and not did_latex):
                print("\n" + text + "\n")
        return text

    if eZ is not None:
        k = abs(int(eZ))
        if orientable:
            parity_note = " (spin)" if (k % 2 == 0) else " (not spin)"
            lines.append(_tline("Euler number:", f"±{k}{parity_note}"))
        else:
            lines.append(_tline("(twisted) Euler number:", f"±{k}"))
    else:
        if orientable:
            lines.append(_tline("Euler class:", "e ≠ 0 (non-trivial)"))
            w2_is_zero = _infer_w2_is_zero()
            if w2_is_zero is not None:
                lines.append(_tline("Spin class:", "w₂ = 0 (spin)" if bool(w2_is_zero) else "w₂ ≠ 0 (not spin)"))
        else:
            lines.append(_tline("(twisted) Euler class:", "ẽ ≠ 0"))

    bt = getattr(classes, "bundle_trivial_on_this_complex", None)
    if bt is not None:
        lines.append(_tline("bundle trivial:", f"{bool(bt)}"))

    text = "\n".join(lines)

    if show:
        did_latex = False
        if mode in {"latex", "auto", "both"}:
            did_latex = _display_summary_latex(
                classes,
                quality=quality,
                rounding_dist=rounding_dist,
                e_zero_for_print=e_zero_for_print,
                w1_is_zero=bool(w1_is_cob),
            )
        if mode == "both" or (mode == "text") or (mode == "auto" and not did_latex):
            print("\n" + text + "\n")

    return text


def _display_summary_latex(
    classes,
    *,
    quality=None,
    rounding_dist: float,
    e_zero_for_print: bool,
    w1_is_zero: bool,
) -> bool:
    try:
        from IPython.display import display, Math  # type: ignore
    except Exception:
        return False

    eps_triv = getattr(quality, "eps_align_euc", None) if quality is not None else None
    eps_triv_mean = getattr(quality, "eps_align_euc_mean", None) if quality is not None else None
    delta = getattr(quality, "delta", None) if quality is not None else None
    alpha = getattr(quality, "alpha", None) if quality is not None else None
    eps_coc = getattr(quality, "cocycle_defect", None) if quality is not None else None

    orientable = bool(getattr(classes, "orientable", False))
    n_tri = int(getattr(classes, "n_triangles", 0))
    eZ = getattr(classes, "twisted_euler_number_Z", None)

    def _infer_w2_is_zero() -> Optional[bool]:
        spn = getattr(classes, "spin_on_this_complex", None)
        if spn is not None:
            return bool(spn)
        sw2 = getattr(classes, "sw2_Z2", None)
        if isinstance(sw2, dict) and len(sw2) > 0:
            return all((int(v) & 1) == 0 for v in sw2.values())
        return None

    def _latex_eps_with_geo_pi(d_euc: Optional[float], *, decimals: int = 3, mean: bool = False) -> str:
        if d_euc is None or not np.isfinite(float(d_euc)):
            return r"\text{—}"
        d_euc = float(d_euc)
        theta = _euc_to_geo_rad(d_euc)
        if theta is None:
            return f"{d_euc:.{decimals}f}"
        if mean:
            return (
                f"{d_euc:.{decimals}f}"
                + r"\ \left(\bar{\varepsilon}_{\text{triv}}^{\text{geo}}="
                + f"{theta/np.pi:.{decimals}f}"
                + r"\pi\right)"
            )
        return (
            f"{d_euc:.{decimals}f}"
            + r"\ \left(\varepsilon_{\text{triv}}^{\text{geo}}="
            + f"{theta/np.pi:.{decimals}f}"
            + r"\pi\right)"
        )

    diag_rows: List[Tuple[str, str]] = []
    if quality is None:
        diag_rows.append((r"\text{(no quality report provided)}", r""))
    else:
        if eps_triv is not None:
            diag_rows.append(
                (
                    r"\text{Trivialization error}",
                    r"\varepsilon_{\text{triv}} := "
                    r"\sup_{(j\,k)\in\mathcal{N}(\mathcal{U})}\sup_{x\in\pi^{-1}(U_j\cap U_k)} "
                    r"d_{\mathbb{C}}(\Omega_{jk}f_k(x),f_j(x))"
                    + r" = "
                    + _latex_eps_with_geo_pi(eps_triv, mean=False),
                )
            )
        if eps_triv_mean is not None:
            diag_rows.append(
                (
                    r"\text{Mean triv error}",
                    r"\bar{\varepsilon}_{\text{triv}}"
                    + r" = "
                    + _latex_eps_with_geo_pi(eps_triv_mean, mean=True),
                )
            )
        if delta is not None:
            diag_rows.append(
                (
                    r"\text{Surjectivity defect}",
                    r"\delta := \sup_{(i\,j\,k)\in\mathcal{N}(\mathcal{U})}\min_{v\in\{i,j,k\}} "
                    r"d_H\!\left(f_v(\pi^{-1}(U_i\cap U_j\cap U_k)),\mathbb{S}^1\right)"
                    + r" = "
                    + f"{float(delta):.3f}",
                )
            )
        if alpha is not None:
            if alpha == float("inf"):
                diag_rows.append((r"\text{Stability ratio}", r"\alpha := \varepsilon_{\text{triv}}/(1-\delta) = \infty"))
            else:
                diag_rows.append((r"\text{Stability ratio}", r"\alpha := \varepsilon_{\text{triv}}/(1-\delta) = " + f"{float(alpha):.3f}"))
        if eps_coc is not None:
            diag_rows.append(
                (
                    r"\text{Cocycle error}",
                    r"\varepsilon_{\mathrm{coc}} := "
                    r"\sup_{(i\,j\,k)\in\mathcal{N}(\mathcal{U})}\left\|\Omega_{ij}\Omega_{jk}\Omega_{ki}-I\right\|_F"
                    + r" = "
                    + f"{float(eps_coc):.3f}",
                )
            )
        diag_rows.append((r"\text{Euler rounding dist.}", r"d_\infty(\delta_\omega\theta,\tilde{e}) = " + f"{float(rounding_dist):.3f}"))

    class_rows: List[Tuple[str, str]] = []
    class_rows.append((r"\text{Stiefel--Whitney}", r"w_1 = 0\ (\text{orientable})" if w1_is_zero else r"w_1 \neq 0\ (\text{non-orientable})"))

    if n_tri == 0:
        class_rows.append((r"\text{Euler class}", r"e = 0\ \text{(no 2-simplices)}"))
    else:
        if e_zero_for_print:
            if orientable:
                class_rows.append((r"\text{Euler class}", r"e = 0\ (\text{trivial})"))
            else:
                class_rows.append((r"\text{(twisted) Euler class}", r"\tilde{e} = 0"))
        else:
            if eZ is not None:
                k = abs(int(eZ))
                if orientable:
                    parity_note = r"\ (\text{spin})" if (k % 2 == 0) else r"\ (\text{not spin})"
                    class_rows.append((r"\text{Euler number}", rf"\pm {k}" + parity_note))
                else:
                    class_rows.append((r"\text{(twisted) Euler number}", rf"\pm {k}"))
            else:
                if orientable:
                    class_rows.append((r"\text{Euler class}", r"e \neq 0\ (\text{non-trivial})"))
                    w2_is_zero = _infer_w2_is_zero()
                    if w2_is_zero is not None:
                        class_rows.append((r"\text{Spin class}", r"w_2 = 0\ (\text{spin})" if bool(w2_is_zero) else r"w_2 \neq 0\ (\text{not spin})"))
                else:
                    class_rows.append((r"\text{(twisted) Euler class}", r"\tilde{e} \neq 0"))

    def _rows_to_aligned(rows: List[Tuple[str, str]]) -> str:
        out: List[str] = []
        for label, expr in rows:
            if expr.strip() == "":
                out.append(r"\quad " + label + r" &")
            else:
                out.append(r"\quad " + label + r" &:\quad " + expr)
        return r"\\[3pt]".join(out)

    latex = (
        r"\begin{aligned}"
        r"\textbf{Diagnostics} & \\[6pt]"
        + _rows_to_aligned(diag_rows)
        + r"\\[14pt]"
        r"\textbf{Characteristic Classes} & \\[6pt]"
        + _rows_to_aligned(class_rows)
        + r"\end{aligned}"
    )

    try:
        display(Math(latex))
        return True
    except Exception:
        return False
