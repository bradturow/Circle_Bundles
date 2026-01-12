# characteristic_class.py
"""
Characteristic classes for O(2)-bundles on a nerve (supports up to 3-simplices).

Main entrypoint:
    compute_classes(cover, cocycle, ...)

Assumptions / conventions
-------------------------
- We work on a simplicial complex given by (edges, triangles, tets).
- Edges are represented canonically using combinatorics.canon_edge (min,max).
- Triangles are represented canonically using combinatorics.canon_tri (sorted).
- Tetrahedra are represented canonically as sorted 4-tuples.

- The O(2) cocycle object is assumed to provide:
    - restrict(edges) -> cocycle restricted to those edges
    - omega_Z2()      -> dict edge -> {0,1}  (det -> Z2)
    - omega_O1()      -> dict edge -> {+1,-1}
    - theta_normalized() -> dict edge -> float in R/Z (represented mod 1)
    - orient_if_possible(edges, n_vertices) -> (ok, cocycle_out, phi_pm1_array)

Notes
-----
- The twisted Euler representative e is computed by rounding δ_ω θ on triangles.
- If 3-simplices are present, we DO NOT assume e is a cocycle: we check δ_ω e on tets.
- Euler number computation:
    * If H_2(B; Z~) has free rank 1, compute integer pairing (twisted Euler number),
      regardless of orientability of the base.
    * If free rank != 1, do not compute integer pairing.
    * If H_2(B; Z2) has dimension 1, compute mod-2 pairing.
    * Otherwise, mod-2 pairing is not computed.
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set

import numpy as np
import sympy as sp

from .combinatorics import Edge, Tri, canon_edge, canon_tri

from .class_persistence import build_delta_C0_to_C1_Z2, build_delta_C1_to_C2_Z_twisted
from .class_persistence import in_image_mod2, in_image_Z_fast_pipeline


# ============================================================
# Canonicalization
# ============================================================

Tet = Tuple[int, int, int, int]


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

    Returns an integer.
    """
    i, j, k, l = tet

    w01 = int(omega_O1[canon_edge(i, j)])

    t_jkl = canon_tri(j, k, l)
    t_ikl = canon_tri(i, k, l)
    t_ijl = canon_tri(i, j, l)
    t_ijk = canon_tri(i, j, k)

    return int(w01 * int(euler.get(t_jkl, 0)) - int(euler.get(t_ikl, 0)) + int(euler.get(t_ijl, 0)) - int(euler.get(t_ijk, 0)))


def compute_twisted_euler_class(cocycle: Any, triangles: Iterable[Tri]):
    """
    cocycle: O2Cocycle-like object (expects theta_normalized() and omega_O1()).
    triangles: iterable of triangles (any order; we canonicalize).

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
# Boundary matrices (twisted over Z, untwisted over Z2)
# ============================================================

def build_twisted_boundary_C2_to_C1(
    edges: List[Edge],
    triangles: List[Tri],
    omega_O1: Dict[Edge, int],
) -> np.ndarray:
    """
    Twisted boundary ∂_2^ω : C2 -> C1 over Z.

    Consistent with:
      δ_ω θ(i,j,k) = ω(i,j)θ(j,k) - θ(i,k) + θ(i,j)

    So:
      ∂_ω [i,j,k] = ω(i,j)[j,k] - [i,k] + [i,j]
    """
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
    """
    Twisted boundary ∂_3^ω : C3 -> C2 over Z.

    Convention (i<j<k<l):
      ∂_ω [i,j,k,l] = ω(i,j)[j,k,l] - [i,k,l] + [i,j,l] - [i,j,k]
    """
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

        # require all faces present in triangle basis
        for face in (t_jkl, t_ikl, t_ijl, t_ijk):
            if face not in tri_index:
                raise KeyError(f"Tetrahedron face {face} not found in triangles list; ensure triangles include all faces.")

        D[tri_index[t_jkl], c] += w01
        D[tri_index[t_ikl], c] += -1
        D[tri_index[t_ijl], c] += +1
        D[tri_index[t_ijk], c] += -1

    return D


def build_boundary_mod2_C2_to_C1(edges: List[Edge], triangles: List[Tri]) -> np.ndarray:
    """
    Standard boundary ∂_2 : C2 -> C1 over Z2 (represented as uint8 matrix).
    ∂[i,j,k] = [j,k] + [i,k] + [i,j] mod 2 (signs vanish in Z2).
    """
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
    """
    Standard boundary ∂_3 : C3 -> C2 over Z2 (uint8).
    ∂[i,j,k,l] = [j,k,l] + [i,k,l] + [i,j,l] + [i,j,k] mod 2
    """
    triangles = [canon_tri(*t) for t in triangles]
    tets = [canon_tet(*tt) for tt in tets]
    tri_index = {t: r for r, t in enumerate(triangles)}

    D = np.zeros((len(triangles), len(tets)), dtype=np.uint8)
    for c, (i, j, k, l) in enumerate(tets):
        for face in (canon_tri(j, k, l), canon_tri(i, k, l), canon_tri(i, j, l), canon_tri(i, j, k)):
            if face not in tri_index:
                raise KeyError(f"Tet face {face} not found in triangles list; ensure triangles include all faces.")
            D[tri_index[face], c] ^= 1
    return D


# ============================================================
# Linear algebra helpers (Q and mod2)
# ============================================================

def rank_over_Q(A: np.ndarray) -> int:
    if A.size == 0:
        return 0
    return int(sp.Matrix(A).rank())


def nullspace_over_Q(A: np.ndarray) -> List[sp.Matrix]:
    if A.size == 0:
        # nullspace of 0xN is full space; caller should handle dimensions carefully
        return sp.Matrix(A).nullspace()
    return sp.Matrix(A).nullspace()


def in_colspace_over_Q(M: sp.Matrix, v: sp.Matrix) -> bool:
    """Check if v is in the column space of M over Q."""
    if M.cols == 0:
        return False
    if v.cols != 1:
        v = sp.Matrix(v).reshape(v.rows, 1)
    return int(M.row_join(v).rank()) == int(M.rank())


def rref_mod2(A_in: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Compute RREF of A over GF(2). Returns (RREF_matrix, pivot_cols).
    """
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
    """
    Return a basis for nullspace of A over GF(2).
    A shape: (m,n). Returns list of length nullity, each vector length n (uint8).
    """
    A = (np.asarray(A_in, dtype=np.uint8) & 1).copy()
    m, n = A.shape
    R, pivots = rref_mod2(A)
    pivot_set = set(pivots)
    free_cols = [c for c in range(n) if c not in pivot_set]

    basis: List[np.ndarray] = []
    for free in free_cols:
        x = np.zeros(n, dtype=np.uint8)
        x[free] = 1
        # back substitute (R is RREF, pivot i on row i)
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


def in_image_mod2(A: np.ndarray, b: np.ndarray) -> bool:
    """Decide if b is in the column space of A over GF(2)."""
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    b = (np.asarray(b, dtype=np.uint8) & 1).reshape(-1, 1)

    M = np.concatenate([A, b], axis=1)
    m, n1 = M.shape
    n = n1 - 1

    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if M[r, col] == 1:
                piv = r
                break
        if piv is None:
            continue

        if piv != row:
            M[[row, piv]] = M[[piv, row]]

        for r in range(m):
            if r != row and M[r, col] == 1:
                M[r, :] ^= M[row, :]

        row += 1
        if row == m:
            break

    for r in range(m):
        if np.all(M[r, :n] == 0) and M[r, n] == 1:
            return False
    return True


# ============================================================
# Fundamental class + pairing in general 3D case
# ============================================================

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
    """
    Compute:
      dim_Q = dim H_2(B; Z~) over Q (free rank)
      dim_Z2 = dim H_2(B; Z2)
    using boundaries ∂2 and ∂3.
    """
    edges = sorted({canon_edge(*e) for e in edges})
    triangles = sorted({canon_tri(*t) for t in triangles})
    tets = sorted({canon_tet(*tt) for tt in tets})

    # Twisted (Z~) over Q
    if len(triangles) == 0:
        dim_Q = 0
        dim_Z2 = 0
        return {"dim_Q": dim_Q, "dim_Z2": dim_Z2}

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

    # Mod 2 (untwisted)
    D2 = build_boundary_mod2_C2_to_C1(edges, triangles)
    R2, piv2 = rref_mod2(D2)
    ker2_dim2 = n2 - len(piv2)

    if len(tets) == 0:
        im3_dim2 = 0
    else:
        D3 = build_boundary_mod2_C3_to_C2(triangles, tets)
        # image dim is rank of D3 over GF2; rank is #pivots of RREF of D3
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
    """
    If dim H2(B; Z~) over Q is 1, return an integer 2-cycle z (length #triangles)
    representing the generator (up to sign).
    Otherwise return None.
    """
    edges = sorted({canon_edge(*e) for e in edges})
    triangles = sorted({canon_tri(*t) for t in triangles})
    tets = sorted({canon_tet(*tt) for tt in tets})

    if len(triangles) == 0:
        return None

    D2w = sp.Matrix(build_twisted_boundary_C2_to_C1(edges, triangles, omega_O1=omega_O1))
    ns = D2w.nullspace()  # basis for Z2 cycles over Q (actually kernel of ∂2)

    if len(ns) == 0:
        return None

    if len(tets) == 0:
        # H2 dim = dim ker(∂2)
        if len(ns) != 1:
            return None
        v = ns[0]
    else:
        D3w = sp.Matrix(build_twisted_boundary_C3_to_C2(triangles, tets, omega_O1=omega_O1))
        # pick a kernel vector not in image(∂3)
        candidates = []
        for v0 in ns:
            if not in_colspace_over_Q(D3w, v0):
                candidates.append(v0)
        if len(candidates) != 1:
            # Either dim != 1 or ambiguity; we refuse.
            return None
        v = candidates[0]

    # clear denominators to integer
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
    """
    If dim H2(B; Z2) is 1, return a mod-2 2-cycle vector z2 (length #triangles)
    representing the generator class (up to sign, but sign is irrelevant mod 2).
    Otherwise return None.
    """
    edges = sorted({canon_edge(*e) for e in edges})
    triangles = sorted({canon_tri(*t) for t in triangles})
    tets = sorted({canon_tet(*tt) for tt in tets})

    if len(triangles) == 0:
        return None

    D2 = build_boundary_mod2_C2_to_C1(edges, triangles)  # (E x T)
    # cycles are vectors in ker(D2)
    ker_basis = nullspace_basis_mod2(D2)

    if len(ker_basis) == 0:
        return None

    if len(tets) == 0:
        # H2 dim = dim ker
        if len(ker_basis) != 1:
            return None
        return ker_basis[0]

    D3 = build_boundary_mod2_C3_to_C2(triangles, tets)  # (T x Tet)
    # image(D3) is column space in T-dim
    # find a kernel vector not in image(D3)
    candidates = []
    for v in ker_basis:
        if not in_image_mod2(D3, v):  # v in colspace(D3)?
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
# Class computation + reporting
# ============================================================

@dataclass
class ClassResult:
    n_vertices: int
    n_edges: int
    n_triangles: int
    n_tets: int

    sw1_Z2: Dict[Edge, int]
    sw1_O1: Dict[Edge, int]

    # whether ω passed the triangle cocycle check on this complex
    sw1_is_cocycle: Optional[bool]

    # ω is a coboundary in C^1 over Z2 (i.e. w1 = 0) on this complex
    sw1_is_coboundary_Z2: Optional[bool]

    orientable: bool
    phi_pm1: Optional[np.ndarray]

    cocycle_used: Any

    euler_class: Dict[Tri, int]
    euler_class_real: Dict[Tri, float]
    rounding_dist: float
    omega_O1_used: Dict[Edge, int]

    # Euler cocycle check in 3D (δ_ω e = 0 on tets). None if no tets.
    euler_is_cocycle: Optional[bool]
    max_abs_delta_euler_on_tets: Optional[int]

    # e is a twisted coboundary in C^2 over Z (i.e. e=0 or ẽ=0) on this complex
    euler_is_coboundary_Z: Optional[bool]

    # Homology info
    H2_dim_Q: Optional[int]
    H2_dim_Z2: Optional[int]

    # Pairings / invariants
    twisted_euler_number_Z: Optional[int]
    euler_number_mod2: Optional[int]

    sw2_Z2: Dict[Tri, int]

    bundle_trivial_on_this_complex: Optional[bool]
    spin_on_this_complex: Optional[bool]


        
def _sw1_cocycle_check_on_triangles(triangles: List[Tri], omega_Z2: Dict[Edge, int]) -> bool:
    """
    Check δω = 0 on every triangle (mod 2):
      δω(i,j,k) = ω(j,k) - ω(i,k) + ω(i,j)  (mod 2)
    """
    for (i, j, k) in triangles:
        eij = canon_edge(i, j)
        eik = canon_edge(i, k)
        ejk = canon_edge(j, k)
        val = (int(omega_Z2.get(ejk, 0)) - int(omega_Z2.get(eik, 0)) + int(omega_Z2.get(eij, 0))) & 1
        if val != 0:
            return False
    return True


def _infer_tets_from_cover(cover: Any) -> List[Tet]:
    """
    Try common method names; fall back to empty.
    """
    if hasattr(cover, "nerve_tetrahedra"):
        return [canon_tet(*tt) for tt in cover.nerve_tetrahedra()]
    if hasattr(cover, "nerve_tets"):
        return [canon_tet(*tt) for tt in cover.nerve_tets()]
    if hasattr(cover, "nerve_3simplices"):
        return [canon_tet(*tt) for tt in cover.nerve_3simplices()]
    if hasattr(cover, "nerve_simplices"):
        # sometimes nerve_simplices(dim) exists
        try:
            tets = cover.nerve_simplices(3)
            return [canon_tet(*tt) for tt in tets]
        except Exception:
            pass
    return []


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
    if edges is None:
        edges = cover.nerve_edges()
    if triangles is None:
        triangles = cover.nerve_triangles()

    edges_list = sorted({canon_edge(*e) for e in edges})
    tris_list = sorted({canon_tri(*t) for t in triangles})

    if tets is None:
        tets_list = _infer_tets_from_cover(cover)
    else:
        tets_list = sorted({canon_tet(*tt) for tt in tets})

    if n_vertices is None:
        mx = -1
        for (a, b) in edges_list:
            mx = max(mx, a, b)
        for (a, b, c) in tris_list:
            mx = max(mx, a, b, c)
        for (a, b, c, d) in tets_list:
            mx = max(mx, a, b, c, d)
        n_vertices = (mx + 1) if mx >= 0 else int(cover.U.shape[0])

    # Restrict cocycle to this 1-skeleton
    coc_res = cocycle.restrict(edges_list)

    sw1_Z2 = {canon_edge(*e): int(v) & 1 for e, v in coc_res.omega_Z2().items()}
    sw1_O1 = canonicalize_o1_cochain(coc_res.omega_O1())

    # ---- SW1 cocycle check on triangles (δω=0 on each 2-simplex) ----
    sw1_is_cocycle: Optional[bool] = None
    if len(tris_list) > 0:
        sw1_is_cocycle = bool(_sw1_cocycle_check_on_triangles(tris_list, sw1_Z2))

    # ---- NEW: SW1 coboundary check on edges (ω in Im δ: C^0->C^1 over Z2) ----
    sw1_is_coboundary_Z2: Optional[bool] = None
    if len(edges_list) > 0:
        # vertices as 0-simplices
        V0 = [(i,) for i in range(int(n_vertices))]
        A01 = build_delta_C0_to_C1_Z2(vertices=V0, edges=edges_list)  # (#E, #V)
        b1 = np.array([sw1_Z2.get(e, 0) & 1 for e in edges_list], dtype=np.uint8)
        sw1_is_coboundary_Z2 = bool(in_image_mod2(A01, b1))

    # Try to orient (may modify cocycle)
    orientable = False
    phi_pm1 = None
    coc_used = coc_res

    if try_orient and len(edges_list) > 0:
        ok, coc_oriented, phi_pm1 = coc_res.orient_if_possible(edges_list, n_vertices=n_vertices)
        orientable = bool(ok)
        coc_used = coc_oriented if orientable else coc_res

    # ============================================================
    # Euler representative (triangle level)
    # ============================================================
    if len(tris_list) == 0:
        euler_rep: Dict[Tri, int] = {}
        euler_real: Dict[Tri, float] = {}
        rounding_dist = 0.0
        omega_O1_used = canonicalize_o1_cochain(coc_used.omega_O1())
        sw2: Dict[Tri, int] = {}

        euler_is_cocycle = None
        max_abs_delta = None

        H2_dim_Q = None
        H2_dim_Z2 = None
        eZ = None
        eZ2 = None

        # no triangles => no Euler class rep => no coboundary test
        euler_is_coboundary_Z: Optional[bool] = None

        bundle_trivial = None
        spin = None

    else:
        euler_rep, rounding_dist, euler_real, omega_O1_used = compute_twisted_euler_class(coc_used, tris_list)

        # restrict ω to exactly edges in this complex
        omega_O1_used = {e: int(omega_O1_used.get(e, 1)) for e in edges_list}

        # sw2 on triangles is e mod 2 (cochain-level)
        sw2 = {tri: (int(val) & 1) for tri, val in euler_rep.items()}

        # --- w2 coboundary check (spin) over Z2 ---
        sw2_is_coboundary_Z2: Optional[bool] = None
        if orientable and (len(tris_list) > 0) and (len(edges_list) > 0):
            # δ: C^1 -> C^2 over Z2 is the transpose of ∂2: C2 -> C1
            D2_mod2 = build_boundary_mod2_C2_to_C1(edges_list, tris_list)   # (E x T)
            A12_mod2 = (D2_mod2.T).astype(np.uint8)                         # (T x E)

            b2 = np.array([sw2.get(t, 0) & 1 for t in tris_list], dtype=np.uint8)  # (T,)
            sw2_is_coboundary_Z2 = bool(in_image_mod2(A12_mod2, b2))
        
        
        
        # ---- 3D cocycle check for Euler class if tets exist ----
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

        # ---- Euler coboundary test (e in Im δ_ω: C^1->C^2 over Z) ----
        euler_is_coboundary_Z: Optional[bool] = None
        if len(tris_list) > 0 and len(edges_list) > 0:
            A12 = build_delta_C1_to_C2_Z_twisted(edges=edges_list, triangles=tris_list, omega_O1=omega_O1_used).astype(np.int64)
            b2 = np.array([int(euler_rep.get(t, 0)) for t in tris_list], dtype=np.int64)

            # Only meaningful if Euler rep is a valid 3D cocycle (or we are in 2D)
            ok_e = True if (euler_is_cocycle is None) else bool(euler_is_cocycle)
            if ok_e:
                euler_is_coboundary_Z = bool(in_image_Z_fast_pipeline(A12, b2))
            else:
                euler_is_coboundary_Z = None  # not a cocycle in 3D, so "class" undefined here

        # decide whether to compute pairings
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

                # pairing conventions / availability logic
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

        # ------------------------------------------------------------
        # Quick “bundle trivial / spin” flags (use class-level tests)
        # ------------------------------------------------------------
        if len(tets_list) == 0:
            bundle_trivial = bool(orientable and (euler_is_coboundary_Z is True))

        else:
            bundle_trivial = bool(
                orientable and (euler_is_cocycle is True) and (euler_is_coboundary_Z is True)
            )

        spin = bool(sw2_is_coboundary_Z2) if sw2_is_coboundary_Z2 is not None else None

            
            
    return ClassResult(
        n_vertices=int(n_vertices),
        n_edges=int(len(edges_list)),
        n_triangles=int(len(tris_list)),
        n_tets=int(len(tets_list)),

        sw1_Z2=sw1_Z2,
        sw1_O1=sw1_O1,
        sw1_is_cocycle=sw1_is_cocycle,

        # NEW fields (make sure your ClassResult dataclass includes them)
        sw1_is_coboundary_Z2=sw1_is_coboundary_Z2,
        euler_is_coboundary_Z=euler_is_coboundary_Z,

        orientable=orientable,
        phi_pm1=phi_pm1,
        cocycle_used=coc_used,

        euler_class=euler_rep,
        euler_class_real=euler_real,
        rounding_dist=float(rounding_dist),
        omega_O1_used=omega_O1_used,

        euler_is_cocycle=euler_is_cocycle,
        max_abs_delta_euler_on_tets=max_abs_delta,

        H2_dim_Q=H2_dim_Q,
        H2_dim_Z2=H2_dim_Z2,

        twisted_euler_number_Z=eZ,
        euler_number_mod2=eZ2,

        sw2_Z2=sw2,

        bundle_trivial_on_this_complex=bundle_trivial,
        spin_on_this_complex=spin,
    )




def _euc_to_geo_rad(d: Optional[float]) -> Optional[float]:
    """
    Convert chordal distance d in [0,2] on S^1 ⊂ C to geodesic angle in radians in [0, π]:
        d = 2 sin(theta/2)  =>  theta = 2 arcsin(d/2).
    """
    if d is None:
        return None
    d = float(d)
    if not np.isfinite(d):
        return None
    x = np.clip(d / 2.0, 0.0, 1.0)
    return float(2.0 * np.arcsin(x))


def _fmt_euc_with_geo_pi(d_euc: Optional[float], *, decimals: int = 3) -> str:
    """
    Render:
        <d_euc> (ε_triv^{geo}=<theta/pi>π)
    where <theta> is the geodesic angle in radians derived from chordal d_euc.
    """
    if d_euc is None or not np.isfinite(float(d_euc)):
        return "—"
    d_euc = float(d_euc)
    theta = _euc_to_geo_rad(d_euc)
    if theta is None:
        return f"{d_euc:.{decimals}f}"
    return f"{d_euc:.{decimals}f} (\\varepsilon_{{\\text{{triv}}}}^{{\\text{{geo}}}}={theta/np.pi:.{decimals}f}\\pi)"


def _fmt_mean_euc_with_geo_pi(d_euc_mean: Optional[float], *, decimals: int = 3) -> str:
    """
    Render:
        <d_euc_mean> (\bar{ε}_triv^{geo}=<theta/pi>π)
    """
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
    (Updated: remove RMS lines; use d_C; use ε_triv (chordal) and show geo in π-parens;
     use \bar{ε}_triv for mean triv error.)
    """
    def _describe_H2_iso_tag(c) -> Optional[str]:
        H2_Q = getattr(c, "H2_dim_Q", None)
        H2_Z2 = getattr(c, "H2_dim_Z2", None)
        if H2_Q is None or H2_Z2 is None:
            return None
        H2_Q = int(H2_Q)
        H2_Z2 = int(H2_Z2)
        if H2_Q == 0 and H2_Z2 == 0:
            return "0"
        if H2_Q == 0 and H2_Z2 == 1:
            return "Z2"
        if H2_Q == 1 and H2_Z2 == 1:
            return "Z"
        return "non-cyclic"

    def _pretty_H2_text(tag: str) -> str:
        return {"0": "0", "Z2": "ℤ₂", "Z": "ℤ", "non-cyclic": "non-cyclic"}[tag]

    # ----------------------------
    # Pull quantities (UPDATED)
    # ----------------------------
    eps_triv = getattr(quality, "eps_align_euc", None) if quality is not None else None
    eps_triv_mean = getattr(quality, "eps_align_euc_mean", None) if quality is not None else None

    delta = getattr(quality, "delta", None) if quality is not None else None
    alpha = getattr(quality, "alpha", None) if quality is not None else None
    eps_coc = getattr(quality, "cocycle_defect", None) if quality is not None else None

    rounding_dist = float(getattr(classes, "rounding_dist", 0.0))
    orientable = bool(getattr(classes, "orientable", False))
    n_tri = int(getattr(classes, "n_triangles", 0))

    # “is coboundary?” flags
    w1_is_cob = getattr(classes, "sw1_is_coboundary_Z2", None)
    if w1_is_cob is None:
        w1_is_cob = bool(orientable)  # fallback

    e_is_cob = getattr(classes, "euler_is_coboundary_Z", None)
    euler_rep = getattr(classes, "euler_class", {}) or {}
    e_trivial_cochain = all(int(v) == 0 for v in euler_rep.values()) if euler_rep else True
    e_zero_for_print = bool(e_is_cob) if e_is_cob is not None else bool(e_trivial_cochain)

    H2_tag = _describe_H2_iso_tag(classes)

    # ----------------------------
    # Build plain-text summary
    # ----------------------------
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
            lines.append(_tline(
                "trivialization error:",
                "ε_triv := sup_{(j k)∈N(U)} sup_{x∈π^{-1}(U_j∩U_k)} d_𝕮(Ω_{jk} f_k(x), f_j(x))"
                f" = {_fmt_euc_with_geo_pi(eps_triv)}"
            ))
        if eps_triv_mean is not None:
            lines.append(_tline(
                "mean triv error:",
                f"\\bar{{ε}}_triv = {_fmt_mean_euc_with_geo_pi(eps_triv_mean)}"
            ))
        if delta is not None:
            lines.append(_tline(
                "surjectivity defect:",
                "δ := sup_{(i j k)∈N(U)} min_{v∈{i,j,k}} d_H(f_v(π^{-1}(U_i∩U_j∩U_k)), S^1)"
                f" = {float(delta):.3f}"
            ))
        if alpha is not None:
            if alpha == float("inf"):
                lines.append(_tline("stability ratio:", "α := ε_triv/(1-δ) = ∞  (since δ ≥ 1)"))
            else:
                lines.append(_tline("stability ratio:", f"α := ε_triv/(1-δ) = {float(alpha):.3f}"))
        if eps_coc is not None:
            lines.append(_tline(
                "cocycle error:",
                "ε_coc := sup_{(i j k)∈N(U)} ‖Ω_{ij}Ω_{jk}Ω_{ki} - I‖_F"
                f" = {float(eps_coc):.3f}"
            ))

        lines.append(_tline(
            "Euler rounding diag:",
            f"d_∞(δ_ω θ, ẽ) = {rounding_dist:.6g}"
        ))

    lines.append("")
    lines.append("=== Characteristic Classes ===")
    lines.append(_tline(
        "Stiefel–Whitney:",
        "w₁ = 0 (orientable)" if bool(w1_is_cob) else "w₁ ≠ 0 (non-orientable)"
    ))

    if n_tri == 0:
        lines.append(_tline("Euler class:", "(no 2-simplices) -> not computed"))
        text = "\n".join(lines)
        if show:
            did_latex = False
            if mode in {"latex", "auto", "both"}:
                did_latex = _display_summary_latex(
                    classes,
                    quality=quality,
                    rounding_dist=rounding_dist,
                    H2_tag=H2_tag,
                    e_zero_for_print=e_zero_for_print,
                    w1_is_zero=bool(w1_is_cob),
                )
            if mode == "both" or (mode == "text") or (mode == "auto" and not did_latex):
                print("\n" + text + "\n")
        return text

    if orientable:
        lines.append(_tline("Euler class:", "e = 0 (trivial)" if e_zero_for_print else "e ≠ 0 (non-trivial)"))
    else:
        lines.append(_tline("Euler class:", "ẽ = 0" if e_zero_for_print else "ẽ ≠ 0"))

    if e_zero_for_print:
        bt = getattr(classes, "bundle_trivial_on_this_complex", None)
        if bt is not None:
            lines.append(_tline("bundle trivial:", f"{bool(bt)}"))
        if orientable:
            sp = getattr(classes, "spin_on_this_complex", None)
            if sp is not None:
                lines.append(_tline("spin:", f"{bool(sp)}"))
        text = "\n".join(lines)
        if show:
            did_latex = False
            if mode in {"latex", "auto", "both"}:
                did_latex = _display_summary_latex(
                    classes,
                    quality=quality,
                    rounding_dist=rounding_dist,
                    H2_tag=H2_tag,
                    e_zero_for_print=e_zero_for_print,
                    w1_is_zero=bool(w1_is_cob),
                )
            if mode == "both" or (mode == "text") or (mode == "auto" and not did_latex):
                print("\n" + text + "\n")
        return text

    if H2_tag is not None:
        lines.append(_tline("Second homology:", f"Ĥ₂(U; Z_ω) ≅ {_pretty_H2_text(H2_tag)}"))

    eZ = getattr(classes, "twisted_euler_number_Z", None)
    if eZ is not None:
        lines.append(_tline("Euler number:", f"⟨{'e' if orientable else 'ẽ'}, [U]⟩ = {eZ}"))

    if orientable:
        spin_flag = getattr(classes, "spin_on_this_complex", None)
        if spin_flag is not None:
            lines.append(_tline("mod 2 info:", "w₂ = 0 (spin)" if bool(spin_flag) else "w₂ ≠ 0 (not spin)"))

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
                H2_tag=H2_tag,
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
    H2_tag: Optional[str],
    e_zero_for_print: bool,
    w1_is_zero: bool,
) -> bool:
    """
    Best-effort IPython Math display (UPDATED formatting).
    Returns True iff LaTeX display succeeded.
    """
    try:
        from IPython.display import display, Math  # type: ignore
    except Exception:
        return False

    def _latex_H2_iso(tag: str) -> str:
        if tag == "0":
            return r"0"
        if tag == "Z2":
            return r"\mathbb{Z}_2"
        if tag == "Z":
            return r"\mathbb{Z}"
        return r"\text{non-cyclic}"

    # ----------------------------
    # Pull diagnostics (UPDATED)
    # ----------------------------
    eps_triv = getattr(quality, "eps_align_euc", None) if quality is not None else None
    eps_triv_mean = getattr(quality, "eps_align_euc_mean", None) if quality is not None else None

    delta = getattr(quality, "delta", None) if quality is not None else None
    alpha = getattr(quality, "alpha", None) if quality is not None else None
    eps_coc = getattr(quality, "cocycle_defect", None) if quality is not None else None

    # ----------------------------
    # Class quantities
    # ----------------------------
    orientable = bool(getattr(classes, "orientable", False))
    n_tri = int(getattr(classes, "n_triangles", 0))
    eZ = getattr(classes, "twisted_euler_number_Z", None)
    eZ2 = getattr(classes, "euler_number_mod2", None)

    w2_trivial = getattr(classes, "spin_on_this_complex", None)
    if w2_trivial is None:
        w2_trivial = all((int(v) & 1) == 0 for v in (getattr(classes, "sw2_Z2", {}) or {}).values())

    e_sym_round = r"\tilde{e}"
    e_sym = r"e" if orientable else r"\tilde{e}"

    # helpers for latex numeric strings with geo-in-π parentheses
    def _latex_eps_with_geo_pi(d_euc: Optional[float], *, decimals: int = 3, mean: bool = False) -> str:
        if d_euc is None or not np.isfinite(float(d_euc)):
            return r"\text{—}"
        d_euc = float(d_euc)
        theta = _euc_to_geo_rad(d_euc)
        if theta is None:
            return f"{d_euc:.{decimals}f}"
        if mean:
            return f"{d_euc:.{decimals}f}" + r"\ \left(\bar{\varepsilon}_{\text{triv}}^{\text{geo}}=" + f"{theta/np.pi:.{decimals}f}" + r"\pi\right)"
        return f"{d_euc:.{decimals}f}" + r"\ \left(\varepsilon_{\text{triv}}^{\text{geo}}=" + f"{theta/np.pi:.{decimals}f}" + r"\pi\right)"

    # ----------------------------
    # Rows
    # ----------------------------
    diag_rows: List[Tuple[str, str]] = []
    if quality is None:
        diag_rows.append((r"\text{(no quality report provided)}", r""))
    else:
        if eps_triv is not None:
            diag_rows.append((
                r"\text{Trivialization error}",
                r"\varepsilon_{\text{triv}} := "
                r"\sup_{(j\,k)\in\mathcal{N}(\mathcal{U})}\sup_{x\in\pi^{-1}(U_j\cap U_k)} "
                r"d_{\mathbb{C}}(\Omega_{jk}f_k(x),f_j(x))"
                + r" = " + _latex_eps_with_geo_pi(eps_triv, mean=False)
            ))
        if eps_triv_mean is not None:
            diag_rows.append((
                r"\text{Mean triv error}",
                r"\bar{\varepsilon}_{\text{triv}}"
                + r" = " + _latex_eps_with_geo_pi(eps_triv_mean, mean=True)
            ))
        if delta is not None:
            diag_rows.append((
                r"\text{Surjectivity defect}",
                r"\delta := \sup_{(i\,j\,k)\in\mathcal{N}(\mathcal{U})}\min_{v\in\{i,j,k\}} "
                r"d_H\!\left(f_v(\pi^{-1}(U_i\cap U_j\cap U_k)),\mathbb{S}^1\right)"
                + r" = " + f"{float(delta):.3f}"
            ))
        if alpha is not None:
            if alpha == float("inf"):
                diag_rows.append((r"\text{Stability ratio}", r"\alpha := \varepsilon_{\text{triv}}/(1-\delta) = \infty"))
            else:
                diag_rows.append((r"\text{Stability ratio}", r"\alpha := \varepsilon_{\text{triv}}/(1-\delta) = " + f"{float(alpha):.3f}"))
        if eps_coc is not None:
            diag_rows.append((
                r"\text{Cocycle error}",
                r"\varepsilon_{\mathrm{coc}} := "
                r"\sup_{(i\,j\,k)\in\mathcal{N}(\mathcal{U})}\left\|\Omega_{ij}\Omega_{jk}\Omega_{ki}-I\right\|_F"
                + r" = " + f"{float(eps_coc):.3f}"
            ))

        diag_rows.append((
            r"\text{Euler rounding dist.}",
            r"d_\infty(\delta_\omega\theta," + e_sym_round + r") = " + f"{float(rounding_dist):.3f}"
        ))

    class_rows: List[Tuple[str, str]] = []
    class_rows.append((
        r"\text{Stiefel-Whitney}",
        r"w_1 = 0\ (\text{orientable})" if w1_is_zero else r"w_1 \neq 0\ (\text{non-orientable})"
    ))

    if n_tri == 0:
        class_rows.append((r"\text{Euler class}", r"\text{(no 2-simplices)}"))
    else:
        if orientable:
            class_rows.append((r"\text{Euler class}", r"e = 0\ (\text{trivial})" if e_zero_for_print else r"e \neq 0\ (\text{non-trivial})"))
        else:
            class_rows.append((r"\text{Euler class}", r"\tilde{e} = 0" if e_zero_for_print else r"\tilde{e} \neq 0"))

        if not e_zero_for_print:
            if H2_tag is not None:
                class_rows.append((
                    r"\text{Second homology}",
                    r"\check{H}_2(\mathcal{U};\mathbb{Z}_\omega)\cong " + _latex_H2_iso(H2_tag)
                ))

            if eZ is not None:
                class_rows.append((
                    r"\text{Euler number}",
                    r"\langle " + e_sym + r",[\mathcal{U}]\rangle = " + str(int(eZ))
                ))

            if orientable:
                class_rows.append((
                    r"\text{mod 2 info}",
                    r"w_2 = 0\ (\text{spin})" if w2_trivial else r"w_2 \neq 0\ (\text{not spin})"
                ))
                if (eZ is None) and (eZ2 is not None):
                    class_rows.append((
                        r"\text{mod 2 pairing}",
                        r"\langle w_2,[\mathcal{U}]\rangle = " + str(int(eZ2) & 1) + r"\ (\mathrm{mod}\ 2)"
                    ))
            else:
                if (eZ is None) and (eZ2 is not None):
                    class_rows.append((
                        r"\text{mod 2 pairing}",
                        r"\langle " + e_sym + r",[\mathcal{U}]\rangle = " + str(int(eZ2) & 1) + r"\ (\mathrm{mod}\ 2)"
                    ))

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
