# class_persistence.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Literal

import numpy as np

from .combinatorics import Edge, Tri, canon_edge, canon_tri

Simp = Tuple[int, ...]  # generic simplex as sorted tuple
Tet = Tuple[int, int, int, int]


# ============================================================
# Canonicalization / simplex helpers
# ============================================================

def canon_simplex(sig: Iterable[int]) -> Simp:
    return tuple(sorted(int(x) for x in sig))


def simplex_dim(sig: Simp) -> int:
    return len(sig) - 1


def canon_edge_tuple(e: Edge) -> Edge:
    return canon_edge(int(e[0]), int(e[1]))


def canon_tri_tuple(t: Tri) -> Tri:
    return canon_tri(int(t[0]), int(t[1]), int(t[2]))


def canon_tet_tuple(t: Tet) -> Tet:
    a, b, c, d = (int(x) for x in t)
    tt = tuple(sorted((a, b, c, d)))
    return (tt[0], tt[1], tt[2], tt[3])


# ============================================================
# Results
# ============================================================

@dataclass
class CobirthResult:
    # edge-driven: k_removed edges have been removed when property first holds
    k_removed: int
    cutoff_weight: float
    removed_edges: List[Edge]


@dataclass
class CodeathResult:
    k_removed: int
    cutoff_weight: float
    removed_edges: List[Edge]


# ============================================================
# Coboundary matrices
# ============================================================

def build_delta_C0_to_C1_Z2(vertices: List[Simp], edges: List[Edge]) -> np.ndarray:
    """
    δ: C^0 -> C^1 over Z2, oriented edge convention i<j:
      (δφ)(i,j) = φ(j) - φ(i)  (mod 2).

    Matrix A has shape (#edges, #vertices).
    """
    verts = [canon_simplex(v) for v in vertices]
    if any(simplex_dim(v) != 0 for v in verts):
        raise ValueError("vertices must be 0-simplices (singletons).")

    v_index = {v[0]: c for c, v in enumerate(verts)}
    edges = [canon_edge_tuple(e) for e in edges]

    m = len(edges)
    n = len(verts)
    A = np.zeros((m, n), dtype=np.uint8)

    for r, (i, j) in enumerate(edges):
        A[r, v_index[i]] ^= 1
        A[r, v_index[j]] ^= 1
    return A


def build_delta_C1_to_C2_Z_twisted(
    edges: List[Edge],
    triangles: List[Tri],
    omega_O1: Dict[Edge, int],
) -> np.ndarray:
    """
    δ_ω: C^1 -> C^2 over Z using convention:
      (δ_ω θ)(i,j,k) = ω(i,j) θ(j,k) - θ(i,k) + θ(i,j).

    Matrix A has shape (#triangles, #edges): A * theta_vec = delta_theta_vec.
    """
    edges = [canon_edge_tuple(e) for e in edges]
    triangles = [canon_tri_tuple(t) for t in triangles]
    e_index = {e: c for c, e in enumerate(edges)}

    omega_O1 = {canon_edge_tuple(e): int(v) for e, v in omega_O1.items()}

    for e in edges:
        if e not in omega_O1:
            raise KeyError(f"omega_O1 missing required edge {e}")

    A = np.zeros((len(triangles), len(edges)), dtype=np.int64)

    for r, (i, j, k) in enumerate(triangles):
        eij = canon_edge(i, j)
        ejk = canon_edge(j, k)
        eik = canon_edge(i, k)

        w01 = int(omega_O1[eij])  # ±1
        A[r, e_index[ejk]] += w01
        A[r, e_index[eik]] += -1
        A[r, e_index[eij]] += +1

    return A


def build_delta_C2_to_C3_Z_twisted(
    triangles: List[Tri],
    tets: List[Tet],
    omega_O1: Dict[Edge, int],
) -> np.ndarray:
    """
    δ_ω: C^2 -> C^3 over Z on tetrahedra, consistent with our C^1->C^2 convention.

    For i<j<k<l:
      (δ_ω c)(i,j,k,l) =
          ω(i,j) c(j,k,l) - c(i,k,l) + c(i,j,l) - c(i,j,k).

    Matrix A has shape (#tets, #tris): A * c_vec = delta_c_vec.
    """
    triangles = [canon_tri_tuple(t) for t in triangles]
    tets = [canon_tet_tuple(tt) for tt in tets]
    tri_index = {t: c for c, t in enumerate(triangles)}

    omega_O1 = {canon_edge_tuple(e): int(v) for e, v in omega_O1.items()}

    A = np.zeros((len(tets), len(triangles)), dtype=np.int64)

    for r, (i, j, k, l) in enumerate(tets):
        # faces: (j,k,l), (i,k,l), (i,j,l), (i,j,k)
        t_jkl = canon_tri(j, k, l)
        t_ikl = canon_tri(i, k, l)
        t_ijl = canon_tri(i, j, l)
        t_ijk = canon_tri(i, j, k)

        for t in (t_jkl, t_ikl, t_ijl, t_ijk):
            if t not in tri_index:
                raise KeyError(f"Triangle {t} (a face of tet {(i,j,k,l)}) missing from triangles list.")

        eij = canon_edge(i, j)
        if eij not in omega_O1:
            raise KeyError(f"omega_O1 missing required edge {eij} (needed for tet {(i,j,k,l)})")

        w01 = int(omega_O1[eij])  # ±1

        A[r, tri_index[t_jkl]] += w01
        A[r, tri_index[t_ikl]] += -1
        A[r, tri_index[t_ijl]] += +1
        A[r, tri_index[t_ijk]] += -1

    return A


# ============================================================
# Membership tests (GF(2) and Z)
# ============================================================

def in_image_mod2(A: np.ndarray, b: np.ndarray) -> bool:
    """Decide if b is in the column space of A over GF(2)."""
    A = (A.copy() % 2).astype(np.uint8)
    b = (b.copy() % 2).astype(np.uint8).reshape(-1, 1)

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


def in_image_mod_p(A, b, p: int) -> bool:
    """Decide if b is in the column space of A over F_p."""
    A = (np.array(A, dtype=np.int64) % p).copy()
    b = (np.array(b, dtype=np.int64).reshape(-1, 1) % p).copy()

    m, n = A.shape
    Aug = np.concatenate([A, b], axis=1)

    row = 0
    for col in range(n):
        piv = None
        for r in range(row, m):
            if Aug[r, col] % p != 0:
                piv = r
                break
        if piv is None:
            continue

        if piv != row:
            Aug[[row, piv], :] = Aug[[piv, row], :]

        inv = pow(int(Aug[row, col]), -1, p)
        Aug[row, :] = (Aug[row, :] * inv) % p

        for r in range(m):
            if r != row and Aug[r, col] % p != 0:
                Aug[r, :] = (Aug[r, :] - Aug[r, col] * Aug[row, :]) % p

        row += 1
        if row == m:
            break

    for r in range(m):
        if np.all(Aug[r, :n] % p == 0) and (Aug[r, n] % p != 0):
            return False
    return True


def fast_reject_mod_primes(A, b, primes=(2, 3, 5, 7, 11)) -> bool:
    """False if any mod-prime test proves UNSOLVABLE over Z; True means 'maybe'."""
    for p in primes:
        if not in_image_mod_p(A, b, p):
            return False
    return True


def egcd(a: int, b: int):
    """Extended gcd: returns (g,x,y) with ax+by=g."""
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)


def is_in_image_Z_fast(A_in, b_in) -> bool:
    """
    Exact test for solvability over Z: exists x in Z^n with A x = b.

    Uses unimodular row/col operations to reduce A while updating b by the same
    row operations (without forming U,V).
    """
    A = np.array(A_in, dtype=object, copy=True)
    b = np.array(b_in, dtype=object, copy=True).reshape(-1)

    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError(f"b has length {b.shape[0]} but A has {m} rows.")

    i = j = 0
    while i < m and j < n:
        piv = None
        best = None
        for r in range(i, m):
            for c in range(j, n):
                if A[r, c] != 0:
                    v = abs(int(A[r, c]))
                    if best is None or v < best:
                        best = v
                        piv = (r, c)
        if piv is None:
            break

        r0, c0 = piv
        if r0 != i:
            A[[i, r0], :] = A[[r0, i], :]
            b[i], b[r0] = b[r0], b[i]
        if c0 != j:
            A[:, [j, c0]] = A[:, [c0, j]]

        # clear below pivot (row ops => update b)
        for r in range(i + 1, m):
            while A[r, j] != 0:
                a = int(A[i, j])
                c = int(A[r, j])
                g, x, y = egcd(a, c)
                a_div = a // g
                c_div = c // g

                Ri = A[i, :].copy()
                Rr = A[r, :].copy()
                bi = int(b[i])
                br = int(b[r])

                A[i, :] = x * Ri + y * Rr
                A[r, :] = (-c_div) * Ri + (a_div) * Rr

                b[i] = x * bi + y * br
                b[r] = (-c_div) * bi + (a_div) * br

                if A[i, j] < 0:
                    A[i, :] *= -1
                    b[i] *= -1

        # clear right in pivot row (column ops; b unchanged)
        for c in range(j + 1, n):
            while A[i, c] != 0:
                a = int(A[i, j])
                d = int(A[i, c])
                g, x, y = egcd(a, d)
                a_div = a // g
                d_div = d // g

                Cj = A[:, j].copy()
                Cc = A[:, c].copy()

                A[:, j] = x * Cj + y * Cc
                A[:, c] = (-d_div) * Cj + (a_div) * Cc

                if A[i, j] < 0:
                    A[i, :] *= -1
                    b[i] *= -1

        pivv = int(A[i, j])
        if pivv == 0:
            j += 1
            continue

        # clear other entries in pivot column (row ops => update b)
        for r in range(m):
            if r != i and A[r, j] != 0:
                k = int(A[r, j]) // pivv
                A[r, :] -= k * A[i, :]
                b[r] -= k * b[i]

        i += 1
        j += 1

    # divisibility / consistency checks
    rnk = min(m, n)
    for k in range(rnk):
        d = int(A[k, k])
        if d == 0:
            if int(b[k]) != 0:
                return False
        else:
            if int(b[k]) % d != 0:
                return False

    for k in range(rnk, m):
        if int(b[k]) != 0:
            return False

    return True


def in_image_Z_fast_pipeline(A, b, primes=(2, 3, 5, 7, 11)) -> bool:
    """Fast pipeline: mod-prime reject + exact integer membership."""
    A64 = np.array(A, dtype=np.int64, copy=False)
    b64 = np.array(b, dtype=np.int64, copy=False)
    if not fast_reject_mod_primes(A64, b64, primes=primes):
        return False
    return is_in_image_Z_fast(A, b)


# ============================================================
# Edge-driven filtration helpers
# ============================================================

def edge_removal_order(edges: List[Edge], edge_weights: Dict[Edge, float]) -> List[Edge]:
    """
    Return edges sorted by weight descending (heaviest removed first).
    Ties are broken deterministically by edge tuple.
    """
    E = [canon_edge_tuple(e) for e in edges]
    E = sorted(set(E))
    ew = {canon_edge_tuple(e): float(w) for e, w in edge_weights.items()}

    missing = [e for e in E if e not in ew]
    if missing:
        raise KeyError(f"Missing weights for {len(missing)} edges (e.g. {missing[:5]}).")

    E.sort(key=lambda e: (-ew[e], e))
    return E


def induced_triangles_from_edges(triangles: List[Tri], kept_edges: Set[Edge]) -> List[Tri]:
    """Keep exactly those triangles whose 3 edges are all present in kept_edges."""
    kept: List[Tri] = []
    for t in triangles:
        i, j, k = canon_tri_tuple(t)
        eij = canon_edge(i, j)
        eik = canon_edge(i, k)
        ejk = canon_edge(j, k)
        if (eij in kept_edges) and (eik in kept_edges) and (ejk in kept_edges):
            kept.append((i, j, k))
    return kept


def induced_tetrahedra_from_edges(tets: List[Tet], kept_edges: Set[Edge]) -> List[Tet]:
    """Keep exactly those tetrahedra whose 6 edges are all present in kept_edges."""
    kept: List[Tet] = []
    for tt in tets:
        a, b, c, d = canon_tet_tuple(tt)
        all_edges = [
            canon_edge(a, b), canon_edge(a, c), canon_edge(a, d),
            canon_edge(b, c), canon_edge(b, d),
            canon_edge(c, d),
        ]
        if all(e in kept_edges for e in all_edges):
            kept.append((a, b, c, d))
    return kept


def cutoff_weight_from_k(removal_order: List[Edge], edge_weights: Dict[Edge, float], k_removed: int) -> float:
    ew = {canon_edge_tuple(e): float(w) for e, w in edge_weights.items()}
    if len(removal_order) == 0:
        return float("inf")

    if k_removed <= 0:
        return float("inf")      # full complex

    if k_removed >= len(removal_order):
        return -float("inf")     # everything removed

    return ew[removal_order[k_removed - 1]]


# ============================================================
# SW1 persistence (edge-driven)
# ============================================================

def sw1_persistence_edge_driven(
    *,
    vertices: List[Simp],
    edges: List[Edge],
    triangles: List[Tri],
    omega_Z2: Dict[Edge, int],
    edge_weights: Dict[Edge, float],
) -> Dict[str, object]:
    """
    Edge-driven filtration:
      remove edges in descending weight order (heaviest first).
      keep triangles only if all 3 edges are still present.

    cobirth: first k such that ω is a cocycle on the induced kept 2-skeleton
             (i.e. δω=0 on every kept triangle).
    codeath: first k >= cobirth such that ω is a coboundary on kept 1-skeleton
             (i.e. ω ∈ Im(δ: C^0 -> C^1) over Z2, restricted to kept edges).
    """
    V = [canon_simplex(v) for v in vertices]
    E_all = sorted({canon_edge_tuple(e) for e in edges})
    T_all = sorted({canon_tri_tuple(t) for t in triangles})

    omega = {canon_edge_tuple(e): int(v) % 2 for e, v in omega_Z2.items()}

    rem_order = edge_removal_order(E_all, edge_weights)
    ew = {canon_edge_tuple(e): float(w) for e, w in edge_weights.items()}

    def delta_omega_on_tri(tri: Tri) -> int:
        i, j, k = tri
        eij = canon_edge(i, j)
        eik = canon_edge(i, k)
        ejk = canon_edge(j, k)
        return int((omega[ejk] - omega[eik] + omega[eij]) % 2)

    tri_bad = {t: (delta_omega_on_tri(t) % 2 == 1) for t in T_all}

    A_full = build_delta_C0_to_C1_Z2(vertices=V, edges=E_all)
    b_full = np.array([omega[e] for e in E_all], dtype=np.uint8)
    e_index = {e: idx for idx, e in enumerate(E_all)}

    m = len(rem_order)
    cob: Optional[CobirthResult] = None
    cod: Optional[CodeathResult] = None

    removed_set: Set[Edge] = set()
    removed_list: List[Edge] = []

    for k in range(0, m + 1):
        kept_edges = set(E_all) - removed_set
        kept_tris = induced_triangles_from_edges(T_all, kept_edges)

        is_cocycle = True
        for t in kept_tris:
            if tri_bad[t]:
                is_cocycle = False
                break

        if cob is None and is_cocycle:
            cob = CobirthResult(
                k_removed=k,
                cutoff_weight=cutoff_weight_from_k(rem_order, ew, k),
                removed_edges=list(removed_list),
            )

        if cob is not None and cod is None:
            kept_edge_rows = [e_index[e] for e in E_all if e in kept_edges]
            if len(kept_edge_rows) == 0:
                is_cob = True
            else:
                A_t = A_full[np.ix_(kept_edge_rows, np.arange(len(V)))]
                b_t = b_full[kept_edge_rows]
                is_cob = in_image_mod2(A_t, b_t)

            if is_cob:
                cod = CodeathResult(
                    k_removed=k,
                    cutoff_weight=cutoff_weight_from_k(rem_order, ew, k),
                    removed_edges=list(removed_list),
                )

        if cob is not None and cod is not None:
            break

        if k < m:
            e_remove = rem_order[k]
            removed_set.add(e_remove)
            removed_list.append(e_remove)

    if cob is None:
        cob = CobirthResult(k_removed=-1, cutoff_weight=float("nan"), removed_edges=[])
    if cod is None:
        cod = CodeathResult(k_removed=-1, cutoff_weight=float("nan"), removed_edges=[])

    return {"cobirth": cob, "codeath": cod, "removal_order": rem_order}


# ============================================================
# Twisted Euler persistence (edge-driven)
# ============================================================

def twisted_euler_persistence_2complex_edge_driven(
    *,
    edges: List[Edge],
    triangles: List[Tri],
    euler_class: Dict[Tri, int],
    omega_O1: Dict[Edge, int],
    edge_weights: Dict[Edge, float],
    tets: Optional[List[Tet]] = None,
) -> Dict[str, object]:
    """
    Edge-driven filtration for the twisted Euler representative e ∈ C^2(Z_ω).

    If no tets are provided (or tets is empty), this matches the old behavior:
      - cobirth is set to 0 (we assume/accept e is a cocycle on the 2-complex)
      - codeath is first k such that e becomes a twisted coboundary:
            e ∈ Im(δ_ω: C^1 -> C^2) restricted to kept edges/triangles.

    If tets are provided and nonempty:
      - cobirth is first k such that e is a twisted 2-cocycle on kept tetrahedra:
            δ_ω e = 0 on every kept tet
      - codeath is first k >= cobirth such that e becomes a twisted coboundary
            e ∈ Im(δ_ω: C^1 -> C^2) on the kept 2-skeleton.
    """
    E_all = sorted({canon_edge_tuple(e) for e in edges})
    T_all = sorted({canon_tri_tuple(t) for t in triangles})
    TT_all = sorted({canon_tet_tuple(tt) for tt in (tets or [])})

    ew = {canon_edge_tuple(e): float(w) for e, w in edge_weights.items()}
    rem_order = edge_removal_order(E_all, edge_weights)

    omega = {canon_edge_tuple(e): int(v) for e, v in omega_O1.items()}
    euler = {canon_tri_tuple(t): int(v) for t, v in euler_class.items()}

    # old δ_ω: C^1 -> C^2 membership test matrix
    A12_full = build_delta_C1_to_C2_Z_twisted(edges=E_all, triangles=T_all, omega_O1=omega).astype(np.int64)
    b2_full = np.array([euler.get(t, 0) for t in T_all], dtype=np.int64)

    e_index = {e: idx for idx, e in enumerate(E_all)}
    t_index = {t: idx for idx, t in enumerate(T_all)}

    # new δ_ω: C^2 -> C^3 cocycle test matrix (only if tets exist)
    A23_full = None
    if len(TT_all) > 0:
        A23_full = build_delta_C2_to_C3_Z_twisted(triangles=T_all, tets=TT_all, omega_O1=omega).astype(np.int64)
        tt_index = {tt: idx for idx, tt in enumerate(TT_all)}
    else:
        tt_index = {}

    m = len(rem_order)

    cob: Optional[CobirthResult] = None
    cod: Optional[CodeathResult] = None

    removed_set: Set[Edge] = set()
    removed_list: List[Edge] = []

    for k in range(0, m + 1):
        kept_edges = set(E_all) - removed_set
        kept_tris = induced_triangles_from_edges(T_all, kept_edges)

        # -------- cobirth (only meaningful if we have tetrahedra) --------
        if len(TT_all) == 0:
            if cob is None:
                cob = CobirthResult(k_removed=0, cutoff_weight=cutoff_weight_from_k(rem_order, ew, 0), removed_edges=[])
        else:
            kept_tets = induced_tetrahedra_from_edges(TT_all, kept_edges)

            # vacuous if no kept tets
            is_cocycle = True
            if len(kept_tets) > 0:
                assert A23_full is not None
                kept_tri_cols = [t_index[t] for t in kept_tris]
                kept_tet_rows = [tt_index[tt] for tt in kept_tets]

                if len(kept_tri_cols) == 0:
                    # no triangles present => e restricted is zero cochain; cocycle vacuously
                    is_cocycle = True
                else:
                    A_t = A23_full[np.ix_(kept_tet_rows, kept_tri_cols)]
                    b_t = b2_full[kept_tri_cols]
                    # exact integer check: A_t @ b_t == 0
                    is_cocycle = bool(np.all((A_t @ b_t) == 0))

            if cob is None and is_cocycle:
                cob = CobirthResult(
                    k_removed=k,
                    cutoff_weight=cutoff_weight_from_k(rem_order, ew, k),
                    removed_edges=list(removed_list),
                )

        # -------- codeath (twisted coboundary) --------
        if cob is not None and cod is None:
            if len(kept_tris) == 0:
                # no 2-simplices => any 2-cochain restricts to zero, hence coboundary
                cod = CodeathResult(
                    k_removed=k,
                    cutoff_weight=cutoff_weight_from_k(rem_order, ew, k),
                    removed_edges=list(removed_list),
                )
            else:
                kept_cols = [e_index[e] for e in E_all if e in kept_edges]
                kept_rows = [t_index[t] for t in kept_tris]

                A_t = A12_full[np.ix_(kept_rows, kept_cols)]
                b_t = b2_full[kept_rows]

                if in_image_Z_fast_pipeline(A_t, b_t):
                    cod = CodeathResult(
                        k_removed=k,
                        cutoff_weight=cutoff_weight_from_k(rem_order, ew, k),
                        removed_edges=list(removed_list),
                    )

        if cob is not None and cod is not None:
            break

        if k < m:
            e_remove = rem_order[k]
            removed_set.add(e_remove)
            removed_list.append(e_remove)

    if cob is None:
        cob = CobirthResult(k_removed=-1, cutoff_weight=float("nan"), removed_edges=[])
    if cod is None:
        cod = CodeathResult(k_removed=-1, cutoff_weight=float("nan"), removed_edges=[])

    return {"cobirth": cob, "codeath": cod, "removal_order": rem_order}


# ============================================================
# Utilities: simplices from cover
# ============================================================

def cover_vertices_from_simplices(
    edges: Iterable[Edge],
    triangles: Iterable[Tri],
    tets: Iterable[Tet],
) -> List[Simp]:
    verts: Set[int] = set()
    for a, b in edges:
        verts.add(int(a))
        verts.add(int(b))
    for a, b, c in triangles:
        verts.add(int(a))
        verts.add(int(b))
        verts.add(int(c))
    for a, b, c, d in tets:
        verts.add(int(a))
        verts.add(int(b))
        verts.add(int(c))
        verts.add(int(d))
    return [(v,) for v in sorted(verts)]


def ensure_edges_tris_tets(
    cover: Any,
    edges: Optional[Iterable[Edge]] = None,
    triangles: Optional[Iterable[Tri]] = None,
    tets: Optional[Iterable[Tet]] = None,
) -> Tuple[List[Edge], List[Tri], List[Tet], List[Simp]]:
    if edges is None:
        if hasattr(cover, "nerve_edges"):
            edges = cover.nerve_edges()
        else:
            raise ValueError("edges not provided and cover has no nerve_edges().")

    if triangles is None:
        if hasattr(cover, "nerve_triangles"):
            triangles = cover.nerve_triangles()
        else:
            triangles = []

    if tets is None:
        if hasattr(cover, "nerve_tetrahedra"):
            tets = cover.nerve_tetrahedra()
        else:
            tets = []

    edges_list = sorted({canon_edge(int(a), int(b)) for (a, b) in edges})
    tris_list = sorted({canon_tri(int(a), int(b), int(c)) for (a, b, c) in triangles})
    tets_list = sorted({canon_tet_tuple((int(a), int(b), int(c), int(d))) for (a, b, c, d) in tets})
    verts_list = cover_vertices_from_simplices(edges_list, tris_list, tets_list)
    return edges_list, tris_list, tets_list, verts_list


# Backwards-compat helper (old name)
def ensure_edges_tris(
    cover: Any,
    edges: Optional[Iterable[Edge]] = None,
    triangles: Optional[Iterable[Tri]] = None,
) -> Tuple[List[Edge], List[Tri], List[Simp]]:
    e, t, tt, v = ensure_edges_tris_tets(cover, edges=edges, triangles=triangles, tets=None)
    _ = tt  # ignored
    return e, t, v


# ============================================================
# Weight construction
# ============================================================

def build_edge_weights_from_transition_report(
    edges: Iterable[Edge],
    *,
    rms_angle_err: Optional[Dict[Edge, float]] = None,
    witness_err: Optional[Dict[Edge, float]] = None,
    prefer: str = "rms",
) -> Dict[Edge, float]:
    edges = sorted({canon_edge_tuple(e) for e in edges})

    rms = {canon_edge_tuple(e): float(v) for e, v in (rms_angle_err or {}).items()}
    wit = {canon_edge_tuple(e): float(v) for e, v in (witness_err or {}).items()}

    out: Dict[Edge, float] = {}
    for e in edges:
        if prefer == "witness":
            if e in wit:
                out[e] = wit[e]
            elif e in rms:
                out[e] = rms[e]
            else:
                raise KeyError(f"No weight available for edge {e} in witness_err or rms_angle_err.")
        else:
            if e in rms:
                out[e] = rms[e]
            elif e in wit:
                out[e] = wit[e]
            else:
                raise KeyError(f"No weight available for edge {e} in rms_angle_err or witness_err.")
    return out


# ============================================================
# Persistence runner
# ============================================================

@dataclass
class PersistenceResult:
    edges: List[Edge]
    triangles: List[Tri]
    tets: List[Tet]
    vertices: List[Simp]
    edge_weights: Dict[Edge, float]
    sw1: Dict[str, object]
    twisted_euler: Dict[str, object]


SubcomplexMode = Literal["full", "cocycle", "max_trivial"]

def _edges_for_subcomplex_from_persistence(p, mode: SubcomplexMode) -> List[Edge]:
    """
    p is a PersistenceResult from compute_bundle_persistence.
    Returns the list of kept edges (in canonical (i<j) form) for the chosen mode.
    """
    # In your persistence code, the true edge filtration order lives here:
    rem_order = list(p.sw1["removal_order"])  # heaviest first, length = #edges

    if mode == "full":
        return list(rem_order)

    sw1_cob = int(p.sw1["cobirth"].k_removed)
    sw1_cod = int(p.sw1["codeath"].k_removed)

    eu_cob  = int(p.twisted_euler["cobirth"].k_removed)
    eu_cod  = int(p.twisted_euler["codeath"].k_removed)

    if mode == "cocycle":
        k = max(sw1_cob, eu_cob)
    elif mode == "max_trivial":
        k = max(sw1_cod, eu_cod)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    k = max(0, min(k, len(rem_order)))
    return rem_order[k:]


def compute_bundle_persistence(
    *,
    cover: Any,
    classes: Any,
    edges: Optional[Iterable[Edge]] = None,
    triangles: Optional[Iterable[Tri]] = None,
    tets: Optional[Iterable[Tet]] = None,
    edge_weights: Optional[Dict[Edge, float]] = None,
    rms_angle_err: Optional[Dict[Edge, float]] = None,
    witness_err: Optional[Dict[Edge, float]] = None,
    prefer_edge_weight: str = "rms",
) -> PersistenceResult:
    """
    Edge-driven persistence (remove edges one-by-one, heaviest first).

    Expects on `classes`:
      - classes.sw1_Z2: Dict[Edge, int]
      - classes.euler_class: Dict[Tri, int]
      - classes.omega_O1_used: Dict[Edge, int]
    """
    edges_list, tris_list, tets_list, verts_list = ensure_edges_tris_tets(
        cover, edges=edges, triangles=triangles, tets=tets
    )

    if edge_weights is None:
        edge_weights = build_edge_weights_from_transition_report(
            edges_list,
            rms_angle_err=rms_angle_err,
            witness_err=witness_err,
            prefer=prefer_edge_weight,
        )
    else:
        edge_weights = {canon_edge_tuple(e): float(w) for e, w in edge_weights.items()}

    sw1_report = sw1_persistence_edge_driven(
        vertices=verts_list,
        edges=edges_list,
        triangles=tris_list,
        omega_Z2=classes.sw1_Z2,
        edge_weights=edge_weights,
    )

    te_report = twisted_euler_persistence_2complex_edge_driven(
        edges=edges_list,
        triangles=tris_list,
        tets=tets_list,
        euler_class=classes.euler_class,
        omega_O1=classes.omega_O1_used,
        edge_weights=edge_weights,
    )

    return PersistenceResult(
        edges=edges_list,
        triangles=tris_list,
        tets=tets_list,
        vertices=verts_list,
        edge_weights=edge_weights,
        sw1=sw1_report,
        twisted_euler=te_report,
    )


def summarize_edge_driven_persistence(
    p: PersistenceResult,
    *,
    top_k: int = 10,
    show: bool = True,
    mode: str = "auto",  # {"auto","text","latex","both"}
) -> Dict[str, Any]:
    """
    Persistence summary in the same style as other summaries, with:
      - rows: full, w1 cobirth, w1 codeath, Euler cobirth, Euler codeath
      - columns: k, r, |W_r^(1)|, |W_r^(2)|, |W_r^(3)|
    where W_r is the induced subcomplex on the kept edges at cutoff r.

    Notes
    -----
    - "k" means number of edges removed (heaviest-first).
    - "r" means cutoff weight at that stage (∞ for the full complex).
    - |W_r^(d)| are the counts of d-simplices in the induced subcomplex.
    """

    # ----------------------------
    # Canonicalize full complex
    # ----------------------------
    E_all = sorted({canon_edge_tuple(e) for e in p.edges})
    T_all = sorted({canon_tri_tuple(t) for t in p.triangles})
    TT_all = sorted({canon_tet_tuple(tt) for tt in p.tets})

    ew = {canon_edge_tuple(e): float(w) for e, w in p.edge_weights.items()}

    # Filtration order (heaviest removed first) — this defines the induced subcomplex at stage k
    rem_order = [canon_edge_tuple(e) for e in list(p.sw1["removal_order"])]

    # ----------------------------
    # Helpers
    # ----------------------------
    def worst(edges: List[Edge]) -> List[Tuple[Edge, float]]:
        arr = [(canon_edge_tuple(e), ew[canon_edge_tuple(e)]) for e in edges if canon_edge_tuple(e) in ew]
        arr.sort(key=lambda t: (-t[1], t[0]))
        return arr[:top_k]

    def fmt_r_3(w: float) -> str:
        """Cutoff display: 3 decimals, or ±∞."""
        if np.isposinf(w):
            return "∞"
        if np.isneginf(w):
            return "-∞"
        return f"{float(w):.3f}"

    def stage_sizes(k_removed: int) -> Tuple[int, int, int]:
        """
        Induced subcomplex sizes after removing the first k_removed edges from rem_order,
        and keeping exactly those triangles/tets from the FULL nerve supported by kept edges.
        """
        k = int(k_removed)
        if k < 0:
            raise ValueError(f"k_removed must be >= 0 (got {k}). This should not happen.")
        k = max(0, min(k, len(rem_order)))

        removed = set(rem_order[:k])
        kept_edges = set(E_all) - removed

        kept_tris = induced_triangles_from_edges(T_all, kept_edges)
        kept_tets = induced_tetrahedra_from_edges(TT_all, kept_edges) if TT_all else []

        return (len(kept_edges), len(kept_tris), len(kept_tets))

    # ----------------------------
    # Pull events
    # ----------------------------
    sw1_cob: CobirthResult = p.sw1["cobirth"]
    sw1_cod: CodeathResult = p.sw1["codeath"]
    te_cob: CobirthResult = p.twisted_euler["cobirth"]
    te_cod: CodeathResult = p.twisted_euler["codeath"]

    # "Should never happen" guard
    for name, res in [
        ("sw1_cobirth", sw1_cob),
        ("sw1_codeath", sw1_cod),
        ("euler_cobirth", te_cob),
        ("euler_codeath", te_cod),
    ]:
        if int(res.k_removed) < 0:
            raise ValueError(f"{name}.k_removed is {res.k_removed}. Expected >= 0.")

    # ----------------------------
    # Stage rows (always include full)
    # ----------------------------
    # Each row: (label, k, r_str, |W_r^(1)|, |W_r^(2)|, |W_r^(3)|, removed_edges_top)
    rows: List[Tuple[str, int, str, int, int, int, List[Tuple[Edge, float]]]] = []

    # full complex row = k=0, r=∞
    full_ke, full_kt, full_ktt = stage_sizes(0)
    rows.append(("Full nerve", 0, "∞", full_ke, full_kt, full_ktt, []))

    def add_row(label: str, res_obj):
        k = int(res_obj.k_removed)
        r_str = fmt_r_3(float(res_obj.cutoff_weight))
        ke, kt, ktt = stage_sizes(k)
        rows.append((label, k, r_str, int(ke), int(kt), int(ktt), worst(list(res_obj.removed_edges))))

    add_row("w₁ cobirth", sw1_cob)
    add_row("w₁ codeath", sw1_cod)
    add_row("Euler cobirth", te_cob)
    add_row("Euler codeath", te_cod)

    # ----------------------------
    # Output dict 
    # ----------------------------
    out: Dict[str, Any] = {
        "n_edges_total": int(len(E_all)),
        "n_triangles_total": int(len(T_all)),
        "n_tets_total": int(len(TT_all)),
        "rows": [
            {
                "stage": lab,
                "k_removed": int(k),
                "r_str": r_str,
                "W1": int(ne),
                "W2": int(nt),
                "W3": int(ntt),
                "removed_edges_top": red_top,
            }
            for (lab, k, r_str, ne, nt, ntt, red_top) in rows
        ],
        # Back-compat keys for the four “events”
        "SW1 cobirth": {
            "k_removed": int(sw1_cob.k_removed),
            "cutoff_weight": float(sw1_cob.cutoff_weight),
            "r_str": fmt_r_3(float(sw1_cob.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(sw1_cob.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(sw1_cob.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(sw1_cob.k_removed))[2],
            "removed_edges_top": worst(list(sw1_cob.removed_edges)),
        },
        "SW1 codeath": {
            "k_removed": int(sw1_cod.k_removed),
            "cutoff_weight": float(sw1_cod.cutoff_weight),
            "r_str": fmt_r_3(float(sw1_cod.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(sw1_cod.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(sw1_cod.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(sw1_cod.k_removed))[2],
            "removed_edges_top": worst(list(sw1_cod.removed_edges)),
        },
        "Euler cobirth": {
            "k_removed": int(te_cob.k_removed),
            "cutoff_weight": float(te_cob.cutoff_weight),
            "r_str": fmt_r_3(float(te_cob.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(te_cob.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(te_cob.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(te_cob.k_removed))[2],
            "removed_edges_top": worst(list(te_cob.removed_edges)),
        },
        "Euler codeath": {
            "k_removed": int(te_cod.k_removed),
            "cutoff_weight": float(te_cod.cutoff_weight),
            "r_str": fmt_r_3(float(te_cod.cutoff_weight)),
            "|W_r^(1)|": stage_sizes(int(te_cod.k_removed))[0],
            "|W_r^(2)|": stage_sizes(int(te_cod.k_removed))[1],
            "|W_r^(3)|": stage_sizes(int(te_cod.k_removed))[2],
            "removed_edges_top": worst(list(te_cod.removed_edges)),
        },
    }

    # ----------------------------
    # Renderers
    # ----------------------------
    def _display_summary_latex_persistence(rows_for_tex) -> bool:
        """
        rows_for_tex: list of tuples (stage, k, r_str, ne, nt, ntt)
        """
        try:
            from IPython.display import display, Math  # type: ignore
        except Exception:
            return False

        def _fmt_r_tex(r_str: str) -> str:
            if r_str == "∞":
                return r"\infty"
            if r_str == "-∞":
                return r"-\infty"
            return r_str

        body = []
        for stage, k, r_str, ne, nt, ntt in rows_for_tex:
            # stage is text; keep it inside \text{...}
            body.append(rf"\text{{{stage}}} & {k} & {_fmt_r_tex(r_str)} & {ne} & {nt} & {ntt}")

        latex = (
            r"\begin{array}{lrrrrr}"
            r"\textbf{Stage} & k & r & |W_r^{(1)}| & |W_r^{(2)}| & |W_r^{(3)}| \\ \hline "
            + r" \\ ".join(body)
            + r"\end{array}"
        )

        try:
            display(Math(latex))
            return True
        except Exception:
            return False

    def _print_text_table(rows_for_txt):
        title = "=== Characteristic Class Persistence ==="
        print("\n" + title)

        stage_w = max(12, max(len(r[0]) for r in rows_for_txt))
        k_w = 6
        r_w = 10
        w1_w = 12
        w2_w = 12
        w3_w = 12

        header = (
            f"{'Stage':<{stage_w}}"
            f"{'k':>{k_w}}  "
            f"{'r':>{r_w}}  "
            f"{'|W_r^(1)|':>{w1_w}}  "
            f"{'|W_r^(2)|':>{w2_w}}  "
            f"{'|W_r^(3)|':>{w3_w}}"
        )
        print(header)
        print("-" * len(header))

        for stage, k, r_str, ne, nt, ntt in rows_for_txt:
            print(
                f"{stage:<{stage_w}}"
                f"{int(k):>{k_w}}  "
                f"{r_str:>{r_w}}  "
                f"{int(ne):>{w1_w}}  "
                f"{int(nt):>{w2_w}}  "
                f"{int(ntt):>{w3_w}}"
            )
        print("")

    # ----------------------------
    # Show (text/latex/auto/both)
    # ----------------------------
    if show:
        core_rows = [(lab, k, r_str, ne, nt, ntt) for (lab, k, r_str, ne, nt, ntt, _top) in rows]

        did_latex = False
        if mode in {"latex", "auto", "both"}:
            did_latex = _display_summary_latex_persistence(core_rows)

        if mode == "both" or mode == "text" or (mode == "auto" and not did_latex):
            _print_text_table(core_rows)

    return out
