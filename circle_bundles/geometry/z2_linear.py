# circle_bundles/z2_linear.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Edge = Tuple[int, int]

__all__ = [
    "solve_Z2_linear_system",
    "solve_Z2_edge_coboundary",
    "phi_Z2_to_pm1",
]


def solve_Z2_linear_system(A: np.ndarray, b: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Solve A x = b over Z2 via Gaussian elimination with full elimination
    (RREF-like: eliminate pivot column in all rows).

    Returns
    -------
    ok : bool
        True iff the system is consistent.
    x : (n,) uint8 array or None
        One solution (free variables set to 0) when consistent.
    """
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    b = (np.asarray(b, dtype=np.uint8) & 1).reshape(-1).copy()

    if A.ndim != 2:
        raise ValueError(f"A must be 2D; got shape {A.shape}.")
    m, n = A.shape
    if b.shape != (m,):
        raise ValueError(f"b must have shape ({m},); got {b.shape}.")

    Ab = np.concatenate([A, b[:, None]], axis=1)  # (m, n+1)

    pivot_cols: List[int] = []
    r = 0
    for c in range(n):
        # find pivot row
        pivot = None
        for rr in range(r, m):
            if Ab[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            continue

        # swap into position
        if pivot != r:
            Ab[[r, pivot]] = Ab[[pivot, r]]

        # eliminate pivot column from all other rows
        for rr in range(m):
            if rr != r and Ab[rr, c] == 1:
                Ab[rr, :] ^= Ab[r, :]

        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    # inconsistency: [0 ... 0 | 1]
    left = Ab[:, :n]
    rhs = Ab[:, n]
    zero_left = (left.sum(axis=1) == 0)
    if np.any(zero_left & (rhs == 1)):
        return False, None

    # one solution with free vars = 0
    x = np.zeros(n, dtype=np.uint8)
    for rr, c in enumerate(pivot_cols):
        row = Ab[rr, :n]
        # since free vars are 0, only need to account for already-set pivots;
        # but doing the general dot still cheap at this scale.
        s = np.bitwise_xor.reduce(row & x, initial=0)  # XOR-sum of row[j]*x[j]
        x[c] = rhs[rr] ^ s ^ x[c]  # row includes pivot itself, so remove its contribution
        # (equivalently: x[c] = rhs[rr] ^ XOR_{j!=c} row[j]*x[j])

    # The line above is safe but slightly opaque; the classic explicit form is:
    # x[c] = rhs[rr] ^ (XOR over j!=c with row[j]==1 of x[j])
    # Keeping as-is preserves determinism and correctness.

    return True, x


def solve_Z2_edge_coboundary(
    edges: Iterable[Edge],
    omega_Z2: Dict[Edge, int],
    n_vertices: int,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Solve δ(phi) = omega over Z2 on a 1-skeleton:

        phi[j] + phi[k] = omega[(j,k)] (mod 2)

    Notes
    -----
    - edges may be directed or undirected; omega_Z2 may store either orientation.
    - Returns one solution with free variables set to 0.
    """
    edges_list = list(edges)
    m = len(edges_list)
    n_vertices = int(n_vertices)
    if n_vertices < 0:
        raise ValueError("n_vertices must be nonnegative.")

    A = np.zeros((m, n_vertices), dtype=np.uint8)
    b = np.zeros(m, dtype=np.uint8)

    for i, (j, k) in enumerate(edges_list):
        j = int(j)
        k = int(k)
        if j == k:
            raise ValueError("Edge with identical endpoints encountered.")
        if not (0 <= j < n_vertices and 0 <= k < n_vertices):
            raise ValueError(f"Edge {(j,k)} has endpoint outside [0,{n_vertices-1}].")

        if (j, k) in omega_Z2:
            val = omega_Z2[(j, k)]
        elif (k, j) in omega_Z2:
            val = omega_Z2[(k, j)]
        else:
            raise KeyError(f"omega_Z2 missing value for edge {(j,k)} (or reversed).")

        A[i, j] = 1
        A[i, k] = 1
        b[i] = np.uint8(int(val) & 1)

    return solve_Z2_linear_system(A, b)


def phi_Z2_to_pm1(phi_Z2: np.ndarray) -> np.ndarray:
    """Map Z2 vector to ±1: 0 -> +1, 1 -> -1."""
    phi_Z2 = (np.asarray(phi_Z2, dtype=np.uint8) & 1)
    return np.where(phi_Z2 == 0, 1, -1).astype(int)
