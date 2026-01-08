# z2_linear.py
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Edge = Tuple[int, int]


def solve_Z2_linear_system(A: np.ndarray, b: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Solve A x = b over Z2 via Gaussian elimination (full reduction).
    Returns one solution (free vars set to 0).
    """
    A = (np.asarray(A, dtype=np.uint8) & 1).copy()
    b = (np.asarray(b, dtype=np.uint8) & 1).copy()

    m, n = A.shape
    Ab = np.concatenate([A, b[:, None]], axis=1)  # (m, n+1)

    pivot_cols: List[int] = []
    r = 0
    for c in range(n):
        pivot = None
        for rr in range(r, m):
            if Ab[rr, c] == 1:
                pivot = rr
                break
        if pivot is None:
            continue

        if pivot != r:
            Ab[[r, pivot]] = Ab[[pivot, r]]

        # eliminate everywhere
        for rr in range(m):
            if rr != r and Ab[rr, c] == 1:
                Ab[rr, :] ^= Ab[r, :]

        pivot_cols.append(c)
        r += 1
        if r == m:
            break

    # inconsistency: [0 ... 0 | 1]
    for rr in range(m):
        if Ab[rr, :n].sum() == 0 and Ab[rr, n] == 1:
            return False, None

    x = np.zeros(n, dtype=np.uint8)
    for rr, c in enumerate(pivot_cols):
        rhs = Ab[rr, n]
        s = 0
        row = Ab[rr, :n]
        for j in range(n):
            if j != c and row[j] == 1:
                s ^= int(x[j])
        x[c] = rhs ^ s

    return True, x


def solve_Z2_edge_coboundary(
    edges: Iterable[Edge],
    omega_Z2: Dict[Edge, int],
    n_vertices: int,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Solve delta(phi) = omega over Z2 on the 1-skeleton:
        phi[j] + phi[k] = omega[(j,k)] (mod 2)
    omega_Z2 keys may include either orientation.
    """
    edges = list(edges)
    m = len(edges)

    A = np.zeros((m, n_vertices), dtype=np.uint8)
    b = np.zeros(m, dtype=np.uint8)

    for i, (j, k) in enumerate(edges):
        if j == k:
            raise ValueError("Edge with identical endpoints encountered.")

        if (j, k) in omega_Z2:
            val = omega_Z2[(j, k)]
        elif (k, j) in omega_Z2:
            val = omega_Z2[(k, j)]
        else:
            raise KeyError(f"omega_Z2 missing value for edge {(j,k)} (or reversed).")

        A[i, j] = 1
        A[i, k] = 1
        b[i] = int(val) & 1

    return solve_Z2_linear_system(A, b)


def phi_Z2_to_pm1(phi_Z2: np.ndarray) -> np.ndarray:
    """Map 0 -> +1, 1 -> -1."""
    phi_Z2 = np.asarray(phi_Z2, dtype=np.uint8)
    return np.where(phi_Z2 == 0, 1, -1).astype(int)
