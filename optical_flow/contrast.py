# optical_flow/contrast.py
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
from numpy import linalg as LA

__all__ = [
    "mean_center",
    "get_D_matrix",
    "get_contrast_norms",
    "get_dct_basis",
    "get_predominant_dirs",
    "get_lifted_predom_dirs",
]


# ----------------------------
# Basic utilities
# ----------------------------

def mean_center(X: np.ndarray) -> np.ndarray:
    """Row-wise mean centering."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    return X - X.mean(axis=1, keepdims=True)


@lru_cache(maxsize=None)
def get_D_matrix(n: int) -> np.ndarray:
    """
    Dense n^2 × n^2 matrix D such that x^T D x equals the sum of squared
    finite differences (horizontal + vertical) on an n×n grid.

    Notes
    -----
    This is the graph-Laplacian quadratic form on the n×n grid.
    It is dense and scales like O(n^4) memory, but is fine for small n (e.g., 3,5,7).
    """
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    N = n * n
    D = np.zeros((N, N), dtype=float)

    for i in range(n):
        for j in range(n):
            idx = i * n + j

            # horizontal edge (i,j) -- (i,j+1)
            if j < n - 1:
                r = i * n + (j + 1)
                D[idx, idx] += 1
                D[r, r] += 1
                D[idx, r] -= 1
                D[r, idx] -= 1

            # vertical edge (i,j) -- (i+1,j)
            if i < n - 1:
                d = (i + 1) * n + j
                D[idx, idx] += 1
                D[d, d] += 1
                D[idx, d] -= 1
                D[d, idx] -= 1

    return D


def get_contrast_norms(data: np.ndarray, *, patch_type: str = "opt_flow") -> np.ndarray:
    """
    Vectorized contrast norms.

    Parameters
    ----------
    data : array of shape (N, n^2) for intensity patches,
           or (N, 2*n^2) for optical flow (u then v).
    patch_type : {'img','opt_flow','flow'}
        'flow' is accepted as an alias for 'opt_flow'.

    Returns
    -------
    norms : (N,) array, sqrt( x^T D x ) (and sum across u,v for flow)
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("data must be a 2D array of shape (N, d)")

    if patch_type == "flow":
        patch_type = "opt_flow"

    if patch_type not in {"img", "opt_flow"}:
        raise ValueError("patch_type must be one of {'img','opt_flow','flow'}")

    if patch_type == "opt_flow":
        if X.shape[1] % 2 != 0:
            raise ValueError("opt_flow patches must have even length 2*n^2")
        n2 = X.shape[1] // 2
    else:
        n2 = X.shape[1]

    n = int(np.sqrt(n2))
    if n * n != n2:
        raise ValueError(f"Expected n^2 columns; got {n2} which isn't a perfect square.")
    d = n * n

    D = get_D_matrix(n)

    if patch_type == "opt_flow":
        u = X[:, :d]
        v = X[:, d:2 * d]
        quad = (
            np.einsum("ni,ij,nj->n", u, D, u) +
            np.einsum("ni,ij,nj->n", v, D, v)
        )
    else:
        quad = np.einsum("ni,ij,nj->n", X, D, X)

    quad = np.maximum(quad, 0.0)  # numerical safety
    return np.sqrt(quad)


# ----------------------------
# DCT basis (general N)
# ----------------------------

def get_dct_basis(
    N: int,
    *,
    normalize: bool = True,
    opt_flow: bool = False,
    top_two: bool = False,
) -> np.ndarray:
    """
    Generate a 2D DCT-II basis for N×N patches.

    Returns
    -------
    basis : (M, N^2) if opt_flow=False
            (2*M, 2*N^2) if opt_flow=True (stack horizontal then vertical)
    where M = N^2 (or N^2-1 if normalize=True, since DC is dropped).

    Notes
    -----
    - If normalize=True, drop the DC component, mean-center each basis vector,
      then contrast-normalize using get_contrast_norms (via flow padding trick).
    """
    N = int(N)
    if N <= 0:
        raise ValueError("N must be positive")

    basis = []
    for u in range(N):
        for v in range(N):
            patch = np.zeros((N, N), dtype=float)
            for x in range(N):
                for y in range(N):
                    patch[x, y] = (
                        np.cos((2 * x + 1) * u * np.pi / (2 * N)) *
                        np.cos((2 * y + 1) * v * np.pi / (2 * N))
                    )

            # Light scaling (not crucial since we normalize later)
            patch *= (2.0 / N) * (2.0 / N)
            if u == 0:
                patch *= 1.0 / np.sqrt(2.0)
            if v == 0:
                patch *= 1.0 / np.sqrt(2.0)

            basis.append(patch.reshape(-1))

    basis = np.asarray(basis, dtype=float)  # (N^2, N^2)

    if normalize:
        # Drop DC component (u=v=0 term)
        basis = basis[1:]
        basis = mean_center(basis)

        # contrast-normalize: treat each as u component with zero v
        tmp = np.hstack([basis, np.zeros_like(basis)])
        norms = get_contrast_norms(tmp, patch_type="opt_flow")
        norms_safe = np.where(norms > 0, norms, 1.0)
        basis = basis / norms_safe[:, None]

    if opt_flow:
        h = np.hstack([basis, np.zeros_like(basis)])
        v = np.hstack([np.zeros_like(basis), basis])
        basis = np.vstack([h, v])

    if top_two:
        if opt_flow:
            # Choose two low-frequency modes from each component block
            m = basis.shape[0] // 2
            take0 = 0
            take1 = min(N - 1, m - 1) if m > 1 else 0
            return basis[[take0, take1, take0 + m, take1 + m]]
        else:
            take0 = 0
            take1 = min(N - 1, basis.shape[0] - 1) if basis.shape[0] > 1 else 0
            return basis[[take0, take1]]

    return basis


# ----------------------------
# Predominant direction utilities
# ----------------------------

def get_predominant_dirs(patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predominant direction in RP^1 via PCA of (u_i, v_i) vectors across pixels.

    Parameters
    ----------
    patches : (N, 2*n^2)

    Returns
    -------
    predom_dirs : (N,) angles in [0, pi)
    ratios      : (N,) directionality score in [0,1] ( (λ1-λ2)/λ1 )
    """
    X = np.asarray(patches, dtype=float)
    if X.ndim != 2 or X.shape[1] % 2 != 0:
        raise ValueError("patches must have shape (N, 2*n^2)")

    N, L = X.shape
    d = L // 2

    u = X[:, :d]
    v = X[:, d:]

    uu = np.sum(u * u, axis=1)
    vv = np.sum(v * v, axis=1)
    uv = np.sum(u * v, axis=1)

    predom = np.zeros(N, dtype=float)
    ratios = np.zeros(N, dtype=float)

    for i in range(N):
        C = np.array([[uu[i], uv[i]],
                      [uv[i], vv[i]]], dtype=float)

        eigvals, eigvecs = LA.eigh(C)
        idx = int(np.argmax(eigvals))
        vec = eigvecs[:, idx]
        lam1 = float(eigvals[idx])
        lam2 = float(eigvals[1 - idx])

        predom[i] = float(np.arctan2(vec[1], vec[0]) % np.pi)
        if lam1 > 0:
            ratios[i] = (lam1 - lam2) / lam1

    return predom, ratios


def get_lifted_predom_dirs(flow_patches: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    For each patch, find the pixel with largest flow magnitude and return its unit vector.

    Parameters
    ----------
    flow_patches : (N, 2*n^2)

    Returns
    -------
    unit : (N,2) unit vectors (zeros where magnitude is tiny)
    """
    X = np.asarray(flow_patches, dtype=float)
    if X.ndim != 2:
        raise ValueError("flow_patches must be 2D (N, 2*n^2)")
    N, L = X.shape
    if L % 2 != 0:
        raise ValueError("Each patch must have even length 2*n^2.")

    n2 = L // 2
    n = int(np.sqrt(n2))
    if n * n != n2:
        raise ValueError("Expected length 2*n^2 with n^2 a perfect square.")

    # NOTE: keep Fortran-order convention consistent with your sampling pipeline.
    V = X.reshape(N, n, n, 2, order="F")
    mags = np.linalg.norm(V, axis=3)  # (N,n,n)

    k = np.argmax(mags.reshape(N, -1), axis=1)
    ii, jj = np.unravel_index(k, (n, n))
    vecs = V[np.arange(N), ii, jj, :]  # (N,2)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = np.zeros_like(vecs)
    mask = norms[:, 0] > float(eps)
    unit[mask] = vecs[mask] / norms[mask]
    return unit
