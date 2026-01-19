# geometry.py
from __future__ import annotations
import numpy as np


def get_bary_coords(points: np.ndarray, tri_vertices: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinates of points w.r.t. a triangle in R^D.

    Parameters
    ----------
    points : (N, D) or (D,) array
    tri_vertices : (3, D) array, vertices [v0,v1,v2]

    Returns
    -------
    bary : (N, 3) array
        Rows are (u,v,w). For each row, u+v+w ≈ 1 (least-squares affine fit).
    """
    P = np.asarray(points, dtype=float)
    V = np.asarray(tri_vertices, dtype=float)

    if V.ndim != 2 or V.shape[0] != 3:
        raise ValueError(f"tri_vertices must have shape (3,D); got {V.shape}")

    if P.ndim == 1:
        P = P.reshape(1, -1)
    if P.ndim != 2:
        raise ValueError(f"points must have shape (N,D) or (D,); got {P.shape}")

    if P.shape[1] != V.shape[1]:
        raise ValueError(f"Dimension mismatch: points have D={P.shape[1]} but tri has D={V.shape[1]}")

    # Solve affine system:
    # [V^T; 1 1 1] * bary = [p^T; 1]
    A = np.vstack([V.T, np.ones((1, 3), dtype=float)])           # (D+1, 3)
    B = np.hstack([P, np.ones((P.shape[0], 1), dtype=float)])    # (N, D+1)

    # Least squares for all points at once: A @ X ≈ B^T
    X, residuals, rank, svals = np.linalg.lstsq(A, B.T, rcond=None)  # X shape (3, N)

    if rank < 3:
        raise ValueError(
            "Degenerate triangle for barycentric coordinates: affine system rank < 3. "
            "Check that the 3 vertices are affinely independent."
        )

    return X.T  # (N,3)


def points_in_triangle_mask(bary: np.ndarray, tol: float = 1e-8, sum_tol: float | None = None) -> np.ndarray:
    """
    Given barycentric coords (N,3), return mask of points inside triangle.

    A point is considered inside if:
      - each coordinate >= -tol
      - sum(u,v,w) is close to 1 (within sum_tol)

    Notes
    -----
    For least-squares barycentric coordinates, sum-to-1 can drift slightly.
    """
    B = np.asarray(bary, dtype=float)
    if B.ndim == 1:
        B = B.reshape(1, -1)
    if B.ndim != 2 or B.shape[1] != 3:
        raise ValueError(f"bary must have shape (N,3) or (3,); got {B.shape}")

    if sum_tol is None:
        sum_tol = 10 * tol

    return (B >= -tol).all(axis=1) & (np.abs(B.sum(axis=1) - 1.0) <= sum_tol)
