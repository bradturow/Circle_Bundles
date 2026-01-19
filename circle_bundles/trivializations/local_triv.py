# circle_bundles/local_triv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import warnings

from .status_utils import _status, _status_clear


__all__ = [
    "LocalTrivResult",
    "compute_circular_coords_pca2",
    "compute_circular_coords_dreimac",
    "compute_local_triv",
]


@dataclass
class LocalTrivResult:
    f: np.ndarray
    valid: np.ndarray
    n_retries: np.ndarray
    n_landmarks: np.ndarray
    errors: Dict[int, str]


def _pca2_project(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return 2D PCA coordinates and PCA basis.

    Returns
    -------
    Y : (m,2) projected coords
    V : (D,2) principal directions
    mu : (D,) mean
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (m,D). Got shape {X.shape}.")
    m = int(X.shape[0])
    if m == 0:
        raise ValueError("Empty patch: cannot run PCA.")

    mu = X.mean(axis=0)
    Xc = X - mu

    # SVD is stable and fast for PCA:
    # Xc = U S Vt, columns of V are principal directions in R^D
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V = Vt.T[:, :2]  # (D,2)
    Y = Xc @ V       # (m,2)
    return Y, V, mu


def _mds2_from_dist(D: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Classical MDS to 2D from a distance matrix.

    Parameters
    ----------
    D : (m,m) distances (assumed symmetric, 0 diagonal)

    Returns
    -------
    Y : (m,2) Euclidean embedding
    """
    D = np.asarray(D, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be square (m,m). Got {D.shape}.")
    m = int(D.shape[0])
    if m == 0:
        raise ValueError("Empty distance matrix: cannot run MDS.")

    # Double-centering: B = -1/2 * J * D^2 * J
    J = np.eye(m) - np.ones((m, m), dtype=float) / float(m)
    D2 = D * D
    B = -0.5 * (J @ D2 @ J)

    # Eigendecomposition (symmetric)
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    w = np.maximum(w, 0.0)
    if w.size == 0 or w[0] < eps:
        # totally degenerate
        return np.zeros((m, 2), dtype=float)

    Y = V[:, :2] * np.sqrt(w[:2])[None, :]
    return Y


def compute_circular_coords_pca2(
    X: Optional[np.ndarray] = None,
    *,
    dist_mat: Optional[np.ndarray] = None,
    anchor: str = "farthest",  # "farthest" or "first"
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute simple circular coordinates via 2D PCA (or via 2D classical MDS if dist_mat given).

    This is a lightweight fallback when Dreimac is unavailable/unwanted.

    Parameters
    ----------
    X : (m,D) array, required if dist_mat is None
    dist_mat : (m,m) optional precomputed distances (if provided, uses MDS->2D->atan2)
    anchor : orientation stabilization:
        - "farthest": pick the point farthest from the mean in 2D and rotate so it has angle 0
        - "first": rotate so the first point has angle 0
    eps : small guard

    Returns
    -------
    angles : (m,) in [0, 2pi)
    """
    if dist_mat is None:
        if X is None:
            raise ValueError("Provide X or dist_mat.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (m,D). Got shape {X.shape}.")
        Y, _, _ = _pca2_project(X)
    else:
        D = np.asarray(dist_mat, dtype=float)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise ValueError(f"dist_mat must be square (m,m). Got {D.shape}.")
        Y = _mds2_from_dist(D, eps=eps)

    u = Y[:, 0]
    v = Y[:, 1]

    # If the projection collapsed, return zeros
    if np.std(u) < eps and np.std(v) < eps:
        return np.zeros(len(u), dtype=float)

    ang = np.arctan2(v, u)  # (-pi,pi]
    ang = np.mod(ang, 2.0 * np.pi)

    # ---- orientation stabilization (optional but helpful) ----
    anchor = str(anchor)
    if anchor == "first" and len(ang) > 0:
        ang0 = float(ang[0])
        ang = np.mod(ang - ang0, 2.0 * np.pi)

    elif anchor == "farthest" and len(ang) > 0:
        r2 = u * u + v * v
        k = int(np.argmax(r2))
        ang0 = float(ang[k])
        ang = np.mod(ang - ang0, 2.0 * np.pi)

    elif anchor not in {"first", "farthest"}:
        raise ValueError(f"anchor must be 'first' or 'farthest'. Got {anchor!r}")

    return ang


def compute_circular_coords_dreimac(
    X: np.ndarray,
    *,
    n_landmarks_init: int,
    prime: int = 41,
    update_frac: float = 0.25,
    standard_range: bool = False,
    CircularCoords_cls=None,
    dist_mat: Optional[np.ndarray] = None,  # optional precomputed distance matrix
) -> Tuple[np.ndarray, int, int]:
    """
    Compute circular coordinates using Dreimac.

    If dist_mat is provided, it must be (n_points, n_points) and we pass
    distance_matrix=True to Dreimac.
    """
    if CircularCoords_cls is None:
        raise ValueError("CircularCoords_cls must be provided (e.g., dreimac.CircularCoords).")

    X = np.asarray(X)
    n_points = int(X.shape[0])
    if n_points == 0:
        raise ValueError("Empty patch: cannot compute circular coordinates.")

    use_dist = dist_mat is not None
    if use_dist:
        D = np.asarray(dist_mat, dtype=float)
        if D.shape != (n_points, n_points):
            raise ValueError(f"dist_mat has shape {D.shape}, expected ({n_points},{n_points}).")
        X_or_D = D
    else:
        X_or_D = X

    n_landmarks = min(int(n_landmarks_init), n_points)
    n_retries = 0

    while True:
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                cc = CircularCoords_cls(
                    X_or_D,
                    n_landmarks,
                    prime=prime,
                    distance_matrix=use_dist,  # IMPORTANT
                )
                angles = cc.get_coordinates(standard_range=standard_range)

                # Dreimac sometimes emits coverage warnings; treat as retry.
                for w in wlist:
                    msg = str(w.message).lower()
                    if "not covered by a landmark" in msg:
                        raise RuntimeError("Dreimac: not covered by a landmark")

            angles = np.asarray(angles, dtype=float).reshape(-1)
            if angles.shape != (n_points,):
                raise ValueError(f"Dreimac returned shape {angles.shape}, expected ({n_points},).")

            # Robust wrap to circle
            angles = np.mod(angles, 2.0 * np.pi)
            return angles, n_retries, n_landmarks

        except Exception as e:
            n_retries += 1
            if n_landmarks >= n_points:
                raise ValueError(
                    f"Circular coordinates failed even with n_landmarks=n_points={n_points}. "
                    f"Last error: {type(e).__name__}: {e}"
                ) from e

            n_landmarks = min(int(np.ceil((1.0 + float(update_frac)) * n_landmarks)), n_points)


def compute_local_triv(
    data: np.ndarray,
    U: np.ndarray,
    *,
    cc_alg: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    total_metric: Optional[object] = None,

    # Dreimac defaults
    landmarks_per_patch: int = 200,
    prime: int = 41,
    update_frac: float = 0.25,
    standard_range: bool = False,
    CircularCoords_cls=None,

    # PCA fallback knobs
    pca_anchor: str = "farthest",   # NOTE: matches compute_circular_coords_pca2
    notify_pca_fallback: bool = True,

    # Robustness knobs
    min_patch_size: int = 10,
    verbose: bool = True,
    fail_fast: bool = True,
) -> LocalTrivResult:
    """
    Compute local circle coordinates f[j, s] on each cover set U[j].

    Conventions (UNCHANGED):
    - U has shape (n_sets, n_samples) bool
    - f has shape (n_sets, n_samples) radians
    - f[j, s] meaningful only when U[j, s] is True

    Behavior:
    - If cc_alg is provided: use it on each patch.
    - Else if CircularCoords_cls is provided: use Dreimac (optionally with total_metric dist mats).
    - Else: fall back to PCA-based circular coordinates (metric-PCA via MDS if total_metric is provided).
    """
    data = np.asarray(data)
    U = np.asarray(U, dtype=bool)

    if U.ndim != 2:
        raise ValueError(f"U must be 2D (n_sets, n_samples). Got shape {U.shape}.")

    n_sets, n_samples = U.shape
    if data.shape[0] != n_samples:
        raise ValueError(f"data has n={data.shape[0]} samples but U has n_samples={n_samples}.")

    # Decide default method
    use_pca_default = (cc_alg is None) and (CircularCoords_cls is None)
    if use_pca_default and verbose and notify_pca_fallback:
        _status("CircularCoords_cls not provided; using PCA-based circular coordinates.")

    f = np.zeros((n_sets, n_samples), dtype=float)
    valid = np.zeros(n_sets, dtype=bool)
    n_retries = np.zeros(n_sets, dtype=int)
    n_landmarks = np.zeros(n_sets, dtype=int)
    errors: Dict[int, str] = {}

    for j in range(n_sets):
        if verbose:
            _status(f"Coordinatizing set {j+1}/{n_sets}...")

        mask = U[j]
        m = int(mask.sum())

        if m < int(min_patch_size):
            msg = f"Patch too small for set j={j}: |U[j]|={m} < min_patch_size={min_patch_size}."
            if fail_fast:
                if verbose:
                    _status_clear()
                raise ValueError(msg)
            errors[j] = msg
            continue

        Xj = data[mask]

        try:
            # 1) Explicit cc_alg always wins
            if cc_alg is not None:
                ang = np.asarray(cc_alg(Xj), dtype=float).reshape(-1)
                if ang.shape != (m,):
                    raise ValueError(f"cc_alg returned shape {ang.shape}, expected ({m},).")
                f[j, mask] = np.mod(ang, 2.0 * np.pi)
                valid[j] = True
                continue

            # 2) PCA default fallback
            if use_pca_default:
                Dj = None
                if total_metric is not None:
                    if not hasattr(total_metric, "pairwise"):
                        raise TypeError("total_metric must have a .pairwise(X, Y=None) method.")
                    Dj = np.asarray(total_metric.pairwise(Xj), dtype=float)
                    if Dj.shape != (m, m):
                        raise ValueError(f"total_metric.pairwise returned shape {Dj.shape}, expected ({m},{m}).")

                ang = compute_circular_coords_pca2(Xj, dist_mat=Dj, anchor=str(pca_anchor))
                f[j, mask] = ang
                valid[j] = True
                continue

            # 3) Otherwise Dreimac path (requires CircularCoords_cls)
            Dj = None
            if total_metric is not None:
                if not hasattr(total_metric, "pairwise"):
                    raise TypeError("total_metric must have a .pairwise(X, Y=None) method.")
                Dj = np.asarray(total_metric.pairwise(Xj), dtype=float)
                if Dj.shape != (m, m):
                    raise ValueError(f"total_metric.pairwise returned shape {Dj.shape}, expected ({m},{m}).")

            ang, retries, n_lmks = compute_circular_coords_dreimac(
                Xj,
                dist_mat=Dj,
                n_landmarks_init=landmarks_per_patch,
                prime=prime,
                update_frac=update_frac,
                standard_range=standard_range,
                CircularCoords_cls=CircularCoords_cls,
            )
            f[j, mask] = ang
            valid[j] = True
            n_retries[j] = int(retries)
            n_landmarks[j] = int(n_lmks)

        except Exception as e:
            msg = f"Failed on set j={j} (|U[j]|={m}): {type(e).__name__}: {e}"
            if fail_fast:
                if verbose:
                    _status_clear()
                raise ValueError(msg) from e
            errors[j] = msg
            valid[j] = False

    if verbose:
        _status_clear()

    return LocalTrivResult(
        f=f,
        valid=valid,
        n_retries=n_retries,
        n_landmarks=n_landmarks,
        errors=errors,
    )
