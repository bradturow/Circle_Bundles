# circle_bundles/trivializations/local_triv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any

import numpy as np
import warnings

from ..utils.status_utils import _status, _status_clear


__all__ = [
    "LocalTrivResult",
    "DreimacCCConfig",
    "compute_circular_coords_pca2",
    "compute_circular_coords_dreimac",
    "compute_local_triv",
]


@dataclass
class LocalTrivResult:
    """
    Result container for local trivialization (circle-coordinate) computations.

    This object is returned by :func:`compute_local_triv` and packages the
    per-cover-set circular coordinates together with diagnostic metadata.

    Conventions
    -----------
    - U has shape (n_sets, n_samples)
    - f has shape (n_sets, n_samples)
    - f[j, s] is meaningful only when U[j, s] is True

    Attributes
    ----------
    f : ndarray of shape (n_sets, n_samples)
        Local circular coordinates in radians, wrapped to [0, 2π).
        Values are only meaningful on samples belonging to the corresponding
        cover set.

    valid : ndarray of shape (n_sets,)
        Boolean mask indicating which cover sets were successfully
        coordinatized.

    n_retries : ndarray of shape (n_sets,)
        Number of retries used for each cover set (relevant for iterative
        methods such as Dreimac).

    n_landmarks : ndarray of shape (n_sets,)
        Number of landmarks ultimately used for each cover set
        (method-dependent; meaningful for Dreimac-based methods).

    errors : dict[int, str]
        Mapping from cover-set index to error message for any set that
        failed to produce valid coordinates.

    Notes
    -----
    This class is intended as a lightweight, inspection-friendly summary
    of the local trivialization step. Most users will encounter it through
    higher-level bundle construction workflows.
    """
    f: np.ndarray
    valid: np.ndarray
    n_retries: np.ndarray
    n_landmarks: np.ndarray
    errors: Dict[int, str]


# ----------------------------
# CC method config(s)
# ----------------------------

@dataclass(frozen=True)
class DreimacCCConfig:
    """
    Configuration object for Dreimac-based circular coordinates.

    This dataclass specifies how Dreimac's circular coordinates algorithm
    should be applied within local trivialization routines such as
    :func:`compute_local_triv`.

    Attributes
    ----------
    CircularCoords_cls : Any
        The Dreimac circular coordinates class (e.g. ``dreimac.CircularCoords``).

    landmarks_per_patch : int, default=200
        Initial number of landmarks to use per patch. The algorithm may
        increase this value automatically if coverage is insufficient.

    prime : int, default=41
        Prime number used internally by Dreimac for coefficient computations.

    update_frac : float, default=0.25
        Fractional increase applied to the number of landmarks when a retry
        is required.

    standard_range : bool, default=False
        Whether to return angles in Dreimac's standard range instead of
        wrapping to [0, 2π).

    Notes
    -----
    - This configuration is passed as the ``cc`` argument to
      :func:`compute_local_triv`.
    - If a ``total_metric`` is supplied to :func:`compute_local_triv`,
      distance matrices are passed to Dreimac with ``distance_matrix=True``.
    - The dataclass is frozen to emphasize its role as an immutable
      configuration object.
    """
    CircularCoords_cls: Any
    landmarks_per_patch: int = 200
    prime: int = 41
    update_frac: float = 0.25
    standard_range: bool = False  


# ----------------------------
# PCA2 / MDS helpers
# ----------------------------

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

    J = np.eye(m) - np.ones((m, m), dtype=float) / float(m)
    D2 = D * D
    B = -0.5 * (J @ D2 @ J)

    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]

    w = np.maximum(w, 0.0)
    if w.size == 0 or w[0] < eps:
        return np.zeros((m, 2), dtype=float)

    Y = V[:, :2] * np.sqrt(w[:2])[None, :]
    return Y


def compute_circular_coords_pca2(
    X: Optional[np.ndarray] = None,
    *,
    dist_mat: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute simple circular coordinates via 2D PCA (or via 2D classical MDS if dist_mat given).

    Orientation stabilization is fixed to the "farthest point" convention.

    Parameters
    ----------
    X : (m,D) array, required if dist_mat is None
    dist_mat : (m,m) optional precomputed distances (if provided, uses MDS->2D->atan2)
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

    if np.std(u) < eps and np.std(v) < eps:
        return np.zeros(len(u), dtype=float)

    ang = np.arctan2(v, u)
    ang = np.mod(ang, 2.0 * np.pi)

    # ---- orientation stabilization: ALWAYS "farthest" ----
    if len(ang) > 0:
        r2 = u * u + v * v
        k = int(np.argmax(r2))
        ang0 = float(ang[k])
        ang = np.mod(ang - ang0, 2.0 * np.pi)

    return ang


def compute_circular_coords_dreimac(
    X: np.ndarray,
    *,
    n_landmarks_init: int,
    prime: int = 41,
    update_frac: float = 0.25,
    standard_range: bool = False,
    CircularCoords_cls=None,
    dist_mat: Optional[np.ndarray] = None,
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
                    distance_matrix=use_dist,
                )
                angles = cc.get_coordinates(standard_range=standard_range)

                for w in wlist:
                    msg = str(w.message).lower()
                    if "not covered by a landmark" in msg:
                        raise RuntimeError("Dreimac: not covered by a landmark")

            angles = np.asarray(angles, dtype=float).reshape(-1)
            if angles.shape != (n_points,):
                raise ValueError(f"Dreimac returned shape {angles.shape}, expected ({n_points},).")

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


def _pairwise_dist_or_none(total_metric: Optional[object], Xj: np.ndarray) -> Optional[np.ndarray]:
    if total_metric is None:
        return None
    if not hasattr(total_metric, "pairwise"):
        raise TypeError("total_metric must have a .pairwise(X, Y=None) method.")
    Dj = np.asarray(total_metric.pairwise(Xj), dtype=float)
    m = int(Xj.shape[0])
    if Dj.shape != (m, m):
        raise ValueError(f"total_metric.pairwise returned shape {Dj.shape}, expected ({m},{m}).")
    return Dj


def compute_local_triv(
    data: np.ndarray,
    U: np.ndarray,
    *,
    cc: object = "pca2",
    total_metric: Optional[object] = None,
    min_patch_size: int = 10,
    verbose: bool = True,
    fail_fast: bool = True,
) -> LocalTrivResult:
    """
    Compute local circle coordinates f[j, s] on each cover set U[j].

    Supports two data modes:
    - Point cloud: data shape (n_samples, D)
    - Full distance matrix: data shape (n_samples, n_samples)

    Conventions (UNCHANGED):
    - U has shape (n_sets, n_samples) bool
    - f has shape (n_sets, n_samples) radians
    - f[j, s] meaningful only when U[j, s] is True

    Circular coordinates method (cc):
    - "pca2" (default): compute_circular_coords_pca2
      * uses PCA2 on points, or MDS2 if a dist_mat is provided
    - DreimacCCConfig(...): uses Dreimac on points or distance matrices
    - callable: advanced hook; we call cc(Xj, dist_mat=Dj) if possible, else cc(Xj)

    Notes
    -----
    - For a callable cc, returning angles in radians is expected; we wrap to [0,2pi).
    - If `data` is a distance matrix, we pass per-patch submatrices as `dist_mat`.
      In that case, `Xj` is a dummy placeholder (unless the callable ignores dist_mat).
    """
    data = np.asarray(data)
    U = np.asarray(U, dtype=bool)

    if U.ndim != 2:
        raise ValueError(f"U must be 2D (n_sets, n_samples). Got shape {U.shape}.")

    n_sets, n_samples = U.shape

    # Determine whether data is a full distance matrix
    data_is_dist = (data.ndim == 2 and data.shape[0] == data.shape[1])

    if data_is_dist:
        if data.shape[0] != n_samples:
            raise ValueError(
                f"Distance matrix has n={data.shape[0]} but U has n_samples={n_samples}."
            )
    else:
        # point cloud mode
        if data.ndim != 2:
            raise ValueError(
                f"data must be either a point cloud (n_samples, D) or a distance matrix (n_samples, n_samples). "
                f"Got shape {data.shape}."
            )
        if data.shape[0] != n_samples:
            raise ValueError(f"data has n={data.shape[0]} samples but U has n_samples={n_samples}.")

    cc_is_pca2 = (isinstance(cc, str) and str(cc).lower() == "pca2")
    cc_is_dreimac = isinstance(cc, DreimacCCConfig)
    cc_is_callable = callable(cc)

    if not (cc_is_pca2 or cc_is_dreimac or cc_is_callable):
        raise ValueError(
            "cc must be 'pca2', a DreimacCCConfig(...), or a callable. "
            f"Got {type(cc).__name__}: {cc!r}"
        )

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

        try:
            # Build per-patch data
            if data_is_dist:
                idx = np.where(mask)[0]
                Dj = np.asarray(data[np.ix_(idx, idx)], dtype=float)
                if Dj.shape != (m, m):
                    raise ValueError(f"Patch dist submatrix has shape {Dj.shape}, expected ({m},{m}).")
                # Dummy placeholder for callables that insist on Xj
                Xj = np.zeros((m, 1), dtype=float)
            else:
                Xj = np.asarray(data[mask], dtype=float)
                Dj = _pairwise_dist_or_none(total_metric, Xj)

            # 1) Custom callable wins
            if cc_is_callable:
                try:
                    ang = cc(Xj, dist_mat=Dj)  # type: ignore[misc]
                except TypeError:
                    ang = cc(Xj)  # type: ignore[misc]

                ang = np.asarray(ang, dtype=float).reshape(-1)
                if ang.shape != (m,):
                    raise ValueError(f"cc callable returned shape {ang.shape}, expected ({m},).")

                f[j, mask] = np.mod(ang, 2.0 * np.pi)
                valid[j] = True
                continue

            # 2) PCA2 (PCA2 on points, or MDS2 if Dj is provided)
            if cc_is_pca2:
                if data_is_dist:
                    ang = compute_circular_coords_pca2(X=None, dist_mat=Dj)
                else:
                    ang = compute_circular_coords_pca2(Xj, dist_mat=Dj)
                f[j, mask] = ang
                valid[j] = True
                continue

            # 3) Dreimac config
            assert cc_is_dreimac
            ang, retries, n_lmks = compute_circular_coords_dreimac(
                Xj,
                dist_mat=Dj,  # if Dj is not None, dreimac uses distance_matrix=True internally
                n_landmarks_init=cc.landmarks_per_patch,
                prime=cc.prime,
                update_frac=cc.update_frac,
                standard_range=cc.standard_range,
                CircularCoords_cls=cc.CircularCoords_cls,
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
