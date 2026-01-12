# circle_bundles/local_triv.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import warnings

from .status_utils import _status, _status_clear  # consistent status printing


@dataclass
class LocalTrivResult:
    f: np.ndarray
    valid: np.ndarray
    n_retries: np.ndarray
    n_landmarks: np.ndarray
    errors: Dict[int, str]


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

            # Robust wrap to circle (does not change your convention; just enforces it)
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

    # total-space metric object (vectorized): expects .pairwise(X)->(m,m)
    total_metric: Optional[object] = None,

    # Dreimac defaults
    landmarks_per_patch: int = 200,
    prime: int = 41,
    update_frac: float = 0.25,
    standard_range: bool = False,
    CircularCoords_cls=None,

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

    If total_metric is provided, Dreimac runs on the induced distance matrix
    total_metric.pairwise(Xj) with distance_matrix=True.
    """
    data = np.asarray(data)
    U = np.asarray(U, dtype=bool)

    if U.ndim != 2:
        raise ValueError(f"U must be 2D (n_sets, n_samples). Got shape {U.shape}.")

    n_sets, n_samples = U.shape
    if data.shape[0] != n_samples:
        raise ValueError(f"data has n={data.shape[0]} samples but U has n_samples={n_samples}.")

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
            if cc_alg is not None:
                ang = np.asarray(cc_alg(Xj), dtype=float).reshape(-1)
                if ang.shape != (m,):
                    raise ValueError(f"cc_alg returned shape {ang.shape}, expected ({m},).")
                ang = np.mod(ang, 2.0 * np.pi)
                f[j, mask] = ang
                valid[j] = True

            else:
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
                if ang.shape != (m,):
                    raise ValueError(f"Dreimac returned shape {ang.shape}, expected ({m},).")

                # Store (radians) on the masked positions
                f[j, mask] = ang
                valid[j] = True
                n_retries[j] = int(retries)
                n_landmarks[j] = int(n_lmks)

                if verbose and retries > 0:
                    _status(f"Coordinatizing set {j+1}/{n_sets}... (retries={retries}, n_landmarks={n_lmks})")

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
