from __future__ import annotations

from typing import Literal, Optional, Tuple, Union, overload

import numpy as np
from scipy.spatial.transform import Rotation as R

Rule = Optional[Literal["fiber", "equator"]]

__all__ = ["sample_SO3", "project_O3"]


def _unit(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n <= eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return x / n


def _orthonormal_frame_from_first_column(
    u: np.ndarray,
    angles: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Build rotation matrices whose first column is u[i], and whose remaining
    columns are obtained by choosing a canonical tangent basis and rotating it
    by angles[i] about u[i].

    Parameters
    ----------
    u : (n,3) ndarray
        Unit vectors = first columns.
    angles : (n,) ndarray
        Rotation angles around u.

    Returns
    -------
    mats : (n,3,3) ndarray
        Rotation matrices with first column u.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2 or u.shape[1] != 3:
        raise ValueError(f"u must have shape (n,3). Got {u.shape}.")
    n = u.shape[0]

    angles = np.asarray(angles, dtype=float)
    if angles.shape != (n,):
        raise ValueError(f"angles must have shape (n,). Got {angles.shape} vs n={n}.")

    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    ey = np.array([0.0, 1.0, 0.0], dtype=float)

    # choose helper axis per-row: use ex unless too parallel, else ey
    dotx = np.abs(u @ ex)  # (n,)
    a = np.where(dotx[:, None] < 0.9, ex[None, :], ey[None, :])  # (n,3)

    orth1 = np.cross(u, a)
    n1 = np.linalg.norm(orth1, axis=1, keepdims=True)
    if np.any(n1 <= eps):
        raise ValueError("Failed to build tangent direction; encountered near-parallel configuration.")
    orth1 = orth1 / n1

    orth2 = np.cross(u, orth1)

    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    col2 = c * orth1 + s * orth2
    col3 = np.cross(u, col2)

    mats = np.stack([u, col2, col3], axis=2)  # stack as columns
    return mats


# --- nicer type signatures (optional but helpful) ---
@overload
def sample_SO3(
    n_samples: int,
    *,
    rule: None = None,
    v: Optional[np.ndarray] = ...,
    rng: Optional[np.random.Generator] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def sample_SO3(
    n_samples: int,
    *,
    rule: Literal["fiber"],
    v: Optional[np.ndarray] = ...,
    rng: Optional[np.random.Generator] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def sample_SO3(
    n_samples: int,
    *,
    rule: Literal["equator"],
    v: Optional[np.random.Generator] = ...,
    rng: Optional[np.random.Generator] = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...


def sample_SO3(
    n_samples: int,
    *,
    rule: Rule = None,
    v: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample SO(3) (flattened rotation matrices) with optional structured rules.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    rule : None | 'fiber' | 'equator'
        - None: Haar-random rotations; returns (data, base_points) where base_points are first columns.
        - 'fiber': fix first column to v (or random) and sample a fiber angle θ;
                   returns (data, theta) with theta shape (n_samples,).
        - 'equator': choose u on great circle orthogonal to v via angle φ, then choose fiber angle θ;
                     returns (data, angles) with angles shape (n_samples,2) = (phi, theta).
    v : (3,) array-like, optional
        If None, sampled randomly.
    rng : np.random.Generator, optional

    Returns
    -------
    data : (n_samples, 9) ndarray
        Flattened rotation matrices (row-major from (n,3,3) reshape).
    extra : ndarray
        Depends on rule (base_points / theta / [phi,theta]).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got {n_samples}.")
    rng = np.random.default_rng() if rng is None else rng

    if rule is None:
        rotations = R.random(n_samples, random_state=rng)
        mats = rotations.as_matrix()  # (n,3,3)
        data = mats.reshape(n_samples, 9)
        base_points = mats[:, :, 0]   # first column
        return data, base_points

    # normalize v (or sample it)
    if v is None:
        v0 = _unit(rng.normal(size=3))
    else:
        v0 = _unit(np.asarray(v, dtype=float))
    v0 = v0.reshape(1, 3)

    if rule == "fiber":
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        u = np.repeat(v0, repeats=n_samples, axis=0)  # (n,3)
        mats = _orthonormal_frame_from_first_column(u, theta)
        data = mats.reshape(n_samples, 9)
        return data, theta

    if rule == "equator":
        # pick u on great circle orthogonal to v0 via φ
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)

        vflat = v0.ravel()

        # choose an "up" axis not too parallel to v
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        if np.abs(float(vflat @ up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        b1 = _unit(np.cross(vflat, up))
        b2 = np.cross(vflat, b1)  # unit automatically

        u = (np.cos(phi)[:, None] * b1[None, :]) + (np.sin(phi)[:, None] * b2[None, :])
        mats = _orthonormal_frame_from_first_column(u, theta)
        data = mats.reshape(n_samples, 9)
        angles = np.column_stack([phi, theta])
        return data, angles

    raise ValueError(f"Unknown rule={rule!r}. Expected None, 'fiber', or 'equator'.")


def project_O3(O3_data: np.ndarray, v: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Given flattened O(3) matrices (N, 9), return the image of v under each matrix.
    """
    O3_data = np.asarray(O3_data, dtype=float)
    if O3_data.ndim != 2 or O3_data.shape[1] != 9:
        raise ValueError(f"O3_data must have shape (N,9). Got {O3_data.shape}.")

    if v is None:
        v = np.array([1.0, 0.0, 0.0], dtype=float)
    v = np.asarray(v, dtype=float).reshape(3,)

    mats = O3_data.reshape(-1, 3, 3)
    return mats @ v
