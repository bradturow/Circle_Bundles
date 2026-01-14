# synthetic/tori_and_kb.py
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

__all__ = [
    "AngleFunc",
    "const",
    "small_to_big",
    "wrap_angle",
    "sample_C2_torus",
    "torus_base_projection_from_data",
    "kb_pairwise_distances_from_data",
]

AngleFunc = Callable[[np.ndarray], np.ndarray]


def const(value: float) -> AngleFunc:
    value_f = float(value)

    def f(angle: np.ndarray) -> np.ndarray:
        angle = np.asarray(angle)
        return np.full_like(angle, fill_value=value_f, dtype=float)

    return f


def small_to_big(smallest: float, largest: float) -> AngleFunc:
    """
    V-shaped function with maximum at angle=pi and minimum at angle=0,2pi.

    Returns values in [smallest, largest] for angles in [0,2pi].
    """
    smallest_f = float(smallest)
    largest_f = float(largest)
    slope = (smallest_f - largest_f) / np.pi

    def f(angle: np.ndarray) -> np.ndarray:
        angle = np.asarray(angle, dtype=float)
        return slope * np.abs(angle - np.pi) + largest_f

    return f


# ----------------------------
# angle helpers
# ----------------------------

def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles into [0, 2pi)."""
    a = np.asarray(a, dtype=float)
    return np.mod(a, 2.0 * np.pi)


def _angle_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Minimal circular distance on S^1 between angles a and b (both arrays broadcastable).
    Returns values in [0, pi].
    """
    d = np.abs(a - b)
    return np.minimum(d, 2.0 * np.pi - d)


# ----------------------------
# Torus in C^2 = R^4
# ----------------------------

def sample_C2_torus(
    n_points: int,
    *,
    # Base circle radius in the (x3,x4) plane (constant or varying with theta)
    R_func: AngleFunc = const(1.0),
    # Fiber circle radius in the (x1,x2) plane (allowed to vary with theta)
    r_func: AngleFunc = const(1.0),
    # Relative noise level on the radius: r_noisy = r * (1 + sigma * N(0,1))
    sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    # Returns
    return_theta: bool = False,
    return_alpha: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Sample a product-torus-style embedding in R^4 = C^2.

        z1 = r(theta) * exp(i alpha)
        z2 = R(theta) * exp(i theta)

    So the base projection is directly recoverable from data via arg(z2).

    Returns
    -------
    data : (n_points, 4)
    base_points : (n_points, 2) = (cos(theta), sin(theta))
    alpha : (n_points,)    (fiber angle)
    theta : (n_points,)    (base angle) optional
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")

    rng = np.random.default_rng() if rng is None else rng

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha = angles[:, 0]
    theta = angles[:, 1]

    R_vals = np.asarray(R_func(theta), dtype=float).reshape(-1)
    r_vals = np.asarray(r_func(theta), dtype=float).reshape(-1)
    if R_vals.shape != (n_points,) or r_vals.shape != (n_points,):
        raise ValueError(
            "R_func and r_func must return arrays of shape (n_points,) "
            "when given (n_points,) input."
        )
    if np.any(R_vals <= 0.0) or np.any(r_vals <= 0.0):
        raise ValueError("R_func and r_func must be strictly positive everywhere.")

    if sigma != 0.0:
        # relative noise on radius (keeps scale meaningful across varying r(theta))
        r_vals = r_vals * (1.0 + float(sigma) * rng.normal(size=n_points))
        # keep radii positive (rare if sigma is modest, but guard anyway)
        r_vals = np.maximum(r_vals, 1e-12)

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    data = np.empty((n_points, 4), dtype=float)
    data[:, 0] = r_vals * ca
    data[:, 1] = r_vals * sa
    data[:, 2] = R_vals * ct
    data[:, 3] = R_vals * st

    base_points = np.column_stack([ct, st])

    if return_theta and return_alpha:
        return data, base_points, alpha, theta, r_vals
    if return_theta:
        return data, base_points, alpha, theta
    if return_alpha:
        return data, base_points, alpha
    return data, base_points, alpha


def torus_base_projection_from_data(data: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Recover base points (cos theta, sin theta) from data in R^4 by normalizing (x3,x4).
    This works even if R(theta) varies (as long as it's positive).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected data shape (n,4). Got {data.shape}.")
    z2 = data[:, 2:4]
    nrm = np.linalg.norm(z2, axis=1, keepdims=True)
    nrm = np.maximum(nrm, eps)
    return z2 / nrm


# ----------------------------
# Klein bottle metric on the torus parameterization
# ----------------------------

def _kb_angle_dist(
    alpha1: np.ndarray,
    theta1: np.ndarray,
    alpha2: np.ndarray,
    theta2: np.ndarray,
) -> np.ndarray:
    """
    Distance on the Klein bottle viewed as a quotient of the flat torus by:
        (alpha, theta) ~ (alpha + pi, -theta)

    We use the induced quotient metric:
        d_KB(p,q) = min( d_T(p,q), d_T(p, g(q)) )
    where d_T is the flat torus metric with circular distances in each angle,
    and g(alpha,theta) = (alpha+pi, -theta).

    Returns array broadcasted over inputs.
    """
    # torus distance between (a1,t1) and (a2,t2)
    da = _angle_dist(alpha1, alpha2)
    dt = _angle_dist(theta1, theta2)
    d0 = np.sqrt(da * da + dt * dt)

    # torus distance between (a1,t1) and g(a2,t2) = (a2+pi, -t2)
    da_g = _angle_dist(alpha1, wrap_angle(alpha2 + np.pi))
    dt_g = _angle_dist(theta1, wrap_angle(-theta2))
    d1 = np.sqrt(da_g * da_g + dt_g * dt_g)

    return np.minimum(d0, d1)


def kb_pairwise_distances_from_data(
    data: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute an (n,n) distance matrix using the Klein bottle quotient metric
    on angles extracted from the R^4 torus data.

    - alpha is recovered from (x1,x2) via atan2
    - theta is recovered from (x3,x4) via atan2

    This does NOT change the ambient embedding; it gives you a KB-topology metric
    on the same point cloud.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(f"Expected data shape (n,4). Got {data.shape}.")
    n = data.shape[0]

    # Recover angles; robust even if radii vary (atan2 ignores scale)
    alpha = wrap_angle(np.arctan2(data[:, 1], data[:, 0]))
    theta = wrap_angle(np.arctan2(data[:, 3], data[:, 2]))

    # Pairwise quotient metric (O(n^2))
    A1 = alpha[:, None]
    T1 = theta[:, None]
    A2 = alpha[None, :]
    T2 = theta[None, :]

    D = _kb_angle_dist(A1, T1, A2, T2)
    # zero diagonal numerically
    np.fill_diagonal(D, 0.0)
    return D
