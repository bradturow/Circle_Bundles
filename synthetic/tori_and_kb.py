# synthetic/tori_and_kb.py
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np

__all__ = [
    "AngleFunc",
    "const",
    "small_to_big",
    "sample_R3_torus",
    "sample_S3_torus",
    "sample_R4_kb",
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


def sample_R3_torus(
    n_points: int,
    *,
    noise_sigma: float = 0.0,
    R_func: AngleFunc = const(5.0),
    r_func: AngleFunc = const(2.0),
    rng: Optional[np.random.Generator] = None,
    return_theta: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Sample a (possibly variable radii) torus in R^3.

    Returns
    -------
    data : (n_points, 3)
    base_points : (n_points, 2) = (cos θ, sin θ)
    alpha : (n_points,)
    theta : (n_points,)  (if return_theta=True)
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")

    rng = np.random.default_rng() if rng is None else rng

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha = angles[:, 0]
    theta = angles[:, 1]

    R_vals = np.asarray(R_func(alpha), dtype=float)
    r_vals = np.asarray(r_func(theta), dtype=float)
    if R_vals.shape != (n_points,) or r_vals.shape != (n_points,):
        raise ValueError(
            "R_func and r_func must return arrays of shape (n_points,) "
            "when given (n_points,) input."
        )

    if noise_sigma != 0.0:
        r_vals = r_vals + rng.normal(loc=0.0, scale=float(noise_sigma), size=n_points)

    cth = np.cos(theta)
    sth = np.sin(theta)
    cal = np.cos(alpha)
    sal = np.sin(alpha)

    data = np.empty((n_points, 3), dtype=float)
    data[:, 0] = (R_vals + r_vals * cal) * cth
    data[:, 1] = (R_vals + r_vals * cal) * sth
    data[:, 2] = r_vals * sal

    base_points = np.column_stack([cth, sth])

    if return_theta:
        return data, base_points, alpha, theta
    return data, base_points, alpha


def sample_S3_torus(
    n_points: int,
    *,
    noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    project_back: bool = False,
    return_theta: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Sample a Clifford-torus-style embedding in S^3 ⊂ R^4 with base angle theta ∈ [0, π).

    Returns
    -------
    data : (n_points, 4)
    base_points : (n_points, 2) = (cos theta, sin theta)
    alpha : (n_points,)
    theta : (n_points,) (if return_theta=True)
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")

    rng = np.random.default_rng() if rng is None else rng

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha = angles[:, 0].copy()
    theta = angles[:, 1].copy()

    # Fold theta into [0, pi) and adjust alpha accordingly.
    mask = theta > np.pi
    theta[mask] = theta[mask] - np.pi
    alpha[mask] = (alpha[mask] - np.pi) % (2.0 * np.pi)

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    data = np.empty((n_points, 4), dtype=float)
    data[:, 0] = ct * ca
    data[:, 1] = ct * sa
    data[:, 2] = st * ca
    data[:, 3] = st * sa

    if noise_sigma != 0.0:
        data = data + rng.normal(loc=0.0, scale=float(noise_sigma), size=data.shape)

    if project_back:
        norms = np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
        data = data / norms

    base_points = np.column_stack([ct, st])

    if return_theta:
        return data, base_points, alpha, theta
    return data, base_points, alpha


def sample_R4_kb(
    n_points: int,
    *,
    noise_sigma: float = 0.0,
    R_func: AngleFunc = const(5.0),
    r_func: AngleFunc = const(2.0),
    rng: Optional[np.random.Generator] = None,
    return_theta: bool = False,
    return_alpha: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Sample a (variable radii) Klein bottle embedding in R^4.

    Returns
    -------
    data : (n_points, 4)
    base_points : (n_points, 2) = (cos theta, sin theta)
    theta : (n_points,) (optional)
    alpha : (n_points,) (optional)
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")

    rng = np.random.default_rng() if rng is None else rng

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha = angles[:, 0]
    theta = angles[:, 1]

    R_vals = np.asarray(R_func(alpha), dtype=float)
    r_vals = np.asarray(r_func(theta), dtype=float)
    if R_vals.shape != (n_points,) or r_vals.shape != (n_points,):
        raise ValueError(
            "R_func and r_func must return arrays of shape (n_points,) "
            "when given (n_points,) input."
        )

    if noise_sigma != 0.0:
        r_vals = r_vals + rng.normal(loc=0.0, scale=float(noise_sigma), size=n_points)

    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cth2 = np.cos(theta / 2.0)
    sth2 = np.sin(theta / 2.0)

    data = np.empty((n_points, 4), dtype=float)
    data[:, 0] = (R_vals + r_vals * ca) * ct
    data[:, 1] = (R_vals + r_vals * ca) * st
    data[:, 2] = r_vals * sa * cth2
    data[:, 3] = r_vals * sa * sth2

    base_points = np.column_stack([ct, st])

    if return_theta and return_alpha:
        return data, base_points, theta, alpha
    if return_theta:
        return data, base_points, theta
    if return_alpha:
        return data, base_points, alpha
    return data, base_points
