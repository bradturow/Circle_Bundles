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
    "sample_foldy_klein_bottle",
    "sample_R3_torus",
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



def sample_R3_torus(
    n_points: int,
    *,
    # Major radius (constant)
    R: float = 2.0,
    # Minor radius r(theta) = center + amplitude * sin(frequency * theta + phase)
    r_center: float = 0.8,
    r_amplitude: float = 0.3,
    r_frequency: int = 2,
    r_phase: float = 0.0,
    # Relative noise on minor radius: r_noisy = r * (1 + sigma * N(0,1))
    sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    require_ring_torus: bool = True,
    # Returns
    return_theta: bool = False,
    return_alpha: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Sample a torus-of-revolution in R^3 where the tube (fiber) radius varies sinusoidally with base angle theta.

    Parameterization:
        r(theta) = r_center + r_amplitude * sin(r_frequency * theta + r_phase)

        x = (R + r(theta) cos(alpha)) cos(theta)
        y = (R + r(theta) cos(alpha)) sin(theta)
        z = r(theta) sin(alpha)

    Returns
    -------
    data : (n_points, 3)
    base_points : (n_points, 2) = (cos(theta), sin(theta))
    alpha : (n_points,)       fiber angle
    theta : (n_points,)       base angle (optional)
    r_vals : (n_points,)      realized tube radius after noise (only returned when
                              return_theta and return_alpha are both True)
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")

    rng = np.random.default_rng() if rng is None else rng

    R = float(R)
    r_center = float(r_center)
    r_amplitude = float(r_amplitude)
    r_phase = float(r_phase)
    r_frequency = int(r_frequency)

    if R <= 0.0:
        raise ValueError(f"R must be > 0. Got {R}.")
    if r_frequency < 0:
        raise ValueError(f"r_frequency must be >= 0. Got {r_frequency}.")
    if r_center <= 0.0:
        raise ValueError(f"r_center must be > 0. Got {r_center}.")
    if r_amplitude < 0.0:
        raise ValueError(f"r_amplitude must be >= 0. Got {r_amplitude}.")

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha = angles[:, 0]
    theta = angles[:, 1]

    # sinusoidal tube radius
    r_vals = r_center + r_amplitude * np.sin(r_frequency * theta + r_phase)

    if np.any(r_vals <= 0.0):
        rmin = float(np.min(r_vals))
        raise ValueError(
            "Sinusoidal r(theta) became non-positive. "
            f"Minimum value was {rmin:.6g}. "
            "Increase r_center or decrease r_amplitude."
        )

    if sigma != 0.0:
        r_vals = r_vals * (1.0 + float(sigma) * rng.normal(size=n_points))
        r_vals = np.maximum(r_vals, 1e-12)

    if require_ring_torus:
        # pointwise ring torus condition (prevents axis-crossing)
        if np.any(R <= r_vals):
            bad = np.where(R <= r_vals)[0]
            i0 = int(bad[0])
            raise ValueError(
                "require_ring_torus=True but found theta where R <= r(theta). "
                f"Example index {i0}: R={R:.6g}, r={r_vals[i0]:.6g}. "
                "Increase R, decrease r_center/r_amplitude, or set require_ring_torus=False."
            )

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)

    radial = R + r_vals * ca
    data = np.empty((n_points, 3), dtype=float)
    data[:, 0] = radial * ct
    data[:, 1] = radial * st
    data[:, 2] = r_vals * sa

    base_points = np.column_stack([ct, st])

    if return_theta and return_alpha:
        return data, base_points, alpha, theta, r_vals
    if return_theta:
        return data, base_points, alpha, theta
    if return_alpha:
        return data, base_points, alpha
    return data, base_points, alpha





# ------------------------------------------------------------
# Folded circle in R^4 (just add w=0)
# ------------------------------------------------------------

def folded_circle_r4(theta: np.ndarray, amp: float) -> np.ndarray:
    """
    theta: (n,) array
    returns: (n,4) points (cosθ, sinθ, amp*sin(2θ), 0)
    """
    theta = np.asarray(theta, float)
    return np.column_stack([
        np.cos(theta),
        np.sin(theta),
        amp * np.sin(2.0 * theta),
        np.zeros_like(theta),
    ])

# ------------------------------------------------------------
# SO(4) path implementing the reflection symmetry as a rotation
# ------------------------------------------------------------

def _rodrigues_3d(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    3D Rodrigues rotation matrix for unit axis and angle.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    ax, ay, az = axis
    K = np.array([[0, -az, ay],
                  [az, 0, -ax],
                  [-ay, ax, 0]], dtype=float)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def G_of_t(t: float) -> np.ndarray:
    """
    Returns G(t) in SO(4) such that:
      - G(0) = I
      - G(2π) acts as (x,y,z,w) -> (y,x,z,-w)

    Construction:
      rotate by angle alpha=t/2 in the 3D subspace spanned by (x,y,w),
      about the axis (1,1,0) (in (x,y,w)-coords).
      z is left fixed.
    """
    alpha = 0.5 * float(t)  # goes 0 -> π as t goes 0 -> 2π
    axis_xyw = np.array([1.0, 1.0, 0.0])  # axis in (x,y,w)
    R3 = _rodrigues_3d(axis_xyw, alpha)

    # Place R3 into a 4x4 acting on indices [0,1,3] (x,y,w), keeping z (index 2) fixed.
    G = np.eye(4)
    idx = [0, 1, 3]
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            G[ii, jj] = R3[i, j]
    return G




# --- keep your existing folded_circle_r4, _rodrigues_3d, G_of_t ---


def _random_orthogonal(D: int, rng: np.random.Generator, det_sign: int = +1) -> np.ndarray:
    """
    Haar-ish random orthogonal matrix via QR. Optionally force det = +1 (SO(D)).
    """
    A = rng.normal(size=(D, D))
    Q, R = np.linalg.qr(A)
    # make Q deterministic w.r.t QR sign convention
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q @ np.diag(s)

    if det_sign is not None:
        if det_sign not in (+1, -1):
            raise ValueError("det_sign must be +1, -1, or None.")
        if np.linalg.det(Q) * det_sign < 0:
            Q[:, 0] *= -1.0  # flip one column to change determinant
    return Q


def sample_foldy_klein_bottle(
    n: int,
    *,
    amp_fiber: float = 1.0,
    amp_base: Optional[float] = 1.0,
    base_radius: float = 1.0,   # only used if amp_base is None (plain circle base)
    noise: float = 0.05,
    rigid_motion: bool = True,
    translate_scale: float = 0.0,
    det_sign: int = +1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Like sample_foldy_klein_bottle, but optionally:
      - embed the base circle nonlinearly in R^4 (using folded_circle_r4),
      - then apply a global rigid motion in ambient space to intermix coordinates.

    Returns
    -------
    X : (n, D) array
        Ambient coordinates. D=6 if base is 2D circle; D=8 if base is 4D folded.
    t : (n,) array
        Base parameter (ground truth).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Base + fiber parameters
    t = rng.uniform(0.0, 2.0 * np.pi, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)

    # --- Base embedding ---
    if amp_base is None:
        # original base in R^2
        base = np.column_stack([
            base_radius * np.cos(t),
            base_radius * np.sin(t),
        ])
    else:
        # folded base in R^4 (same style as fiber)
        # NOTE: base_radius can be absorbed by scaling; we keep it for convenience
        base = base_radius * folded_circle_r4(t, amp=float(amp_base))  # (n,4)

    # --- Fiber embedding (same as yours) ---
    fib0 = folded_circle_r4(theta, amp=float(amp_fiber))  # (n,4)

    fib = np.empty_like(fib0)
    for i in range(n):
        G = G_of_t(t[i])
        fib[i] = G @ fib0[i]

    # --- Combine ---
    X = np.hstack([base, fib])  # (n, 2+4) or (n, 4+4)

    # --- Global rigid motion (mix coordinates) ---
    if rigid_motion:
        D = X.shape[1]
        Q = _random_orthogonal(D, rng=rng, det_sign=det_sign)  # det_sign=+1 => SO(D)
        X = X @ Q  # right-multiply: mixes coordinate axes
        if translate_scale and translate_scale != 0.0:
            X = X + rng.normal(scale=float(translate_scale), size=(1, D))

    # --- Ambient noise ---
    if noise > 0.0:
        X = X + rng.normal(scale=float(noise), size=X.shape)

    return X, t
