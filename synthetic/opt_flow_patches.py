# synthetic/opt_flow_patches.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["sample_opt_flow_torus", "make_flow_patches"]


def _fold_theta_alpha_to_rp1(alpha: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fold theta into [0, pi) (RP^1 base angle), and adjust alpha so the map is consistent.

    Convention:
      if theta > pi:
          theta <- theta - pi
          alpha <- (alpha - pi) mod 2pi
    """
    alpha = np.asarray(alpha, dtype=float).copy()
    theta = np.asarray(theta, dtype=float).copy()

    mask = theta > np.pi
    theta[mask] = theta[mask] - np.pi
    alpha[mask] = (alpha[mask] - np.pi) % (2.0 * np.pi)
    return alpha, theta


def sample_opt_flow_torus(
    n_points: int,
    *,
    dim: int = 3,
    sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample a torus embedded in S^{2*dim^2 - 1} (optical-flow patch model),
    with RP^1 base angle theta folded to [0, pi).

    Returns
    -------
    data : (n_points, 2*dim^2) ndarray
        Flattened flow patches (u part concat v part) in your DCT basis model.
    base_points : (n_points, 2) ndarray
        (cos(theta), sin(theta)) with theta in [0, pi).
    alpha : (n_points,) ndarray
        Fiber angle used for each sample (after folding adjustment).
    """
    n_points = int(n_points)
    dim = int(dim)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")
    if dim <= 0:
        raise ValueError(f"dim must be positive. Got {dim}.")

    rng = np.random.default_rng() if rng is None else rng

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha0 = angles[:, 0]
    theta0 = angles[:, 1]
    alpha, theta = _fold_theta_alpha_to_rp1(alpha0, theta0)

    # Local import keeps synthetic usable even if optical_flow isn't installed.
    from optical_flow.contrast import get_dct_basis

    e1u, e2u, e1v, e2v = get_dct_basis(dim, normalize=True, opt_flow=True, top_two=True)

    u = np.outer(np.cos(alpha), e1u) + np.outer(np.sin(alpha), e2u)
    v = np.outer(np.cos(alpha), e1v) + np.outer(np.sin(alpha), e2v)

    data = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
    base_points = np.column_stack([np.cos(theta), np.sin(theta)])

    if sigma != 0.0:
        data = data + rng.normal(0.0, float(sigma), size=data.shape)

    return data, base_points, alpha


# ---- Hardwired 3x3 “extended torus” flow patches (length 18) ----

_E1U_3x3 = (1 / np.sqrt(6)) * np.array(
    [1, 0, -1,
     1, 0, -1,
     1, 0, -1,
     0, 0,  0,
     0, 0,  0,
     0, 0,  0],
    dtype=float,
)

_E2U_3x3 = (1 / np.sqrt(6)) * np.array(
    [1, 1,  1,
     0, 0,  0,
    -1, -1, -1,
     0, 0,  0,
     0, 0,  0,
     0, 0,  0],
    dtype=float,
)

_E1V_3x3 = (1 / np.sqrt(6)) * np.array(
    [0, 0, 0,
     0, 0, 0,
     0, 0, 0,
     1, 0, -1,
     1, 0, -1,
     1, 0, -1],
    dtype=float,
)

_E2V_3x3 = (1 / np.sqrt(6)) * np.array(
    [0, 0, 0,
     0, 0, 0,
     0, 0, 0,
     1, 1, 1,
     0, 0, 0,
    -1, -1, -1],
    dtype=float,
)


def make_flow_patches(
    alpha: np.ndarray,
    theta: np.ndarray,
    r: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate samples from the extended torus optical-flow patch model (3x3, length 18).

    Folds theta into [0, pi) (RP^1 base) and adjusts alpha accordingly.

    If r is provided, it mixes a patch with a perpendicular patch:
        patches_mix = λ * patches + sqrt(1-λ^2) * patches_perp
    where λ = 1/sqrt(2-r).
    """
    alpha, theta = _fold_theta_alpha_to_rp1(alpha, theta)

    def _core(a: np.ndarray, t: np.ndarray) -> np.ndarray:
        ca = np.cos(a)
        sa = np.sin(a)
        ct = np.cos(t)
        st = np.sin(t)

        u = np.outer(ca, _E1U_3x3) + np.outer(sa, _E2U_3x3)
        v = np.outer(ca, _E1V_3x3) + np.outer(sa, _E2V_3x3)
        return ct[:, None] * u + st[:, None] * v

    patches = _core(alpha, theta)

    if r is None:
        return patches

    r = np.asarray(r, dtype=float).reshape(-1)
    N = patches.shape[0]
    if r.shape != (N,):
        raise ValueError(f"r must have shape (N,), got {r.shape} vs N={N}.")

    # Perp patch = (alpha + pi/2, theta - pi/2), then re-fold.
    alpha_perp = alpha + (np.pi / 2.0)
    theta_perp = theta - (np.pi / 2.0)
    alpha_perp, theta_perp = _fold_theta_alpha_to_rp1(alpha_perp, theta_perp)
    perp = _core(alpha_perp, theta_perp)

    denom = 2.0 - r
    if np.any(denom <= 0.0):
        raise ValueError("r must satisfy r < 2 everywhere (so that 2-r is positive).")

    lamb = 1.0 / np.sqrt(denom)
    lamb2 = np.sqrt(np.maximum(0.0, 1.0 - lamb**2))
    return patches * lamb[:, None] + perp * lamb2[:, None]
