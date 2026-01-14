# synthetic/opt_flow_patches.py
from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict, Any
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


def _sample_r_trunc_exp(
    n: int,
    *,
    r_min: float = 0.6,
    lam: float = 8.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample r in [r_min, 1] with density proportional to exp(-lam*(1-r)),
    i.e. exponential falloff away from 1, truncated to [r_min, 1].

    lam controls concentration near 1: larger lam => tighter near 1.
    """
    r_min = float(r_min)
    lam = float(lam)
    if not (0.0 < r_min < 1.0):
        raise ValueError(f"r_min must be in (0,1). Got {r_min}.")
    if lam <= 0.0:
        raise ValueError(f"lam must be > 0. Got {lam}.")

    L = 1.0 - r_min  # max distance from 1
    y = rng.random(n)

    # u ~ Exp(lam) truncated to [0, L], then r = 1-u.
    # CDF_trunc(u) = (1 - exp(-lam*u)) / (1 - exp(-lam*L))
    # Invert: u = -log(1 - y*(1-exp(-lam*L))) / lam
    u = -np.log(1.0 - y * (1.0 - np.exp(-lam * L))) / lam
    r = 1.0 - u
    return r


def sample_opt_flow_torus(
    n_points: int,
    *,
    dim: int = 3,
    sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    # ---- r controls ----
    sample_r: bool = False,
    r_min: float = 0.6,
    r_lam: float = 8.0,
    r_values: Optional[np.ndarray] = None,
    return_r: bool = True,
    # ---- noise / normalization ----
    contrast_renorm: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the optical-flow patch torus model (dim x dim DCT basis, flattened length 2*dim^2),
    with RP^1 base angle theta folded to [0, pi).

    Optionally extends the model with an r-parameter in [r_min, 1] that mixes each patch
    with its perpendicular patch, using:
        perp patch = (alpha + pi/2, theta - pi/2) (then re-fold),
        lambda = 1/sqrt(2 - r),
        patch_mix = lambda*patch + sqrt(1-lambda^2)*perp.

    Default behavior:
      - sample_r=True
      - r ~ truncated exponential away from 1 on [r_min, 1]
        (so virtually all mass near 1, but never below r_min).

    Noise:
      - if sigma > 0 and contrast_renorm=True, we add Gaussian noise then contrast-renormalize.

    Returns
    -------
    data : (n_points, 2*dim^2) ndarray
    base_points : (n_points, 2) ndarray  (cos(theta), sin(theta)), theta in [0, pi)
    alpha : (n_points,) ndarray
    r : (n_points,) ndarray   if return_r=True
    """
    n_points = int(n_points)
    dim = int(dim)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")
    if dim <= 0:
        raise ValueError(f"dim must be positive. Got {dim}.")

    rng = np.random.default_rng() if rng is None else rng

    # Sample (alpha, theta) uniformly on [0, 2pi)^2 then fold to RP^1.
    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha0 = angles[:, 0]
    theta0 = angles[:, 1]
    alpha, theta = _fold_theta_alpha_to_rp1(alpha0, theta0)

    # Local import keeps synthetic usable even if optical_flow isn't installed.
    from optical_flow.contrast import get_dct_basis, get_contrast_norms

    e1u, e2u, e1v, e2v = get_dct_basis(dim, normalize=True, opt_flow=True, top_two=True)

    def _core(a: np.ndarray, t: np.ndarray) -> np.ndarray:
        u = np.outer(np.cos(a), e1u) + np.outer(np.sin(a), e2u)
        v = np.outer(np.cos(a), e1v) + np.outer(np.sin(a), e2v)
        return np.cos(t)[:, None] * u + np.sin(t)[:, None] * v

    data = _core(alpha, theta)
    base_points = np.column_stack([np.cos(theta), np.sin(theta)])

    # ---- r-mixing ----
    r = None
    use_r = sample_r or (r_values is not None)

    if use_r:
        if r_values is not None:
            r = np.asarray(r_values, dtype=float).reshape(-1)
            if r.shape != (n_points,):
                raise ValueError(f"r_values must have shape (n_points,), got {r.shape}.")
            if np.any((r < 0.0) | (r > 1.0)):
                raise ValueError("r_values must lie in [0,1].")
            if np.any(r < float(r_min) - 1e-15):
                raise ValueError(f"r_values must satisfy r >= r_min={r_min}.")
        else:
            r = _sample_r_trunc_exp(n_points, r_min=r_min, lam=r_lam, rng=rng)

        # Perp patch = (alpha + pi/2, theta - pi/2), then re-fold.
        alpha_perp = alpha + (np.pi / 2.0)
        theta_perp = theta - (np.pi / 2.0)
        alpha_perp, theta_perp = _fold_theta_alpha_to_rp1(alpha_perp, theta_perp)
        perp = _core(alpha_perp, theta_perp)

        lamb = 1.0 / np.sqrt(2.0 - r)                  # in [1/sqrt(2), 1]
        lamb2 = np.sqrt(np.maximum(0.0, 1.0 - lamb**2)) # in [0, 1/sqrt(2)]
        data = data * lamb[:, None] + perp * lamb2[:, None]

    # ---- add noise ----
    if sigma != 0.0:
        data = data + rng.normal(0.0, float(sigma), size=data.shape)

        if contrast_renorm:
            norms = np.asarray(get_contrast_norms(data), dtype=float).reshape(-1)
            norms = np.maximum(norms, float(eps))
            data = data / norms[:, None]

    if return_r:
        if r is None:
            # Interpret "no mixing" as r=1 for bookkeeping.
            r = np.ones(n_points, dtype=float)
        return data, base_points, alpha, r

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
    *,
    contrast_renorm: bool = False,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Generate samples from the extended torus optical-flow patch model (3x3, length 18).

    Folds theta into [0, pi) (RP^1 base) and adjusts alpha accordingly.

    If r is provided, it mixes a patch with a perpendicular patch:
        patches_mix = λ * patches + sqrt(1-λ^2) * patches_perp
    where λ = 1/sqrt(2-r).

    If contrast_renorm=True, rescales each patch to unit "contrast norm".
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

    if r is not None:
        r = np.asarray(r, dtype=float).reshape(-1)
        N = patches.shape[0]
        if r.shape != (N,):
            raise ValueError(f"r must have shape (N,), got {r.shape} vs N={N}.")
        if np.any((r < 0.0) | (r > 1.0)):
            raise ValueError("r must lie in [0,1].")

        # Perp patch = (alpha + pi/2, theta - pi/2), then re-fold.
        alpha_perp = alpha + (np.pi / 2.0)
        theta_perp = theta - (np.pi / 2.0)
        alpha_perp, theta_perp = _fold_theta_alpha_to_rp1(alpha_perp, theta_perp)
        perp = _core(alpha_perp, theta_perp)

        lamb = 1.0 / np.sqrt(2.0 - r)                  # in [1/sqrt(2), 1]
        lamb2 = np.sqrt(np.maximum(0.0, 1.0 - lamb**2)) # in [0, 1/sqrt(2)]
        patches = patches * lamb[:, None] + perp * lamb2[:, None]

    if contrast_renorm:
        # Local import keeps synthetic usable even if optical_flow isn't installed.
        from optical_flow.contrast import get_contrast_norms
        norms = np.asarray(get_contrast_norms(patches), dtype=float).reshape(-1)
        norms = np.maximum(norms, eps)
        patches = patches / norms[:, None]

    return patches
