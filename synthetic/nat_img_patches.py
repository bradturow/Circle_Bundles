from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["sample_nat_img_kb", "get_gradient_dirs"]


def sample_nat_img_kb(
    n_points: int,
    *,
    n: int = 3,
    noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12,
    return_angles: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthetic "natural image" patch model parameterizing a Klein bottle.

    Returns mean-centered and L2-normalized patches built from a 1D quadratic + linear
    profile along a direction in RP^1 (theta folded to [0, pi)), with a corresponding
    alpha "fiber" angle adjusted so the model is continuous under the identification.

    Parameters
    ----------
    n_points : int
    n : int, default 3
        Patch width/height. Must be odd.
    noise : float, default 0
        Additive Gaussian noise on raw patch entries (before normalization).
    rng : np.random.Generator, optional
    eps : float
        Stability floor for norm.
    return_angles : bool
        If True, also return (alpha, theta) used for each sample.

    Returns
    -------
    data : (n_points, n^2) ndarray
        Mean-centered and L2-normalized patch vectors.
    base_points : (n_points, 2) ndarray
        (cos(theta), sin(theta)) with theta folded to [0, pi).
    alpha : (n_points,) ndarray, optional
    theta : (n_points,) ndarray, optional
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")
    if n <= 0:
        raise ValueError(f"n must be positive. Got {n}.")
    if n % 2 != 1:
        raise ValueError(f"n should be odd. Got n={n}.")

    rng = np.random.default_rng() if rng is None else rng

    angles = 2.0 * np.pi * rng.random((n_points, 2))
    alpha = angles[:, 0].copy()
    theta = angles[:, 1].copy()

    # Fold theta into [0, pi) and adjust alpha accordingly.
    # This matches your intent: (theta, alpha) ~ (theta - pi, 2pi - alpha)
    mask = theta > np.pi
    alpha[mask] = (2.0 * np.pi) - alpha[mask]
    theta[mask] = theta[mask] - np.pi

    # keep alpha in [0,2pi)
    alpha = np.mod(alpha, 2.0 * np.pi)

    a = np.cos(theta)
    b = np.sin(theta)
    c = np.cos(alpha)
    d = np.sin(alpha)

    coords = np.arange(n, dtype=float) - (n - 1) / 2.0
    X, Y = np.meshgrid(coords, coords, indexing="ij")  # (n,n)

    # L = a*x + b*y, broadcast over samples
    L = a[:, None, None] * X[None, :, :] + b[:, None, None] * Y[None, :, :]

    # patch = c*L^2 + d*L
    patches = c[:, None, None] * (L ** 2) + d[:, None, None] * L
    vect = patches.reshape(n_points, n * n)

    if noise != 0.0:
        vect = vect + rng.normal(0.0, float(noise), size=vect.shape)

    # mean-center then L2 normalize
    vect = vect - vect.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vect, axis=1, keepdims=True)
    norms = np.maximum(norms, float(eps))
    data = vect / norms

    base_points = np.column_stack([a, b])

    if return_angles:
        return data, base_points, alpha, theta
    return data, base_points


def get_gradient_dirs(
    patches: np.ndarray,
    n: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute predominant GRADIENT-axis angles in RP^1 for n×n grayscale patches.

    Conventions:
      - If patches is (N, n^2), reshaped with order='C' to (N, n, n).
      - Axis meaning after reshape:
          axis=1 is "x" (rows), axis=2 is "y" (cols).
      - Uses a Sobel-style separable derivative/smoothing.
      - Returns gradient-axis (not edge-axis).

    Parameters
    ----------
    patches : (N, n^2) or (N, n, n) ndarray
        Grayscale patches.
    n : int or None
        Patch width/height. If None and patches is flat, inferred from n^2.
    eps : float
        Degeneracy threshold.

    Returns
    -------
    thetas : (N,) ndarray
        Angles in [0, pi) representing dominant gradient axis (RP^1).
    strengths : (N,) ndarray
        Largest eigenvalue of structure tensor (confidence).
    """
    patches = np.asarray(patches, dtype=float)

    if patches.ndim == 3:
        # already (N,n,n)
        N, n1, n2 = patches.shape
        if n1 != n2:
            raise ValueError(f"Expected square patches (N,n,n). Got {patches.shape}.")
        if n is not None and int(n) != n1:
            raise ValueError(f"Provided n={n} but patches have n={n1}.")
        n = n1
        img = patches
    elif patches.ndim == 2:
        N, L = patches.shape
        if n is None:
            n_guess = int(np.rint(np.sqrt(L)))
            if n_guess * n_guess != L:
                raise ValueError(f"Cannot infer n from L={L} (not a perfect square).")
            n = n_guess
        if n * n != L:
            raise ValueError(f"Expected patch length n^2. Got L={L}, n={n}.")
        img = patches.reshape(N, n, n, order="C")
    else:
        raise ValueError("patches must have shape (N, n^2) or (N, n, n).")

    # Separable Sobel-like kernels
    d = np.array([-1.0, 0.0, 1.0], dtype=float)
    s = np.array([ 1.0, 2.0, 1.0], dtype=float)

    def conv1d_axis(A: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
        # A is (N,n,n); axis in {1,2}
        out = np.zeros_like(A)
        mid = [slice(None)] * 3
        lo  = [slice(None)] * 3
        hi  = [slice(None)] * 3

        mid[axis] = slice(1, -1)
        lo[axis]  = slice(0, -2)
        hi[axis]  = slice(2, None)

        out[tuple(mid)] = (
            k[0] * A[tuple(lo)] +
            k[1] * A[tuple(mid)] +
            k[2] * A[tuple(hi)]
        )

        # crude borders (forward/backward)
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s2 = [slice(None)] * 3

        s0[axis] = 0
        s1[axis] = 1
        s2[axis] = 2
        out[tuple(s0)] = k[0]*A[tuple(s0)] + k[1]*A[tuple(s1)] + k[2]*A[tuple(s2)]

        s0[axis] = -1
        s1[axis] = -2
        s2[axis] = -3
        out[tuple(s0)] = k[0]*A[tuple(s2)] + k[1]*A[tuple(s1)] + k[2]*A[tuple(s0)]

        return out

    # fx = ∂/∂x (axis=1), fy = ∂/∂y (axis=2)
    tmp = conv1d_axis(img, s, axis=2)   # smooth along y
    fx  = conv1d_axis(tmp, d, axis=1)   # derivative along x

    tmp = conv1d_axis(img, s, axis=1)   # smooth along x
    fy  = conv1d_axis(tmp, d, axis=2)   # derivative along y

    fx /= 8.0
    fy /= 8.0

    # Structure tensor components (summed over pixels)
    Jxx = np.sum(fx * fx, axis=(1, 2))
    Jxy = np.sum(fx * fy, axis=(1, 2))
    Jyy = np.sum(fy * fy, axis=(1, 2))

    # Dominant gradient axis in RP^1
    thetas = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy))
    thetas = np.mod(thetas, np.pi)

    # Strength = largest eigenvalue
    strengths = 0.5 * ((Jxx + Jyy) + np.sqrt((Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2)))

    # Degenerate patches
    deg = (Jxx + Jyy) < float(eps)
    thetas[deg] = 0.0
    strengths[deg] = 0.0

    return thetas, strengths
