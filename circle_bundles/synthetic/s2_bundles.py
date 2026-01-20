# synthetic/s2_bundles.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "sample_sphere",
    "hopf_projection",
    "spin3_adjoint_to_so3",
    "so3_to_s2_projection",
    "sample_s2_trivial",
    "tangent_frame_on_s2",
    "sample_s2_unit_tangent",
]


# ----------------------------
# RNG + numerics helpers
# ----------------------------

def _get_rng(rng: Optional[np.random.Generator]) -> np.random.Generator:
    return np.random.default_rng() if rng is None else rng


def _safe_normalize(x: np.ndarray, *, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Normalize x along `axis`, guarding against near-zero norms."""
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x, axis=axis, keepdims=True)
    nrm = np.maximum(nrm, float(eps))
    return x / nrm


# ----------------------------
# Sphere sampling
# ----------------------------

def sample_sphere(
    n: int,
    dim: int = 2,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample n points ~uniformly from S^{dim} ⊂ R^{dim+1} via Gaussian normalization.

    Examples
    --------
    dim=2 -> S^2 in R^3, output shape (n,3)
    dim=3 -> S^3 in R^4, output shape (n,4)
    """
    n = int(n)
    dim = int(dim)
    if n <= 0:
        raise ValueError(f"n must be positive. Got {n}.")
    if dim < 0:
        raise ValueError(f"dim must be >= 0. Got {dim}.")

    rng = _get_rng(rng)
    x = rng.normal(size=(n, dim + 1))
    return _safe_normalize(x, axis=1)


# ----------------------------
# Hopf / Spin(3) / SO(3) helpers
# ----------------------------

def hopf_projection(
    data: np.ndarray,
    *,
    v: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Generalized Hopf projection defined by q ↦ q v q^{-1}, where v ∈ S^2 ⊂ Im(H).

    Parameters
    ----------
    data : (n,4) real or (n,2) complex
        Quaternion coordinates: q = (a,b,c,d) with z1=a+ib, z2=c+id.
    v : (3,) array-like, optional
        Axis vector in R^3. Will be normalized. Default is e1 = (1,0,0).

    Returns
    -------
    (n,3) array on S^2.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D. Got shape {data.shape}.")

    if v is None:
        v = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        v = np.asarray(v, dtype=float).reshape(3,)
    v = v / max(np.linalg.norm(v), float(eps))

    # Parse quaternion components
    if np.iscomplexobj(data):
        if data.shape[1] != 2:
            raise ValueError(f"complex data must have shape (n,2). Got {data.shape}.")
        z1 = data[:, 0]
        z2 = data[:, 1]
        a, b = z1.real, z1.imag
        c, d = z2.real, z2.imag
    else:
        if data.shape[1] != 4:
            raise ValueError(f"real data must have shape (n,4). Got {data.shape}.")
        a, b, c, d = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    # Normalize q
    nrm = np.sqrt(a * a + b * b + c * c + d * d)
    nrm = np.maximum(nrm, float(eps))
    a, b, c, d = a / nrm, b / nrm, c / nrm, d / nrm

    # Unit quaternion vector rotation:
    # For q = (a, u) with u=(b,c,d), rotate v by:
    # v' = v + 2a(u×v) + 2(u×(u×v))
    u = np.stack([b, c, d], axis=1)  # (n,3)
    v0 = v[None, :]                  # (1,3) broadcasts

    uv = np.cross(u, v0)
    uuv = np.cross(u, uv)

    out = v0 + 2.0 * a[:, None] * uv + 2.0 * uuv
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), float(eps))
    return out


def spin3_adjoint_to_so3(data: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """
    Adjoint map Spin(3) ≅ S^3 (unit quaternions) -> SO(3), vectorized.

    Input formats (same as hopf_projection):
      - real (n,4): [a,b,c,d] = [Re z1, Im z1, Re z2, Im z2]
      - complex (n,2): [z1,z2] where z1=a+ib, z2=c+id

    Returns
    -------
    R_flat : (n,9) array, row-major flattening of 3x3 matrices.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be D. Got shape {data.shape}.")

    if np.iscomplexobj(data):
        if data.shape[1] != 2:
            raise ValueError(f"complex data must have shape (n,2). Got {data.shape}.")
        z1 = data[:, 0]
        z2 = data[:, 1]
        a, b = z1.real, z1.imag
        c, d = z2.real, z2.imag
    else:
        if data.shape[1] != 4:
            raise ValueError(f"real data must have shape (n,4). Got {data.shape}.")
        a, b, c, d = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    nrm = np.sqrt(a * a + b * b + c * c + d * d)
    nrm = np.maximum(nrm, float(eps))
    a, b, c, d = a / nrm, b / nrm, c / nrm, d / nrm

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    ab, ac, ad = a * b, a * c, a * d
    bc, bd, cd = b * c, b * d, c * d

    R_flat = np.empty((a.shape[0], 9), dtype=float)

    # Row-major order:
    # [r00 r01 r02 r10 r11 r12 r20 r21 r22]
    R_flat[:, 0] = aa + bb - cc - dd
    R_flat[:, 1] = 2.0 * (bc - ad)
    R_flat[:, 2] = 2.0 * (bd + ac)

    R_flat[:, 3] = 2.0 * (bc + ad)
    R_flat[:, 4] = aa - bb + cc - dd
    R_flat[:, 5] = 2.0 * (cd - ab)

    R_flat[:, 6] = 2.0 * (bd - ac)
    R_flat[:, 7] = 2.0 * (cd + ab)
    R_flat[:, 8] = aa - bb - cc + dd

    return R_flat


def so3_to_s2_projection(
    R: np.ndarray,
    *,
    v: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Projection SO(3) -> S^2 defined by R ↦ R v, with v ∈ S^2.

    Accepts:
      - (3,3)
      - (n,3,3)
      - (9,) row-major flatten
      - (n,9) row-major flatten

    Default v is e1 = (1,0,0).
    """
    R = np.asarray(R, dtype=float)

    if v is None:
        v = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        v = np.asarray(v, dtype=float).reshape(3,)
    v = v / max(np.linalg.norm(v), float(eps))

    if R.shape == (3, 3):
        return R @ v

    if R.ndim == 3 and R.shape[1:] == (3, 3):
        return np.einsum("nij,j->ni", R, v)

    if R.shape == (9,):
        return R.reshape(3, 3) @ v

    if R.ndim == 2 and R.shape[1] == 9:
        M = R.reshape(-1, 3, 3)
        return np.einsum("nij,j->ni", M, v)

    raise ValueError(f"Expected (3,3), (n,3,3), (9,), or (n,9). Got {R.shape}.")


# ----------------------------
# S^2 bundles / embeddings
# ----------------------------

def sample_s2_trivial(
    n_points: int,
    *,
    sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    radius_mean: float = 1.0,
    radius_clip: Tuple[float, float] = (0.0, 5.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Product bundle S^2 × S^1 embedded as (base ∈ R^3, fiber ∈ R^2) in R^5.

    Returns
    -------
    data : (n_points, 5) = [base_x, base_y, base_z, fiber_u, fiber_v]
    base_points : (n_points, 3) points on S^2
    angles : (n_points,) fiber angles in radians
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")
    rng = _get_rng(rng)

    base_points = sample_sphere(n_points, dim=2, rng=rng)
    angles = 2.0 * np.pi * rng.random(n_points)

    radii = rng.normal(loc=float(radius_mean), scale=float(sigma), size=n_points)
    radii = np.clip(radii, float(radius_clip[0]), float(radius_clip[1]))

    fibers = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    data = np.empty((n_points, 5), dtype=float)
    data[:, :3] = base_points
    data[:, 3:] = fibers
    return data, base_points, angles


def tangent_frame_on_s2(p: np.ndarray, *, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given p on S^2, return a positively oriented orthonormal basis (e1,e2) for T_p S^2.
    Convention: e2 = p × e1.

    Stable except very near poles; near poles uses a consistent fallback.
    """
    p = np.asarray(p, dtype=float).reshape(3,)
    p = _safe_normalize(p, axis=0)

    x, y, _z = p
    r_xy = np.hypot(x, y)

    if r_xy <= float(eps):
        e1 = np.array([1.0, 0.0, 0.0], dtype=float)
        e1 = e1 - np.dot(e1, p) * p
        e1 = _safe_normalize(e1, axis=0)
    else:
        phi = np.arctan2(y, x)
        e1 = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
        e1 = _safe_normalize(e1, axis=0)

    e2 = np.cross(p, e1)
    e2 = _safe_normalize(e2, axis=0)
    return e1, e2


def sample_s2_unit_tangent(
    n_points: int,
    *,
    sigma: float = 0.0,
    equator: bool = False,
    rng: Optional[np.random.Generator] = None,
    radius_mean: float = 1.0,
    radius_clip: Tuple[float, float] = (0.0, 5.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the (scaled) unit tangent bundle of S^2 as (tangent ∈ R^3, base ∈ R^3) in R^6.

    If equator=True, restrict base points to the equator z=0.
    """
    n_points = int(n_points)
    if n_points <= 0:
        raise ValueError(f"n_points must be positive. Got {n_points}.")
    rng = _get_rng(rng)

    if equator:
        theta0 = 2.0 * np.pi * rng.random(n_points)
        base_points = np.column_stack([np.cos(theta0), np.sin(theta0), np.zeros(n_points)])
    else:
        base_points = sample_sphere(n_points, dim=2, rng=rng)

    angles = 2.0 * np.pi * rng.random(n_points)

    radii = rng.normal(loc=float(radius_mean), scale=float(sigma), size=n_points)
    radii = np.clip(radii, float(radius_clip[0]), float(radius_clip[1]))

    x = base_points[:, 0]
    y = base_points[:, 1]
    r_xy = np.hypot(x, y)
    near = r_xy <= 1e-12

    e1 = np.column_stack([-y, x, np.zeros(n_points)])
    e1 = _safe_normalize(e1, axis=1)

    if np.any(near):
        p_near = base_points[near]
        e1_near = np.tile(np.array([1.0, 0.0, 0.0]), (p_near.shape[0], 1))
        e1_near = e1_near - (np.sum(e1_near * p_near, axis=1, keepdims=True) * p_near)
        e1_near = _safe_normalize(e1_near, axis=1)
        e1[near] = e1_near

    e2 = np.cross(base_points, e1)
    e2 = _safe_normalize(e2, axis=1)

    ca = np.cos(angles)[:, None]
    sa = np.sin(angles)[:, None]
    tangent = radii[:, None] * (ca * e1 + sa * e2)

    data = np.empty((n_points, 6), dtype=float)
    data[:, :3] = tangent
    data[:, 3:] = base_points
    return data, base_points, angles
