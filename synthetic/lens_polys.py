from __future__ import annotations

from typing import Optional, Tuple, List
import numpy as np

from .s2_bundles import hopf_projection  # or duplicate if you want no dependency

__all__ = [
    "sample_S3",
    "is_invariant",
    "invariant_monomials",
    "sample_lens",
]


def sample_S3(n_samples: int, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample n points uniformly from S^3 ⊂ C^2. Returns (n,2) complex array (z1,z2)."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got {n_samples}.")
    rng = np.random.default_rng() if rng is None else rng

    u = rng.normal(size=(n_samples, 4))
    u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-12)
    z1 = u[:, 0] + 1j * u[:, 1]
    z2 = u[:, 2] + 1j * u[:, 3]
    return np.column_stack([z1, z2])


def is_invariant(a: int, b: int, p: int, q: int) -> bool:
    """Invariance condition for z1^a z2^b under (z1,z2)->(e^{2πi/p}z1, e^{2πiq/p}z2): a + b*q ≡ 0 (mod p)."""
    return ((a + b * q) % p) == 0


def invariant_monomials(p: int, q: int, max_degree: int) -> List[Tuple[int, int]]:
    """List (a,b) with a,b>=0, a+b<=max_degree, satisfying a + b*q ≡ 0 (mod p)."""
    if p <= 0:
        raise ValueError(f"p must be positive. Got {p}.")
    if max_degree < 0:
        raise ValueError(f"max_degree must be >= 0. Got {max_degree}.")

    monoms: List[Tuple[int, int]] = []
    for a in range(max_degree + 1):
        for b in range(max_degree + 1 - a):
            if is_invariant(a, b, p, q):
                monoms.append((a, b))
    return monoms


def sample_lens(
    n_samples: int,
    p: int,
    *,
    q: int = 1,
    max_degree: int = 6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed a lens space L(p,q) using invariant monomials on S^3 ⊂ C^2.

    Returns
    -------
    data : (n_samples, 2*M) real features [Re,Im] per invariant monomial
    base_points : (n_samples, 3) Hopf projection to S^2
    zs : (n_samples, 4) real coords [Re(z1),Im(z1),Re(z2),Im(z2)]
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got {n_samples}.")
    if p <= 0:
        raise ValueError(f"p must be positive. Got {p}.")

    rng = np.random.default_rng() if rng is None else rng

    z = sample_S3(n_samples, rng=rng)
    z1 = z[:, 0]
    z2 = z[:, 1]
    zs = np.stack([z1.real, z1.imag, z2.real, z2.imag], axis=1)

    monoms = invariant_monomials(p=p, q=q, max_degree=max_degree)
    if len(monoms) == 0:
        raise ValueError("No invariant monomials found—try increasing max_degree or check (p,q).")

    M = len(monoms)
    feats = np.empty((n_samples, M), dtype=np.complex128)
    for j, (a, b) in enumerate(monoms):
        feats[:, j] = (z1 ** a) * (z2 ** b)

    data = np.empty((n_samples, 2 * M), dtype=float)
    data[:, 0::2] = feats.real
    data[:, 1::2] = feats.imag

    base_points = hopf_projection(z1, z2)
    return data, base_points, zs
