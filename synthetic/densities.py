# circle_bundles/synthetic/densities.py
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

__all__ = [
    "mesh_to_density",
    "get_density_axes",
    "rotate_density",
    "get_mesh_sample",
]


def mesh_to_density(
    mesh: trimesh.Trimesh,
    *,
    grid_size: int = 32,
    sigma: float = 0.05,
    n_surface_samples: int = 5000,
    normalize: bool = True,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Convert a trimesh object into a flattened density on a cubic grid in [-1,1]^3
    by sampling the surface and placing a Gaussian in distance-to-surface.

    Notes
    -----
    - The mesh is copied, centered (center_mass), and scaled into ~unit ball.
    - Deterministic sampling via rng is not guaranteed across trimesh versions;
      rng is accepted for API consistency but is best-effort.

    Returns
    -------
    density : (grid_size^3,) ndarray
        Flattened density values.
    """
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive. Got {grid_size}.")
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0. Got {sigma}.")
    if n_surface_samples <= 0:
        raise ValueError(f"n_surface_samples must be positive. Got {n_surface_samples}.")

    # Normalize mesh to lie in (almost) the unit ball
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.center_mass)

    scale = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
    if scale <= eps:
        raise ValueError("Mesh appears degenerate (near-zero scale).")
    mesh.apply_scale(0.99 / scale)

    # Voxel grid coords in [-1,1]^3
    lin = np.linspace(-1.0, 1.0, grid_size)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    voxel_coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Sample surface points and build KD-tree
    # (Older trimesh versions don't accept rng; keep clean/simple.)
    _ = rng  # best-effort placeholder; avoids unused-arg lint in some setups
    surface_samples = mesh.sample(int(n_surface_samples))

    tree = cKDTree(surface_samples)
    dists, _ = tree.query(voxel_coords, k=1)

    density = np.exp(-(dists**2) / (2.0 * sigma**2))

    if normalize:
        s = float(density.sum())
        if s > eps:
            density = density / s

    return density


def get_density_axes(
    flat_densities: np.ndarray,
    *,
    grid_size: int = 32,
    smallest: bool = True,
    return_eigs: bool = False,
    eps: float = 1e-12,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute principal axes for a batch of 3D densities on a [-1,1]^3 grid.

    For each density rho, compute weighted covariance (inertia):
        M = Σ rho(x) (x - μ)(x - μ)^T
    and return either:
      - smallest-eigenvalue direction (least spread), or
      - largest-eigenvalue direction (most spread).

    Returns
    -------
    directions : (N,3)
    ratios : (N,), optional
        If smallest: (λ_min / λ_mid), else: (λ_max / λ_mid) (with small stabilizer).
    """
    flat_densities = np.asarray(flat_densities, dtype=float)
    if flat_densities.ndim != 2:
        raise ValueError(f"flat_densities must be 2D. Got shape {flat_densities.shape}.")

    N, D = flat_densities.shape
    expected = grid_size**3
    if D != expected:
        raise ValueError(
            f"Mismatch: got D={D}, expected grid_size^3={expected} for grid_size={grid_size}."
        )

    # Coordinates in [-1,1]^3 (D,3)
    lin = np.linspace(-1.0, 1.0, grid_size)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    directions = np.zeros((N, 3), dtype=float)
    ratios = np.zeros(N, dtype=float)

    stab = 1e-6
    for i in range(N):
        rho = flat_densities[i]
        s = float(rho.sum())
        if s <= eps:
            directions[i] = np.array([1.0, 0.0, 0.0])
            ratios[i] = 0.0
            continue

        rho = rho / s
        mu = (coords * rho[:, None]).sum(axis=0)
        centered = coords - mu

        M = (centered.T * rho) @ centered  # (3,3)
        eigvals, eigvecs = np.linalg.eigh(M)  # ascending

        if smallest:
            directions[i] = eigvecs[:, 0]
            ratios[i] = (eigvals[0] + stab) / (eigvals[1] + stab)
        else:
            directions[i] = eigvecs[:, -1]
            ratios[i] = (eigvals[-1] + stab) / (eigvals[1] + stab)

    return (directions, ratios) if return_eigs else directions


def rotate_density(
    density: np.ndarray,
    rotations: np.ndarray,
    *,
    grid_size: int = 32,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """
    Apply 3D rotations to a density grid using interpolation (scipy.ndimage.map_coordinates).

    Parameters
    ----------
    density : (grid_size^3,) or (grid_size,grid_size,grid_size)
    rotations : (k,9) or (k,3,3)
        Rotation matrices (SO(3) or O(3)).
    grid_size : int
    order : int
        Interpolation order (1 = trilinear).
    mode, cval :
        Passed to map_coordinates.

    Returns
    -------
    rotated_densities : (k, grid_size^3)
        Flattened rotated densities.
    """
    dens = np.asarray(density, dtype=float)
    if dens.ndim == 1:
        if dens.size != grid_size**3:
            raise ValueError(
                f"density size mismatch: got {dens.size}, expected {grid_size**3}."
            )
        dens = dens.reshape((grid_size, grid_size, grid_size))
    elif dens.shape != (grid_size, grid_size, grid_size):
        raise ValueError(
            f"density must be shape ({grid_size},{grid_size},{grid_size}) "
            f"or flat length {grid_size**3}."
        )

    rots = np.asarray(rotations, dtype=float)
    if rots.ndim == 2 and rots.shape[1] == 9:
        rots = rots.reshape(-1, 3, 3)
    elif rots.ndim == 3 and rots.shape[1:] == (3, 3):
        pass
    else:
        raise ValueError(f"rotations must be (k,9) or (k,3,3). Got shape {rots.shape}.")

    k = rots.shape[0]
    center = (grid_size - 1) / 2.0

    # Base grid coords (3, N) centered at 0 in index coordinates
    x, y, z = np.meshgrid(
        np.arange(grid_size) - center,
        np.arange(grid_size) - center,
        np.arange(grid_size) - center,
        indexing="ij",
    )
    coords = np.stack([x, y, z], axis=0).reshape(3, -1)  # (3, N)

    out = np.empty((k, grid_size**3), dtype=float)
    for i in range(k):
        Rm = rots[i]
        # sample dens at points corresponding to inverse transform:
        rotated_coords = (Rm.T @ coords) + center  # (3, N) in index space
        out[i] = map_coordinates(dens, rotated_coords, order=order, mode=mode, cval=cval)

    return out


def get_mesh_sample(mesh: trimesh.Trimesh, O3_data: np.ndarray) -> np.ndarray:
    """
    Apply each 3x3 matrix in O3_data to mesh vertices and return flattened vertex arrays.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    O3_data : (n_samples, 9)
        Flattened 3x3 matrices (SO(3) or O(3)).

    Returns
    -------
    mesh_samples : (n_samples, 3*N)
        Each row is rotated vertices flattened in (x0,y0,z0,x1,y1,z1,...) order.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (N,3)
    O3_data = np.asarray(O3_data, dtype=np.float64)

    if O3_data.ndim != 2 or O3_data.shape[1] != 9:
        raise ValueError(f"Expected O3_data with shape (n, 9). Got {O3_data.shape}.")

    rot_mats = O3_data.reshape(-1, 3, 3)  # (n,3,3)

    # rotated[i, v, :] = verts[v, :] @ rot_mats[i].T
    rotated = np.einsum("vj, nij -> nvi", verts, rot_mats)  # (n,N,3)

    return rotated.reshape(rotated.shape[0], -1)
