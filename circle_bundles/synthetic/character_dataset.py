# synthetic/character_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import trimesh

from .densities import mesh_to_density, rotate_density, get_mesh_sample


# ----------------------------
# Helpers: normalization
# ----------------------------

def normalize_mesh_unitball(
    mesh: trimesh.Trimesh,
    *,
    scale_to: float = 0.99,
    eps: float = 1e-12,
) -> trimesh.Trimesh:
    """
    Copy mesh, translate to center of mass, scale so max vertex norm = scale_to.

    This matches the normalization inside mesh_to_density, so that:
      - vertices you rotate live in the same coordinate system as densities
      - densities correspond to coordinates in [-1,1]^3
    """
    m = mesh.copy()
    m.apply_translation(-m.center_mass)

    scale = float(np.max(np.linalg.norm(m.vertices, axis=1)))
    if scale <= float(eps):
        raise ValueError("Mesh appears degenerate (near-zero scale).")
    m.apply_scale(float(scale_to) / scale)
    return m


# ----------------------------
# Rotations
# ----------------------------

def sample_random_so3(
    n: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Uniform random SO(3) via random unit quaternions.

    Returns
    -------
    R : (n, 9) flattened row-major 3x3 rotation matrices
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(n)
    if n <= 0:
        raise ValueError(f"n must be positive. Got {n}.")

    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)

    # Shoemake (1992) uniform quaternions
    q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

    # quaternion (x,y,z,w) -> rotation matrix
    x, y, z, w = q1, q2, q3, q4
    R = np.empty((n, 3, 3), dtype=np.float64)

    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - wz)
    R[:, 0, 2] = 2*(xz + wy)

    R[:, 1, 0] = 2*(xy + wz)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - wx)

    R[:, 2, 0] = 2*(xz - wy)
    R[:, 2, 1] = 2*(yz + wx)
    R[:, 2, 2] = 1 - 2*(xx + yy)

    return R.reshape(n, 9)


# ----------------------------
# Dataset container
# ----------------------------

@dataclass
class CharacterRotationDataset:
    """
    A dataset built from a single template mesh under a set of rotations.

    Attributes
    ----------
    template_mesh : trimesh.Trimesh
        The *normalized* canonical mesh (with faces + visuals).
    rotations : (n,9) ndarray
        Flattened 3x3 rotation matrices used.
    X_verts : (n, 3*V) ndarray
        Flattened rotated vertices.
    X_dens : (n, grid^3) ndarray
        Flattened rotated densities.
    meta : dict
        Any extras you want (paths, params, etc.)
    """
    template_mesh: trimesh.Trimesh
    rotations: np.ndarray
    X_verts: np.ndarray
    X_dens: np.ndarray
    meta: Dict[str, Any]


# ----------------------------
# Main builder
# ----------------------------

def build_character_rotation_dataset(
    mesh_path: str,
    *,
    n_samples: int = 256,
    grid_size: int = 32,
    sigma: float = 0.05,
    n_surface_samples: int = 5000,
    rotations: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    density_interp_order: int = 1,
) -> CharacterRotationDataset:
    """
    Load a character mesh (GLB/OBJ/PLY/etc), normalize it, then build:
      - rotated vertex vectors
      - rotated densities

    Strategy
    --------
    - Compute a single base density from the normalized mesh.
    - Rotate the density grid using rotate_density (fast, consistent).
    - Rotate vertices directly for the vertex-vector dataset.

    Notes on materials/textures
    ---------------------------
    trimesh will load many formats (GLB is great). In Python rendering,
    visuals depend on your environment. The dataset itself keeps the
    original visual info in template_mesh.visual for later rendering.
    """
    rng = np.random.default_rng() if rng is None else rng

    loaded = trimesh.load(mesh_path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        # Some formats load as Scene; merge geometry if needed
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(
                [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            )
        else:
            raise TypeError(f"Unsupported loaded type: {type(loaded)}")

    # Canonical normalized mesh (matches mesh_to_density normalization)
    template = normalize_mesh_unitball(loaded)

    # Rotations
    if rotations is None:
        rotations = sample_random_so3(n_samples, rng=rng)
    else:
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.ndim == 3 and rotations.shape[1:] == (3, 3):
            rotations = rotations.reshape(-1, 9)
        if rotations.ndim != 2 or rotations.shape[1] != 9:
            raise ValueError(f"rotations must be (n,9) or (n,3,3). Got {rotations.shape}.")

    # Vertex vectors
    X_verts = get_mesh_sample(template, rotations)  # (n, 3*V)

    # Base density from canonical mesh
    base_density = mesh_to_density(
        template,
        grid_size=grid_size,
        sigma=sigma,
        n_surface_samples=n_surface_samples,
        normalize=True,
        rng=rng,
    )

    # Rotated densities by interpolation
    X_dens = rotate_density(
        base_density,
        rotations,
        grid_size=grid_size,
        order=int(density_interp_order),
        mode="constant",
        cval=0.0,
    )

    meta = dict(
        mesh_path=mesh_path,
        n_samples=int(rotations.shape[0]),
        grid_size=int(grid_size),
        sigma=float(sigma),
        n_surface_samples=int(n_surface_samples),
        density_interp_order=int(density_interp_order),
        n_vertices=int(template.vertices.shape[0]),
        n_faces=int(template.faces.shape[0]),
    )

    return CharacterRotationDataset(
        template_mesh=template,
        rotations=rotations,
        X_verts=X_verts,
        X_dens=X_dens,
        meta=meta,
    )


# ----------------------------
# Visualizer
# ----------------------------

def make_character_visualizer(
    template_mesh: trimesh.Trimesh,
    *,
    background: Tuple[int, int, int, int] = (255, 255, 255, 255),
):
    """
    Return a function viz(x_flat) that renders the character with vertices set by x_flat.

    - Keeps faces + visual/material info from template_mesh.
    - Replaces only vertex positions.
    - Returns a trimesh.Scene so you can .show() or render to image depending on your setup.
    """
    template = template_mesh.copy()
    faces = np.asarray(template.faces, dtype=np.int64)
    visual = template.visual.copy()

    V = int(template.vertices.shape[0])

    def viz(x_flat: np.ndarray) -> trimesh.Scene:
        x = np.asarray(x_flat, dtype=np.float64).reshape(-1)
        if x.size != 3 * V:
            raise ValueError(f"Expected flat vertex vector of length {3*V}. Got {x.size}.")
        verts = x.reshape(V, 3)

        m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        m.visual = visual.copy()

        scene = trimesh.Scene(m)
        try:
            # background if supported by viewer
            scene.background = np.array(background, dtype=np.uint8)
        except Exception:
            pass
        return scene

    return viz
