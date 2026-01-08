# synthetic/step_edges.py
from __future__ import annotations

from typing import List, Sequence, Tuple, Optional

import numpy as np

from optical_flow.contrast import get_contrast_norms


__all__ = [
    "get_patch_types_list",
    "make_step_edges",
    "make_all_step_edges",
    "sample_binary_step_edges",
    "mean_center",
    "sample_step_edge_torus",
]


def get_patch_types_list() -> List[List[List[int]]]:
    """
    Generate the 28 possible filament patch types (legacy convention).

    Each patch type is a list of [i,j] pixel coordinates (0..2) where a vector arrow lives.
    For a 3x3 patch, there are 28 "filament" patterns; we later include sign flips to get 56.
    """
    corners = [[0, 0], [2, 0], [0, 2], [2, 2]]

    ones_list: List[List[List[int]]] = []
    twos_list: List[List[List[int]]] = []
    threes_list: List[List[List[int]]] = []
    fours_list: List[List[List[int]]] = []

    for k in range(4):
        ones_list.append([corners[k]])
        twos_list.append([corners[k], [corners[k][0], 1]])
        twos_list.append([corners[k], [1, corners[k][1]]])
        threes_list.append([corners[k], [corners[k][0], 1], [1, corners[k][1]]])

        for j in range(k):
            if corners[j][0] == corners[k][0]:
                new_list = [corners[j], corners[k], [corners[j][0], 1]]
                threes_list.append(new_list)

                nl = new_list.copy()
                nl.append([1, 0])
                fours_list.append(nl)

                nl = new_list.copy()
                nl.append([1, 2])
                fours_list.append(nl)

            elif corners[j][1] == corners[k][1]:
                new_list = [corners[j], corners[k], [1, corners[j][1]]]
                threes_list.append(new_list)

                nl = new_list.copy()
                nl.append([0, 1])
                fours_list.append(nl)

                nl = new_list.copy()
                nl.append([2, 1])
                fours_list.append(nl)

    return ones_list + twos_list + threes_list + fours_list


def make_step_edges(
    n_patches: int,
    spots: Sequence[Sequence[int]],
    *,
    angle_range: Tuple[float, float] = (0.0, 2.0 * np.pi),
    normalize: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate optical-flow step edge patches as flattened vectors of length 18 (3x3x2).

    Conventions
    ----------
    - Patch array shaped (n, 3, 3, 2) with last axis = (u,v).
    - Flattening order is 'F' to match legacy notebooks.

    Parameters
    ----------
    n_patches : int
        Number of patches to generate.
    spots : sequence of (i,j)
        Pixel coordinates where the flow vector is placed.
    angle_range : (float, float)
        Sample directions uniformly from [a_min, a_max].
    normalize : bool
        If True, contrast-normalize using get_contrast_norms and mean-center each channel.
    rng : np.random.Generator, optional
        Random generator for reproducibility.

    Returns
    -------
    patch_vectors : (n_patches, 18)
    angles : (n_patches,)
    """
    if n_patches <= 0:
        raise ValueError(f"n_patches must be positive. Got {n_patches}.")

    rng = np.random.default_rng() if rng is None else rng

    a0, a1 = float(angle_range[0]), float(angle_range[1])
    angles = a0 + (a1 - a0) * rng.random(n_patches)
    long_vecs = np.column_stack([np.cos(angles), np.sin(angles)])  # (n,2)

    patch_array = np.zeros((n_patches, 3, 3, 2), dtype=float)
    for (i, j) in spots:
        ii, jj = int(i), int(j)
        if not (0 <= ii < 3 and 0 <= jj < 3):
            raise ValueError(f"Spot {(i, j)} out of bounds for 3x3 patch.")
        patch_array[:, ii, jj, :] = long_vecs

    # Legacy flattening convention
    patch_vectors = patch_array.reshape(n_patches, -1, order="F")

    if normalize:
        norms = get_contrast_norms(patch_vectors)
        norms = np.maximum(norms, 1e-12)
        patch_vectors = patch_vectors / norms[:, None]

        # Mean-center each channel separately (first 9 = u, last 9 = v)
        patch_vectors[:, :9] -= patch_vectors[:, :9].mean(axis=1, keepdims=True)
        patch_vectors[:, 9:] -= patch_vectors[:, 9:].mean(axis=1, keepdims=True)

    return patch_vectors, angles


def make_all_step_edges(angle: Optional[float] = None, *, normalize: bool = True) -> np.ndarray:
    """
    Return the 56 canonical step-edge patterns (28 types + sign flips).

    If angle is None:
        Returns scalar +/-1 edge patterns as (56, 9) using legacy 'F' convention.
    If angle is not None:
        Returns flow vectors (56, 18) at that fixed direction (cos(angle), sin(angle)).
    """
    types_list = get_patch_types_list()

    if angle is None:
        patch_array = -np.ones((56, 3, 3), dtype=float)
        for j in range(28):
            for (i, k) in types_list[j]:
                patch_array[j, int(i), int(k)] = 1.0
        patch_array[28:] = -patch_array[:28]
        return patch_array.reshape(56, -1, order="F")

    patch_vectors = np.zeros((56, 18), dtype=float)
    ang = float(angle)
    for t, spots in enumerate(types_list):
        patch_vectors[t] = make_step_edges(
            1,
            spots,
            angle_range=(ang, ang),
            normalize=normalize,
        )[0][0]
    patch_vectors[28:] = -patch_vectors[:28]
    return patch_vectors


def sample_binary_step_edges(
    samples_per_filament: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample step-edge patches across all 28 filament types.

    Returns
    -------
    patches : (28*samples_per_filament, 18)
    angles : (28*samples_per_filament,)
    """
    if samples_per_filament <= 0:
        raise ValueError(f"samples_per_filament must be positive. Got {samples_per_filament}.")

    rng = np.random.default_rng() if rng is None else rng

    types_list = get_patch_types_list()
    patches_list = []
    angles_list = []
    for spots in types_list:
        pv, ang = make_step_edges(samples_per_filament, spots, rng=rng)
        patches_list.append(pv)
        angles_list.append(ang)

    patches = np.concatenate(patches_list, axis=0)
    angles = np.concatenate(angles_list, axis=0)
    return patches, angles


def mean_center(patch_vector: np.ndarray, *, copy: bool = True) -> np.ndarray:
    """
    Mean-center a single patch vector (length 9 or 18).

    For length 18 (flow):
      - mean-center u-channel entries (first 9)
      - mean-center v-channel entries (last 9)
    """
    x = patch_vector.copy() if copy else patch_vector
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"patch_vector must be 1D. Got shape {x.shape}.")
    if x.shape[0] not in (9, 18):
        raise ValueError(f"patch_vector must have length 9 or 18. Got {x.shape[0]}.")

    x[:9] -= x[:9].mean()
    if x.shape[0] == 18:
        x[9:] -= x[9:].mean()
    return x


def sample_step_edge_torus(
    n_samples: int,
    *,
    d: int = 3,
    m: int = 10,
    thresh: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate step-edge torus samples.

    Construction (legacy):
      - sample half-planes parameterized by (offset r, direction phi) inside a disk
      - build range patches by averaging sign over m×m subgrid per pixel
      - create flow patches by multiplying by (cos theta, sin theta)
      - keep high-contrast patches (contrast norm > thresh)
      - mean-center & contrast normalize

    Parameters
    ----------
    n_samples : int
        Number of candidate samples before thresholding.
    d : int
        Patch width/height (typically 3).
    m : int
        Subgrid resolution per pixel for averaging.
    thresh : float
        Keep only samples with contrast norm > thresh.
    rng : np.random.Generator, optional

    Returns
    -------
    flow_patches : (N_kept, 2*d^2)
    coords : (N_kept, 3) where columns are (offset, phi, theta)
    range_patches : (N_kept, d^2)
    norms : (N_kept,) contrast norms (pre-normalization)
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive. Got {n_samples}.")
    if d <= 0:
        raise ValueError(f"d must be positive. Got {d}.")
    if m <= 1:
        raise ValueError(f"m must be >= 2. Got {m}.")
    if thresh < 0:
        raise ValueError(f"thresh must be >= 0. Got {thresh}.")

    rng = np.random.default_rng() if rng is None else rng

    range_patches = np.zeros((n_samples, d * d), dtype=float)

    # sample points in disk radius sqrt(2)
    Rdisc = np.sqrt(2.0)
    coords = np.zeros((n_samples, 2), dtype=float)
    k = 0
    while k < n_samples:
        p = Rdisc * (2.0 * rng.random(2) - 1.0)
        r = np.linalg.norm(p)
        if r < Rdisc:
            coords[k, 0] = r * rng.choice([-1.0, 1.0])  # signed offset
            coords[k, 1] = np.mod(np.arctan2(p[1], p[0]), 2.0 * np.pi)  # phi
            k += 1

    # build range patch per sample
    for idx, (offset, phi) in enumerate(coords):
        n_vec = np.array([np.cos(phi), np.sin(phi)], dtype=float)

        for ii in range(d):
            x_vals = -1.0 + 2.0 * ii / d + 2.0 / (d * (m - 1)) * np.arange(m)
            for jj in range(d):
                y_vals = -1.0 + 2.0 * jj / d + 2.0 / (d * (m - 1)) * np.arange(m)
                a, b = np.meshgrid(x_vals, y_vals)
                pts = np.column_stack([a.ravel(), b.ravel()])
                inds = (pts @ n_vec) >= offset
                avg_val = (2.0 * np.sum(inds) - (m * m)) / (m * m)
                range_patches[idx, d * ii + jj] = avg_val

    # flow patches
    theta = 2.0 * np.pi * rng.random(n_samples)
    coords3 = np.column_stack([coords, theta])

    flow_patches = np.zeros((n_samples, 2 * d * d), dtype=float)
    flow_patches[:, : d * d] = (np.cos(theta)[:, None] * range_patches)
    flow_patches[:, d * d :] = (np.sin(theta)[:, None] * range_patches)

    norms = get_contrast_norms(flow_patches)
    keep = norms > float(thresh)

    flow_kept = flow_patches[keep]
    range_kept = range_patches[keep]
    coords_kept = coords3[keep]
    norms_kept = norms[keep]

    # mean-center and contrast normalize
    out = np.empty_like(flow_kept)
    for i in range(flow_kept.shape[0]):
        x = mean_center(flow_kept[i], copy=True)
        out[i] = x / max(float(norms_kept[i]), 1e-12)

    return out, coords_kept, range_kept, norms_kept
