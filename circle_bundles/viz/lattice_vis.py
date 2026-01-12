from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .image_utils import render_to_rgba

__all__ = ["lattice_vis"]


def lattice_vis(
    data: Sequence,
    coords: np.ndarray,
    vis_func: Callable[[object], Union[np.ndarray, plt.Figure]],
    *,
    per_row: int = 7,
    per_col: int = 7,
    padding: float = 0.05,
    figsize: float | Tuple[float, float] = 10,
    thumb_px: int = 200,
    dpi: int = 200,
    save_path: Optional[str] = None,
    transparent_border: bool = True,
    white_thresh: int = 250,
):
    """
    Plot thumbnails at (scaled) coordinates in [0,1]^2 while ensuring thumbnails
    are fully visible (no clipping at borders).

    Selection:
      - Nearest neighbors to a lattice of target points (per_row x per_col)
      - No reuse of the same datum.

    Placement:
      - Each selected point is placed at its *true* (scaled) position, but mapped
        into a "safe center region" so thumbnails don't spill outside the figure.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be an (N,2) array. Got {coords.shape}.")

    N = int(coords.shape[0])
    if len(data) != N:
        raise ValueError(f"data length must match coords rows. Got len(data)={len(data)} vs N={N}.")
    if N == 0:
        raise ValueError("Empty coords/data.")

    per_row = int(per_row)
    per_col = int(per_col)
    if per_row <= 0 or per_col <= 0:
        raise ValueError("per_row and per_col must be positive.")

    # Normalize coords to [0,1]^2 (for selection & placement)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    denom = (max_vals - min_vals)
    # avoid division by ~0 in degenerate coordinate sets
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    scaled_coords = (coords - min_vals) / denom

    # Build lattice targets (used only for selection)
    pad = float(np.clip(padding, 0.0, 0.49))
    lin_x = np.linspace(pad, 1 - pad, per_row)
    lin_y = np.linspace(pad, 1 - pad, per_col)
    grid_x, grid_y = np.meshgrid(lin_x, lin_y, indexing="xy")
    lattice_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # Pick nearest data point to each lattice target, without reuse
    selected_indices: list[int] = []
    used: set[int] = set()

    for lp in lattice_pts:
        d = np.linalg.norm(scaled_coords - lp[None, :], axis=1)
        for idx in np.argsort(d):
            idx = int(idx)
            if idx not in used:
                selected_indices.append(idx)
                used.add(idx)
                break

    selected_coords = scaled_coords[selected_indices]

    # Create figure
    if isinstance(figsize, (int, float)):
        figsize = (float(figsize), float(figsize))
    fig = plt.figure(figsize=figsize, dpi=int(dpi))

    fig_w_px = float(figsize[0]) * float(dpi)
    fig_h_px = float(figsize[1]) * float(dpi)

    width_frac = float(thumb_px) / fig_w_px
    height_frac = float(thumb_px) / fig_h_px

    if width_frac >= 1 or height_frac >= 1:
        raise ValueError(
            "thumb_px too large relative to figsize*dpi (thumbnail doesn't fit). "
            f"width_frac={width_frac:.3f}, height_frac={height_frac:.3f}."
        )

    # Safe center region
    x0, x1 = width_frac / 2, 1 - width_frac / 2
    y0, y1 = height_frac / 2, 1 - height_frac / 2

    # Place thumbnails
    for idx, (cx, cy) in zip(selected_indices, selected_coords):
        cx_safe = x0 + float(cx) * (x1 - x0)
        cy_safe = y0 + float(cy) * (y1 - y0)

        left = cx_safe - width_frac / 2
        bottom = cy_safe - height_frac / 2

        ax_in = fig.add_axes([left, bottom, width_frac, height_frac])
        rendered = vis_func(data[idx])
        img = render_to_rgba(
            rendered,
            transparent_border=bool(transparent_border),
            trim=True,
            white_thresh=int(white_thresh),
        )
        ax_in.imshow(img, interpolation="nearest")
        ax_in.set_facecolor("none")
        ax_in.axis("off")

    if save_path is not None:
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")

    return fig
