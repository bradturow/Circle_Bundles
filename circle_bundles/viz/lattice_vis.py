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
    ax=None,
    clear_ax: bool = True,
):
    """
    Plot thumbnails at (scaled) coordinates in [0,1]^2 while ensuring thumbnails
    are fully visible (no clipping at borders).

    Selection:
      - Nearest neighbors to a lattice of target points (per_row x per_col)
      - No reuse of the same datum.

    Placement:
      - Each selected point is placed at its *true* (scaled) position, but mapped
        into a "safe center region" so thumbnails don't spill outside the axes.

    Notes on subplot usage:
      - If `ax` is provided, thumbnails are placed inside that axis' bounding box.
      - We overlay small inset axes positioned in *figure fraction* coordinates
        corresponding to the provided axis' rectangle.
      - `figsize`/`dpi` are only used if `ax is None`.
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
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)  # avoid division by ~0
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

    # --- Figure / axis handling ---
    created_fig = False
    if ax is None:
        if isinstance(figsize, (int, float)):
            figsize = (float(figsize), float(figsize))
        fig = plt.figure(figsize=figsize, dpi=int(dpi))
        ax = fig.add_subplot(111)
        created_fig = True
    else:
        fig = ax.figure

    if clear_ax:
        ax.cla()
    ax.axis("off")

    # We need up-to-date positions and pixel sizes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Axis bounding box in figure-fraction coordinates
    ax_bbox_fig = ax.get_position()  # Bbox in [0,1] figure fraction
    ax_left, ax_bottom, ax_w, ax_h = (
        float(ax_bbox_fig.x0),
        float(ax_bbox_fig.y0),
        float(ax_bbox_fig.width),
        float(ax_bbox_fig.height),
    )

    # Compute figure pixel dimensions from the actual figure
    fig_w_px = float(fig.bbox.width)
    fig_h_px = float(fig.bbox.height)

    # Convert desired thumbnail pixel size to figure fractions,
    # then to fractions of the axis rectangle.
    width_fig_frac = float(thumb_px) / fig_w_px
    height_fig_frac = float(thumb_px) / fig_h_px

    if width_fig_frac >= 1 or height_fig_frac >= 1:
        raise ValueError(
            "thumb_px too large relative to figure pixel size (thumbnail doesn't fit). "
            f"width_fig_frac={width_fig_frac:.3f}, height_fig_frac={height_fig_frac:.3f}."
        )

    width_ax_frac = width_fig_frac / ax_w
    height_ax_frac = height_fig_frac / ax_h

    if width_ax_frac >= 1 or height_ax_frac >= 1:
        raise ValueError(
            "thumb_px too large relative to the provided axis size. "
            f"width_ax_frac={width_ax_frac:.3f}, height_ax_frac={height_ax_frac:.3f}."
        )

    # Safe center region inside the axis (in axis-fraction coordinates)
    x0, x1 = width_ax_frac / 2, 1 - width_ax_frac / 2
    y0, y1 = height_ax_frac / 2, 1 - height_ax_frac / 2

    # Helper: convert an (u,v) in axis-fraction coordinates to figure fraction
    def _axfrac_to_figfrac(u: float, v: float) -> tuple[float, float]:
        return (ax_left + u * ax_w, ax_bottom + v * ax_h)

    # Place thumbnails (as inset axes in figure fraction coordinates)
    for idx, (cx, cy) in zip(selected_indices, selected_coords):
        u = x0 + float(cx) * (x1 - x0)  # axis-fraction x
        v = y0 + float(cy) * (y1 - y0)  # axis-fraction y

        left_fig, bottom_fig = _axfrac_to_figfrac(u, v)
        left_fig -= width_fig_frac / 2
        bottom_fig -= height_fig_frac / 2

        ax_in = fig.add_axes([left_fig, bottom_fig, width_fig_frac, height_fig_frac])
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
        # keep the whole figure; bbox_inches tight is usually fine, but can clip
        # inset axes depending on backend. If you see clipping, remove bbox_inches.
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")

    return fig, ax
