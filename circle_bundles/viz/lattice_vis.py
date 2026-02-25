from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union, Iterable

import numpy as np
import matplotlib.pyplot as plt

from .image_utils import render_to_rgba

__all__ = ["lattice_vis",
          "scatter_lattice_vis"]


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
    Visualize a dataset by placing rendered thumbnails at 2D coordinates, using a
    lattice-based nearest-neighbor selection to pick representative examples.

    The input coordinates are first affine-rescaled to the unit square [0, 1]^2.
    A regular lattice of target points (``per_row`` × ``per_col``) is created
    inside the unit square (inset from the border by ``padding``). For each
    lattice target, the nearest *unused* datum is selected (greedy, without
    replacement). The selected thumbnails are then placed at their true rescaled
    positions, but mapped into a "safe center region" so that each thumbnail
    remains fully visible and is not clipped by the axes boundary.

    Thumbnails are drawn by creating inset axes positioned in *figure-fraction*
    coordinates corresponding to the provided axis' rectangle. This allows
    consistent pixel-sized thumbnails (``thumb_px``) regardless of axis data
    limits.

    Parameters
    ----------
    data
        Sequence of N objects to visualize (one per coordinate).
    coords
        Array of shape (N, 2) giving 2D coordinates for each datum.
    vis_func
        Callable mapping a single datum to either an image array or a Matplotlib
        Figure. The output is converted to an RGBA image via ``render_to_rgba``.
    per_row, per_col
        Number of lattice targets in the x- and y-directions used for selecting
        representative points.
    padding
        Inset margin (in [0, 0.49]) used when constructing lattice targets in
        the rescaled coordinate system.
    figsize
        Figure size in inches if ``ax is None``. If a single number is given,
        a square figure is created.
    thumb_px
        Desired thumbnail size in pixels (square). This is enforced in figure
        pixel space; if too large relative to the figure/axis size, an error is
        raised.
    dpi
        Dots-per-inch used when creating a new figure (``ax is None``) and when
        saving to disk.
    save_path
        Optional path to save the resulting figure.
    transparent_border
        Passed to ``render_to_rgba``. If True, attempts to make background
        whitespace transparent when trimming.
    white_thresh
        Passed to ``render_to_rgba``. Pixel intensity threshold for detecting
        near-white background when trimming/making transparent.
    ax
        Optional Matplotlib axis to draw into. If provided, thumbnails are placed
        within this axis' rectangle in figure coordinates.
    clear_ax
        If True, clears ``ax`` before drawing and turns the axis off.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the visualization.
    ax : matplotlib.axes.Axes
        The base axis used as the placement region (turned off).

    Raises
    ------
    ValueError
        If ``coords`` is not shape (N, 2), if ``len(data) != N`` or ``N == 0``,
        if ``per_row``/``per_col`` are non-positive, or if ``thumb_px`` is too
        large to fit within the figure or provided axis region.

    Notes
    -----
    *Selection vs placement:* lattice targets are used only to choose which data
    points to show; thumbnails are placed at the selected points' actual rescaled
    coordinates (subject to the safe-center remapping that prevents clipping).

    The selection procedure is greedy without replacement and may not produce a
    globally optimal assignment between lattice targets and points.
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



def scatter_lattice_vis(
    data: Sequence,
    coords: np.ndarray,
    vis_func: Callable[[object], Union[np.ndarray, plt.Figure]],
    *,
    # selection
    selected_indices: Optional[Sequence[int]] = None,
    per_row: int = 7,
    per_col: int = 7,
    padding: float = 0.05,
    # appearance
    point_size: float = 3.0,
    point_alpha: float = 0.1,
    highlight_size: float = 50.0,
    highlight_lw: float = 1.5,
    thumb_px: int = 30,
    thumb_offset_px: int = 5,     # gap between dot and thumbnail (pixels)
    leader_line: bool = False,
    leader_lw: float = 0.8,
    # figure
    figsize: float | Tuple[float, float] = 10,
    dpi: int = 200,
    save_path: Optional[str] = None,
    transparent_border: bool = True,
    white_thresh: int = 250,
    ax=None,
    clear_ax: bool = True,
):
    """
    Scatter all points, highlight selected points, and draw each selected point's
    thumbnail just to the left of its dot.

    Selection:
      - If selected_indices is provided, use it.
      - Otherwise, select representatives via the same lattice-nearest-neighbor
        scheme used by lattice_vis (per_row x per_col, no reuse).

    Placement:
      - All dots are plotted at their scaled coordinates in [0,1]^2.
      - Each selected thumbnail is placed in *pixel space* using Matplotlib's
        offsetbox machinery, so the thumbnail is consistently sized and sits
        'thumb_offset_px' pixels left of the dot.

    Returns
    -------
    fig, ax, selected_indices
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be an (N,2) array. Got {coords.shape}.")

    N = int(coords.shape[0])
    if len(data) != N:
        raise ValueError(f"data length must match coords rows. Got len(data)={len(data)} vs N={N}.")
    if N == 0:
        raise ValueError("Empty coords/data.")

    # Normalize coords to [0,1]^2 for plotting (and selection)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    denom = (max_vals - min_vals)
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    scaled = (coords - min_vals) / denom

    # --- choose indices ---
    if selected_indices is None:
        per_row = int(per_row)
        per_col = int(per_col)
        if per_row <= 0 or per_col <= 0:
            raise ValueError("per_row and per_col must be positive.")

        pad = float(np.clip(padding, 0.0, 0.49))
        lin_x = np.linspace(pad, 1 - pad, per_row)
        lin_y = np.linspace(pad, 1 - pad, per_col)
        gx, gy = np.meshgrid(lin_x, lin_y, indexing="xy")
        lattice_pts = np.column_stack([gx.ravel(), gy.ravel()])

        chosen: list[int] = []
        used: set[int] = set()
        for lp in lattice_pts:
            d = np.linalg.norm(scaled - lp[None, :], axis=1)
            for idx in np.argsort(d):
                idx = int(idx)
                if idx not in used:
                    chosen.append(idx)
                    used.add(idx)
                    break
        selected_indices = chosen
    else:
        selected_indices = [int(i) for i in selected_indices]

    # --- figure/axis ---
    created_fig = False
    if ax is None:
        if isinstance(figsize, (int, float)):
            figsize = (float(figsize), float(figsize))
        fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))
        created_fig = True
    else:
        fig = ax.figure

    if clear_ax:
        ax.cla()

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # plot all points
    ax.scatter(scaled[:, 0], scaled[:, 1], s=float(point_size), alpha=float(point_alpha))

    # highlight selected points with red rings
    sel = np.asarray(selected_indices, dtype=int)
    ax.scatter(
        scaled[sel, 0],
        scaled[sel, 1],
        s=float(highlight_size),
        facecolors="none",
        edgecolors="red",
        linewidths=float(highlight_lw),
        zorder=5,
    )

    # --- place thumbnails left of each selected point ---
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    # Convert pixel thumbnail size to zoom based on image array shape.
    # We'll render to RGBA and then set zoom to hit approx thumb_px.
    for idx in selected_indices:
        rendered = vis_func(data[idx])
        img = render_to_rgba(
            rendered,
            transparent_border=bool(transparent_border),
            trim=True,
            white_thresh=int(white_thresh),
        )

        # zoom so that max dimension ~ thumb_px in display pixels
        h, w = img.shape[0], img.shape[1]
        max_hw = max(h, w)
        zoom = float(thumb_px) / float(max_hw)

        oi = OffsetImage(img, zoom=zoom)

        # place the image at an offset in pixels relative to the data point
        # left by (thumb_px/2 + thumb_offset_px), no vertical offset
        xy = (float(scaled[idx, 0]), float(scaled[idx, 1]))
        dx = -(thumb_px / 2 + float(thumb_offset_px))
        dy = 0.0

        ab = AnnotationBbox(
            oi,
            xy=xy,
            xybox=(dx, dy),
            xycoords="data",
            boxcoords="offset points",
            frameon=False,
            pad=0.0,
            arrowprops=(
                dict(arrowstyle="-", lw=float(leader_lw), alpha=0.8, color="black")
                if leader_line
                else None
            ),
            zorder=10,
        )
        ax.add_artist(ab)

    #ax.set_xlabel("Base Angle")
    #ax.set_ylabel("Fiber Angle")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_frame_on(False)    
    if save_path is not None:
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")

    if created_fig:
        plt.show()

    return fig, ax, selected_indices