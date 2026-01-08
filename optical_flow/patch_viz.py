# optical_flow/patch_viz.py
from __future__ import annotations

from typing import Callable, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches

PatchKind = Literal["intensity", "flow"]

__all__ = [
    "PatchKind",
    "make_patch_visualizer",
]


def _infer_patch_kind_and_n(length: int) -> Tuple[PatchKind, int]:
    """Infer patch kind and n from flat vector length."""
    length = int(length)

    n = int(np.sqrt(length))
    if n * n == length:
        return "intensity", n

    n2 = int(np.sqrt(length / 2))
    if 2 * n2 * n2 == length:
        return "flow", n2

    raise ValueError("Invalid patch length: must be n^2 or 2*n^2 for some integer n.")


def make_patch_visualizer(
    *,
    # figure formatting
    fig_size: float = 3.0,
    dpi: int = 120,
    # intensity settings
    intensity_cmap: str = "gray",
    intensity_normalize: Literal["minmax", "none"] = "minmax",
    show_intensity_grid: bool = True,
    intensity_grid_lw: float = 0.8,
    # flow settings
    flipud: bool = True,
    flow_normalize: Literal["maxmag", "none"] = "maxmag",
    quiver_width: float = 0.02,
    headwidth: float = 5,
    headlength: float = 5,
    grid_lw: float = 7,
    border_lw_mult: float = 2.0,
) -> Callable[[np.ndarray], plt.Figure]:
    """
    Canonical patch visualizer.

    - intensity (n^2): uses imshow, optional grid overlay
    - flow (2*n^2): quiver arrows on grid, optional vertical flip for display convention

    Returns
    -------
    vis_func(patch_vector) -> matplotlib Figure
    """
    def make_fig_ax(n: int):
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
        fig.patch.set_alpha(0.0)       # transparent around axes
        ax.set_facecolor("white")      # solid inside
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return fig, ax

    def vis_func(patch_vector: np.ndarray) -> plt.Figure:
        v = np.asarray(patch_vector, dtype=float).ravel()
        kind, n = _infer_patch_kind_and_n(v.size)

        fig, ax = make_fig_ax(n)

        if kind == "intensity":
            img = v.reshape((n, n), order="F")

            if intensity_normalize == "minmax":
                a = float(np.min(img))
                b = float(np.max(img))
                img = (img - a) / (b - a) if b > a else np.zeros_like(img)

            ax.imshow(
                img[::-1, :],           # row 0 appears at bottom
                cmap=intensity_cmap,
                origin="lower",
                extent=(0, n, 0, n),
                interpolation="nearest",
            )

            if show_intensity_grid:
                ax.set_xticks(np.arange(n + 1))
                ax.set_yticks(np.arange(n + 1))
                ax.grid(True, which="both", linewidth=float(intensity_grid_lw), color="black")
                ax.tick_params(axis="both", which="both", length=0)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

            rect = mpl_patches.Rectangle(
                (0, 0), n, n,
                linewidth=float(border_lw_mult) * float(intensity_grid_lw),
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)

        else:
            flow = v.reshape((n, n, 2), order="F")
            if flipud:
                flow = flow[::-1, :, :]

            if flow_normalize == "maxmag":
                mags = np.linalg.norm(flow, axis=2)
                m = float(np.max(mags))
                if m > 0:
                    flow = flow / m

            x, y = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5)
            x = x - flow[:, :, 0] / 2.0
            y = y - flow[:, :, 1] / 2.0

            ax.quiver(
                x, y,
                flow[:, :, 0], flow[:, :, 1],
                scale=n,
                width=float(quiver_width),
                headwidth=float(headwidth),
                headlength=float(headlength),
            )

            ax.set_xticks(np.arange(n + 1))
            ax.set_yticks(np.arange(n + 1))
            ax.grid(True, which="both", linewidth=float(grid_lw), color="black")
            ax.tick_params(axis="both", which="both", length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            rect = mpl_patches.Rectangle(
                (0, 0), n, n,
                linewidth=float(border_lw_mult) * float(grid_lw),
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)

        return fig

    return vis_func
