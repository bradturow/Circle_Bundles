from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle

from .image_utils import render_to_rgba


def _circular_dist(a: np.ndarray, b: float | np.ndarray) -> np.ndarray:
    """Shortest circular distance between angles a and b (radians)."""
    return np.abs(np.angle(np.exp(1j * (a - b))))


def circle_vis(
    data,
    coords,
    vis_func,
    *,
    per_circle: int = 8,
    angle_range: Optional[Tuple[float, float]] = None,
    radius: float = 1.0,
    extent_factor: float = 1.2,
    figsize: float | Tuple[float, float] = 2.8,
    dpi: int = 150,
    zoom: float = 0.13,
    circle_linewidth: float = 1.0,
    circle_color: str = "black",
    save_path: Optional[str] = None,
):
    coords = np.asarray(coords)
    N = len(data)

    # --- Convert coords to angles ---
    if coords.ndim == 1 or (coords.ndim == 2 and coords.shape[1] == 1):
        angles = coords.reshape(-1).astype(float)
    elif coords.ndim == 2 and coords.shape[1] == 2:
        x = coords[:, 0]
        y = coords[:, 1]
        angles = np.arctan2(y, x).astype(float)
    else:
        raise ValueError("coords must be 1D angles or (N,2) circle points.")

    two_pi = 2 * np.pi
    angles = np.mod(angles, two_pi)

    # --- Angle sampling range ---
    if angle_range is None:
        a_min, a_max = 0.0, two_pi
    else:
        a_min, a_max = angle_range

    if a_max <= a_min:
        raise ValueError("angle_range must satisfy a_min < a_max.")

    k = min(int(per_circle), int(N))
    full_circle = np.isclose(a_max - a_min, two_pi)
    target_angles = np.linspace(a_min, a_max, k, endpoint=not full_circle)
    target_angles = np.mod(target_angles, two_pi)

    # --- Mutual-nearest assignment (prevents duplicates; allows blanks) ---
    proposed = []
    for ta in target_angles:
        idx = int(np.argmin(_circular_dist(angles, ta)))
        proposed.append(idx)
    proposed = np.array(proposed, dtype=int)

    accepted: list[Optional[int]] = [None] * k
    used: set[int] = set()

    for t, (ta, idx) in enumerate(zip(target_angles, proposed)):
        if idx in used:
            continue
        t_star = int(np.argmin(_circular_dist(target_angles, angles[idx])))
        if t_star == t:
            accepted[t] = idx
            used.add(idx)

    # --- Figure & axes ---
    if isinstance(figsize, (int, float)):
        figsize = (float(figsize), float(figsize))
    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))

    ax.add_patch(
        Circle(
            (0.0, 0.0),
            radius,
            fill=False,
            linewidth=float(circle_linewidth),
            edgecolor=circle_color,
        )
    )

    for ta, idx in zip(target_angles, accepted):
        if idx is None:
            continue

        x_pos = radius * np.cos(ta)
        y_pos = radius * np.sin(ta)

        rendered = vis_func(data[idx])
        img = render_to_rgba(
            rendered,
            transparent_border=True,
            trim=True,
        )

        ab = AnnotationBbox(OffsetImage(img, zoom=float(zoom)), (x_pos, y_pos), frameon=False)
        ax.add_artist(ab)

    R = radius * extent_factor
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=int(dpi), bbox_inches="tight")

    return fig, ax


def circle_vis_grid(
    datasets,
    angles_list,
    vis_func,
    *,
    titles: Optional[list[str]] = None,
    per_circle: int = 8,
    circle_radius: float = 1.0,
    extent_factor: float = 1.2,
    circle_figsize: float = 5,
    circle_dpi: int = 150,
    circle_zoom: float = 0.13,
    circle_linewidth: float = 1.0,
    circle_color: str = "black",
    n_cols: int = 3,
    title_fontsize: int = 16,
    figsize_per_panel: float = 5,
    fig_dpi: int = 150,
    save_path: Optional[str] = None,
):
    import math

    images = []
    n_components = len(datasets)

    if titles is None:
        titles = [f"Component {i}" for i in range(n_components)]

    # Build each circle_vis and capture as image
    for data, coords in zip(datasets, angles_list):
        fig_cc, _ = circle_vis(
            data,
            coords,
            vis_func,
            per_circle=per_circle,
            angle_range=None,
            radius=circle_radius,
            extent_factor=extent_factor,
            figsize=circle_figsize,
            dpi=circle_dpi,
            zoom=circle_zoom,
            circle_linewidth=circle_linewidth,
            circle_color=circle_color,
            save_path=None,
        )
        fig_cc.canvas.draw()
        w, h = fig_cc.canvas.get_width_height()
        buf = fig_cc.canvas.buffer_rgba()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[..., :3]  # RGB
        plt.close(fig_cc)
        images.append(img)

    n_plots = len(images)
    if n_plots == 0:
        return None, None

    n_cols = min(int(n_cols), n_plots)
    n_rows = int(math.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel * n_cols, figsize_per_panel * n_rows),
        dpi=int(fig_dpi),
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.ravel(axes)

    for k, img in enumerate(images):
        axes[k].imshow(img)
        axes[k].axis("off")
        axes[k].set_title(titles[k], fontsize=title_fontsize)

    for k in range(n_plots, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=int(fig_dpi), bbox_inches="tight")

    plt.show()
    return fig, axes
