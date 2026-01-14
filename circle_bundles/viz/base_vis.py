from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["base_vis"]


def base_vis(
    base_data: np.ndarray,
    center_index: int,
    radius: float,
    dist_mat: np.ndarray,
    *,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
    use_pca: bool = True,
    save_path: Optional[str] = None,
    # NEW:
    ax=None,
    clear_ax: bool = True,
    show: bool = True,
    # Optional styling knobs (handy but not required)
    other_alpha: float = 0.5,
    other_s: float = 10.0,
    neigh_alpha: float = 0.8,
    neigh_s: float = 25.0,
    center_s: float = 60.0,
):
    """
    Static 3D visualization of base points, highlighting:
      - the center point (red)
      - all points within radius (blue)
      - all other points (light gray)

    Subplot usage
    -------------
    If `ax` is provided, it must be a 3D axis (projection='3d'). The function
    draws into that axis and will not create a new figure.
    """
    import matplotlib.pyplot as plt

    base_data = np.asarray(base_data)
    dist_mat = np.asarray(dist_mat)
    N = int(base_data.shape[0])

    if dist_mat.shape != (N, N):
        raise ValueError(f"dist_mat must be shape (N,N). Got {dist_mat.shape} with N={N}.")
    if not (0 <= int(center_index) < N):
        raise ValueError(f"center_index out of range. Got {center_index} for N={N}.")

    # ---- embed to 3D ----
    if use_pca:
        # lazy import
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        base_embedded = pca.fit_transform(base_data)
    else:
        d = int(base_data.shape[1])
        if d < 3:
            base_embedded = np.pad(base_data, ((0, 0), (0, 3 - d)), mode="constant")
        else:
            base_embedded = base_data[:, :3]

    # ---- neighbor sets ----
    neighbor_mask = dist_mat[int(center_index)] < float(radius)
    neighbor_mask[int(center_index)] = False

    all_indices = np.arange(N, dtype=int)
    neighbor_indices = np.where(neighbor_mask)[0].astype(int)
    other_indices = np.setdiff1d(all_indices, np.append(neighbor_indices, int(center_index)))

    # ---- figure / axes ----
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=int(dpi))
        ax = fig.add_subplot(111, projection="3d")
        created_fig = True
    else:
        fig = ax.figure
        ax_is_3d = getattr(ax, "name", "") == "3d"
        if not ax_is_3d:
            raise ValueError("Provided ax is not a 3D axis. Create it with projection='3d'.")

    if clear_ax:
        ax.cla()

    # ---- draw ----
    ax.scatter(
        base_embedded[other_indices, 0],
        base_embedded[other_indices, 1],
        base_embedded[other_indices, 2],
        color="lightgray",
        s=float(other_s),
        alpha=float(other_alpha),
        label="Base Points",
        zorder=1,
    )

    ax.scatter(
        base_embedded[neighbor_indices, 0],
        base_embedded[neighbor_indices, 1],
        base_embedded[neighbor_indices, 2],
        color="blue",
        s=float(neigh_s),
        alpha=float(neigh_alpha),
        label="Neighbors (r)",
        zorder=2,
    )

    ax.scatter(
        *base_embedded[int(center_index)],
        color="red",
        s=float(center_s),
        alpha=1.0,
        label="Center",
        zorder=3,
        edgecolor="black",
        linewidth=1.2,
    )

    # ---- approx equal aspect ----
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = float(np.mean(x_limits))
    y_middle = float(np.mean(y_limits))
    z_middle = float(np.mean(z_limits))

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


    ax.grid(True)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    # ---- save / show ----
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if created_fig:
        plt.tight_layout()
        if show:
            plt.show()

    return fig, ax
