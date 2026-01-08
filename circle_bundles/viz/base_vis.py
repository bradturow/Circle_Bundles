from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def base_vis(
    base_data: np.ndarray,
    center_index: int,
    radius: float,
    dist_mat: np.ndarray,
    *,
    figsize: Tuple[float, float] = (8, 6),
    use_pca: bool = True,
    save_path: Optional[str] = None,
):
    """
    Static 3D visualization of base points, highlighting:
      - the center point (red)
      - all points within radius (blue)
      - all other points (light gray)

    Parameters
    ----------
    base_data : (N,D)
    center_index : int
    radius : float
    dist_mat : (N,N)
    use_pca : bool
        If True, PCA embeds to 3D. Else, uses first 3 coords (with padding if D<3).

    Returns
    -------
    fig, ax
    """
    base_data = np.asarray(base_data)
    dist_mat = np.asarray(dist_mat)
    N = base_data.shape[0]

    if dist_mat.shape != (N, N):
        raise ValueError(f"dist_mat must be shape (N,N). Got {dist_mat.shape} with N={N}.")
    if not (0 <= int(center_index) < N):
        raise ValueError(f"center_index out of range. Got {center_index} for N={N}.")

    if use_pca:
        pca = PCA(n_components=3)
        base_embedded = pca.fit_transform(base_data)
    else:
        d = base_data.shape[1]
        if d < 3:
            base_embedded = np.pad(base_data, ((0, 0), (0, 3 - d)), mode="constant")
        else:
            base_embedded = base_data[:, :3]

    neighbor_mask = dist_mat[int(center_index)] < float(radius)
    neighbor_mask[int(center_index)] = False

    all_indices = np.arange(N)
    neighbor_indices = np.where(neighbor_mask)[0]
    other_indices = np.setdiff1d(all_indices, np.append(neighbor_indices, int(center_index)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # others
    ax.scatter(
        base_embedded[other_indices, 0],
        base_embedded[other_indices, 1],
        base_embedded[other_indices, 2],
        color="lightgray",
        s=10,
        alpha=0.5,
        label="Base Points",
        zorder=1,
    )

    # neighbors
    ax.scatter(
        base_embedded[neighbor_indices, 0],
        base_embedded[neighbor_indices, 1],
        base_embedded[neighbor_indices, 2],
        color="blue",
        s=25,
        alpha=0.8,
        label="Neighbors (r)",
        zorder=2,
    )

    # center
    ax.scatter(
        *base_embedded[int(center_index)],
        color="red",
        s=60,
        alpha=1.0,
        label="Center",
        zorder=3,
        edgecolor="black",
        linewidth=1.2,
    )

    # equal-ish aspect box (same idea as your version)
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
