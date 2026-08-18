from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .image_utils import render_to_rgba

__all__ = ["fiber_vis"]


def fiber_vis(
    data: np.ndarray,
    vis_func: Optional[Callable] = None,
    *,
    vis_data: Optional[np.ndarray] = None,
    selected_indices: Optional[Sequence[int]] = None,
    max_images: int = 12,
    zoom: float = 0.2,
    figsize=(10, 8),
    dpi: int = 150,
    save_path: Optional[str] = None,
    random_state: Optional[int] = None,
    ax=None,
    clear_ax: bool = True,
    show: bool = True,
    scatter_alpha: float = 0.15,
    scatter_s: float = 10.0,
):
    """
    Visualize up to ``max_images`` items from ``data`` by embedding them to 3D with PCA.

    If ``vis_func`` is provided, the plot includes thumbnails rendered by that
    function. Otherwise, it shows only the embedded points.

    Parameters
    ----------
    data : ndarray, shape (N, d)
        Data embedded for the scatter plot.
    vis_func : callable, optional
        Callable returning a Matplotlib figure or image array that
        :func:`render_to_rgba` can handle.
        If None, no thumbnails are rendered; points only.
    vis_data : ndarray, optional
        Alternate thumbnail data with the same length as ``data``.
    selected_indices : sequence of int, optional
        Explicit indices to visualize.
    max_images : int
        Maximum number of points and thumbnails to show.
    zoom : float
        Thumbnail zoom factor.
    figsize : tuple
        Figure size when creating a new figure.
    dpi : int
        Figure resolution.
    save_path : str, optional
        Destination for the rendered figure.
    random_state : int, optional
        Random seed used when ``selected_indices`` is None.
    ax : matplotlib axis, optional
        Existing 3D axis to draw into.
    clear_ax : bool
        Whether to clear an existing axis before drawing.
    show : bool
        Whether to display the figure.
    scatter_alpha : float
        Scatter-point opacity.
    scatter_s : float
        Scatter-point size.

    Subplot usage
    -------------
    If ``ax`` is provided, it must be a 3D axis (``projection='3d'``). In that
    case, this function draws into ``ax`` and does not create a new figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from mpl_toolkits.mplot3d.proj3d import proj_transform

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (N,d). Got shape {data.shape}.")

    N = int(data.shape[0])
    if N == 0:
        raise ValueError("Empty data array.")

    if vis_data is not None:
        vis_data = np.asarray(vis_data)
        if vis_data.shape[0] != N:
            raise ValueError(
                f"vis_data must have same length as data. Got {vis_data.shape[0]} vs {N}."
            )

    max_images = int(max_images)
    if max_images <= 0:
        raise ValueError("max_images must be positive.")

    # ---- choose indices ----
    if selected_indices is None:
        rng = np.random.default_rng(random_state)
        k = min(max_images, N)
        selected_indices = np.sort(
            rng.choice(np.arange(N, dtype=int), size=k, replace=False)
        ).tolist()
    else:
        selected_indices = [int(i) for i in selected_indices][: min(max_images, len(selected_indices))]
        selected_indices = [i for i in selected_indices if 0 <= i < N]
        if len(selected_indices) == 0:
            raise ValueError("selected_indices produced no valid indices in range.")

    selected_indices = list(selected_indices)
    selected_data = data[np.asarray(selected_indices, dtype=int)]

    # ---- embed (PCA) ----
    # Lazy import so viz submodule doesn't require sklearn at import time.
    from sklearn.decomposition import PCA

    if selected_data.shape[0] == 1:
        embedded = np.zeros((1, 3), dtype=float)
    else:
        n_comp = 3 if selected_data.shape[1] >= 3 else min(3, selected_data.shape[1])
        pca = PCA(n_components=n_comp)
        emb = pca.fit_transform(selected_data)
        if emb.shape[1] < 3:
            embedded = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])), mode="constant")
        else:
            embedded = emb[:, :3]

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

    # ---- always show points ----
    ax.scatter(
        embedded[:, 0],
        embedded[:, 1],
        embedded[:, 2],
        alpha=float(scatter_alpha),
        s=float(scatter_s),
    )

    # If no vis_func supplied, we're done (aside from aesthetics/save/show)
    if vis_func is not None:
        # Ensure transforms/projection are ready
        fig.canvas.draw()

        # ---- overlays ----
        for i, (x, y, z) in enumerate(embedded):
            idx = int(selected_indices[i])
            try:
                datum = vis_data[idx] if vis_data is not None else selected_data[i]
                rendered = vis_func(datum)
                img = render_to_rgba(rendered, transparent_border=True, trim=True)

                x2, y2, _ = proj_transform(float(x), float(y), float(z), ax.get_proj())
                ab = AnnotationBbox(
                    OffsetImage(img, zoom=float(zoom)),
                    (x2, y2),
                    xycoords="data",
                    frameon=False,
                )
                ax.add_artist(ab)
            except Exception as e:
                print(f"Error rendering image at index {idx}: {type(e).__name__}: {e}")

    # ---- aesthetics ----
    ax.grid(True)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    # Only do layout/show if we created the figure
    if created_fig:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
    else:
        # In subplot mode, still honor save_path (saves the full figure)
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
