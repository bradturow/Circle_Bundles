from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .image_utils import render_to_rgba

__all__ = ["fiber_vis"]


def fiber_vis(
    data: np.ndarray,
    vis_func: Callable,
    *,
    vis_data: Optional[np.ndarray] = None,
    selected_indices: Optional[Sequence[int]] = None,
    max_images: int = 12,
    zoom: float = 0.2,
    figsize=(10, 8),
    save_path: Optional[str] = None,
    random_state: Optional[int] = None,
):
    """
    Visualize up to `max_images` items from `data` by embedding them to 3D (PCA)
    and overlaying thumbnails rendered by `vis_func`.

    Parameters
    ----------
    data : (N, d) array
    vis_func : callable(datum) -> (Figure | ndarray image)
        Something render_to_rgba can handle after vis_func returns.
    vis_data : optional data source for vis_func (same length N)
        If provided, thumbnails are rendered from vis_data[idx] while embedding uses data[idx].
    selected_indices : optional explicit indices to visualize
    max_images : cap on number of thumbnails
    random_state : RNG seed used when selected_indices is None
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
            raise ValueError(f"vis_data must have same length as data. Got {vis_data.shape[0]} vs {N}.")

    max_images = int(max_images)
    if max_images <= 0:
        raise ValueError("max_images must be positive.")

    # ---- choose indices ----
    if selected_indices is None:
        rng = np.random.default_rng(random_state)
        k = min(max_images, N)
        selected_indices = np.sort(rng.choice(np.arange(N, dtype=int), size=k, replace=False)).tolist()
    else:
        selected_indices = [int(i) for i in selected_indices][: min(max_images, len(selected_indices))]
        selected_indices = [i for i in selected_indices if 0 <= i < N]
        if len(selected_indices) == 0:
            raise ValueError("selected_indices produced no valid indices in range.")

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

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # light scatter so the space is visible (but not noisy)
    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], alpha=0.15, s=10)

    fig.canvas.draw()

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

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
