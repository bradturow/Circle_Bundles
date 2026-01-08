from __future__ import annotations

from typing import Optional
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d.proj3d import proj_transform
from sklearn.decomposition import PCA

from .image_utils import render_to_rgba


def fiber_vis(
    data: np.ndarray,
    vis_func,
    *,
    vis_data=None,
    selected_indices=None,
    max_images: int = 12,
    zoom: float = 0.2,
    figsize=(10, 8),
    save_path: Optional[str] = None,
):
    data = np.asarray(data)
    N = data.shape[0]

    if selected_indices is None:
        selected_indices = sorted(random.sample(range(N), min(int(max_images), N)))
    else:
        selected_indices = np.array(selected_indices, dtype=int)[: int(max_images)].tolist()

    selected_data = data[selected_indices]

    pca = PCA(n_components=3)
    embedded = pca.fit_transform(selected_data)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], alpha=0.1)

    fig.canvas.draw()

    for i, (x, y, z) in enumerate(embedded):
        try:
            datum = vis_data[selected_indices[i]] if vis_data is not None else selected_data[i]
            rendered = vis_func(datum)
            img = render_to_rgba(rendered, transparent_border=True, trim=True)

            x2, y2, _ = proj_transform(x, y, z, ax.get_proj())
            ab = AnnotationBbox(
                OffsetImage(img, zoom=float(zoom)),
                (x2, y2),
                xycoords="data",
                frameon=False,
            )
            ax.add_artist(ab)
        except Exception as e:
            print(f"Error rendering image at index {selected_indices[i]}: {e}")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
