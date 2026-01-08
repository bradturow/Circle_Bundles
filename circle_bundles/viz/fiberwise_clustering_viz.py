from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import networkx as nx

from .image_utils import fig_to_rgba, trim_image

__all__ = [
    "make_patch_cluster_diagram",
    "get_G_vertex_coords",
]


def make_patch_cluster_diagram(
    data: np.ndarray,
    clusters: np.ndarray,
    G: nx.Graph,
    vis_func: Callable[[np.ndarray], Union[np.ndarray, Figure]],
    *,
    image_zoom: float = 0.4,
    row_spacing: float = 1.5,
    col_spacing: float = 1.5,
    line_color: str = "black",
    line_width: float = 5.0,
    save_path: Optional[str] = None,
    white_thresh: int = 250,
    figsize: Optional[Tuple[float, float]] = None,
) -> tuple[Figure, plt.Axes]:
    """
    Display patch thumbnails in a grid by (j,k) with graph edges drawn behind.

    Assumptions
    -----------
    - data is (N, d)
    - clusters is (N, 2) with rows (j,k) corresponding to data rows
    - G has nodes labeled (j,k)
    - vis_func returns either:
        * a Matplotlib Figure, OR
        * an image array (H,W,3) or (H,W,4)
    """
    data = np.asarray(data)
    clusters = np.asarray(clusters)

    if clusters.ndim != 2 or clusters.shape[1] != 2:
        raise ValueError(f"clusters must be shape (N,2). Got {clusters.shape}.")
    if data.shape[0] != clusters.shape[0]:
        raise ValueError(f"data and clusters must have same N. Got {data.shape[0]} vs {clusters.shape[0]}.")

    unique_js = np.unique(clusters[:, 0].astype(int))
    unique_js.sort()

    coord_map: dict[tuple[int, int], tuple[float, float]] = {}
    for col_idx, j in enumerate(unique_js):
        ks = np.unique(clusters[clusters[:, 0] == j, 1].astype(int))
        ks.sort()
        for row_idx, k in enumerate(ks):
            coord_map[(int(j), int(k))] = (col_idx * float(col_spacing), -row_idx * float(row_spacing))

    if figsize is None:
        max_rows = max(len(np.unique(clusters[clusters[:, 0] == j, 1])) for j in unique_js) if len(unique_js) else 1
        figsize = (
            max(2.0, len(unique_js) * float(col_spacing)),
            max(2.0, max_rows * float(row_spacing)),
        )

    fig, ax = plt.subplots(figsize=figsize)

    # edges behind
    for u, v in G.edges():
        if u in coord_map and v in coord_map:
            x0, y0 = coord_map[u]
            x1, y1 = coord_map[v]
            ax.plot([x0, x1], [y0, y1], color=line_color, linewidth=float(line_width), zorder=0)

    # patches
    for idx, (j, k) in enumerate(clusters.astype(int)):
        key = (int(j), int(k))
        if key not in coord_map:
            continue
        x, y = coord_map[key]

        rendered = vis_func(data[idx])

        if isinstance(rendered, Figure):
            rgba = fig_to_rgba(rendered)
            plt.close(rendered)
        else:
            rgba = np.asarray(rendered)
            if rgba.ndim == 3 and rgba.shape[2] == 3:
                alpha = 255 * np.ones((*rgba.shape[:2], 1), dtype=rgba.dtype)
                rgba = np.concatenate([rgba, alpha], axis=2)

        rgba = trim_image(rgba, white_thresh=int(white_thresh))
        imagebox = OffsetImage(rgba, zoom=float(image_zoom))
        ab = AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0, zorder=1)
        ax.add_artist(ab)

    xs = [c[0] for c in coord_map.values()]
    ys = [c[1] for c in coord_map.values()]
    x_pad = float(col_spacing) * 0.2
    y_pad = float(row_spacing) * 0.2
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.axis("off")

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, pad_inches=0)

    return fig, ax


def get_G_vertex_coords(G: nx.Graph) -> np.ndarray:
    """
    Compute 2D coordinates for nodes labeled (j,k):
      - same j -> same x column
      - within column: y evenly spaced in [0,1]
    Returns coords in the same order as list(G.nodes()).
    """
    nodes = list(G.nodes())
    j_groups: dict[int, list[int]] = defaultdict(list)
    for (j, k) in nodes:
        j_groups[int(j)].append(int(k))

    sorted_j = sorted(j_groups.keys())
    num_cols = len(sorted_j)
    x_map = {j: (i / (num_cols - 1) if num_cols > 1 else 0.5) for i, j in enumerate(sorted_j)}

    coords = np.zeros((len(nodes), 2), dtype=float)
    node_to_i = {node: i for i, node in enumerate(nodes)}

    for j in sorted_j:
        ks = sorted(j_groups[j])
        ys = [0.5] if len(ks) == 1 else np.linspace(0.0, 1.0, len(ks))
        for y, k in zip(ys, ks):
            idx = node_to_i[(j, k)]
            coords[idx] = (float(x_map[j]), float(y))

    return coords
