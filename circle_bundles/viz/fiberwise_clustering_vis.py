#circle_bundles/fiberwise_clustering_vis.py
from __future__ import annotations

from dataclasses import dataclass

from collections import defaultdict
from typing import Callable, Optional, Tuple, Union

import numpy as np
import networkx as nx
from matplotlib.figure import Figure

from .image_utils import fig_to_rgba, trim_image

__all__ = [
    "make_patch_cluster_diagram",
    "get_G_vertex_coords",
    "GraphComponentData",
    "extract_component_subgraph",
    "representative_indices_for_clusters",
    "component_patch_reps",
    "plot_component_patch_diagram",
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
):
    """
    Display patch thumbnails in a grid by (j,k) with graph edges drawn behind.

    Assumptions
    -----------
    - data is (N, d)
    - clusters is (N, 2) with rows (j,k) corresponding to data rows
    - G has nodes labeled (j,k)
    - vis_func returns either a Matplotlib Figure OR an image array (H,W,3/4)
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    data = np.asarray(data)
    clusters = np.asarray(clusters)

    if clusters.ndim != 2 or clusters.shape[1] != 2:
        raise ValueError(f"clusters must be shape (N,2). Got {clusters.shape}.")
    if data.shape[0] != clusters.shape[0]:
        raise ValueError(f"data and clusters must have same N. Got {data.shape[0]} vs {clusters.shape[0]}.")

    if data.shape[0] == 0:
        raise ValueError("Empty data.")

    unique_js = np.unique(clusters[:, 0].astype(int))
    unique_js.sort()

    coord_map: dict[tuple[int, int], tuple[float, float]] = {}
    for col_idx, j in enumerate(unique_js):
        ks = np.unique(clusters[clusters[:, 0] == j, 1].astype(int))
        ks.sort()
        for row_idx, k in enumerate(ks):
            coord_map[(int(j), int(k))] = (col_idx * float(col_spacing), -row_idx * float(row_spacing))

    if not coord_map:
        raise ValueError("No cluster coordinates could be constructed (coord_map empty).")

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
            if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
                raise ValueError(f"vis_func returned array with invalid shape {rgba.shape} at idx={idx}.")
            if rgba.shape[2] == 3:
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
    if len(nodes) == 0:
        return np.zeros((0, 2), dtype=float)

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


@dataclass(frozen=True)
class GraphComponentData:
    Gm: nx.Graph
    clusters: np.ndarray           # (n_nodes, 2) array of (j,k) in node order
    rep_indices: np.ndarray        # (n_nodes,) indices into data, -1 if missing
    valid_mask: np.ndarray         # (n_nodes,) bool, rep_indices >= 0
    patch_reps: Optional[np.ndarray] = None  # (n_valid, ...) subset of data
    clusters_valid: Optional[np.ndarray] = None  # (n_valid, 2)


def extract_component_subgraph(G: nx.Graph, component_index: int = 0) -> nx.Graph:
    comps = list(nx.connected_components(G))
    if component_index < 0 or component_index >= len(comps):
        raise IndexError(f"component_index={component_index} out of range (n_components={len(comps)}).")
    return G.subgraph(comps[component_index]).copy()


def representative_indices_for_clusters(
    Gm: nx.Graph,
    cl: np.ndarray,
    *,
    require_match: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each node (j,k) in Gm, pick the smallest n such that cl[j,n] == k.

    Returns
    -------
    clusters : (n_nodes,2) int array of (j,k) in node order
    rep_indices : (n_nodes,) int array; -1 if no match (unless require_match=True)
    """
    nodes = list(Gm.nodes())
    clusters = np.zeros((len(nodes), 2), dtype=int)
    rep_indices = np.empty(len(nodes), dtype=int)

    cl = np.asarray(cl)
    if cl.ndim != 2:
        raise ValueError(f"cl must be 2D (n_charts, n_samples). Got {cl.shape}.")

    for idx, (j, k) in enumerate(nodes):
        j = int(j)
        k = int(k)
        clusters[idx] = (j, k)

        if j < 0 or j >= cl.shape[0]:
            raise IndexError(f"Node has j={j} but cl has shape {cl.shape}.")

        n_matches = np.where(cl[j] == k)[0]
        if n_matches.size == 0:
            if require_match:
                raise ValueError(f"No datapoint found for cluster (j,k)=({j},{k}).")
            rep_indices[idx] = -1
        else:
            rep_indices[idx] = int(n_matches.min())

    return clusters, rep_indices


def component_patch_reps(
    G: nx.Graph,
    data: np.ndarray,
    cl: np.ndarray,
    *,
    component_index: int = 0,
    require_match: bool = False,
    return_only_valid: bool = True,
) -> GraphComponentData:
    """
    End-to-end: pick a connected component, compute cluster reps.

    By default, returns only valid patch reps (so missing clusters are dropped).
    """
    data = np.asarray(data)

    Gm = extract_component_subgraph(G, component_index=component_index)
    clusters, rep_indices = representative_indices_for_clusters(Gm, cl, require_match=require_match)

    valid_mask = rep_indices >= 0

    if return_only_valid:
        patch_reps = data[rep_indices[valid_mask]]
        clusters_valid = clusters[valid_mask]
        return GraphComponentData(
            Gm=Gm,
            clusters=clusters,
            rep_indices=rep_indices,
            valid_mask=valid_mask,
            patch_reps=patch_reps,
            clusters_valid=clusters_valid,
        )

    # If caller wants “all nodes” semantics, do not index with -1.
    patch_reps_all = None
    clusters_valid = None
    return GraphComponentData(
        Gm=Gm,
        clusters=clusters,
        rep_indices=rep_indices,
        valid_mask=valid_mask,
        patch_reps=patch_reps_all,
        clusters_valid=clusters_valid,
    )


def plot_component_patch_diagram(
    G: nx.Graph,
    data: np.ndarray,
    cl: np.ndarray,
    vis_func: Callable[[np.ndarray], Union[np.ndarray, Figure]],
    *,
    component_index: int = 0,
    require_match: bool = False,
    image_zoom: float = 0.4,
    row_spacing: float = 1.5,
    col_spacing: float = 1.5,
    line_color: str = "black",
    line_width: float = 5.0,
    save_path: Optional[str] = None,
    white_thresh: int = 250,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    One-call helper: extract component reps and plot them.

    Missing clusters are dropped unless require_match=True.
    """
    comp = component_patch_reps(
        G,
        data,
        cl,
        component_index=component_index,
        require_match=require_match,
        return_only_valid=True,
    )

    fig, ax = make_patch_cluster_diagram(
        comp.patch_reps,        # type: ignore[arg-type]
        comp.clusters_valid,    # type: ignore[arg-type]
        comp.Gm,
        vis_func,
        image_zoom=image_zoom,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        line_color=line_color,
        line_width=line_width,
        save_path=save_path,
        white_thresh=white_thresh,
        figsize=figsize,
    )
    return fig, ax, comp
