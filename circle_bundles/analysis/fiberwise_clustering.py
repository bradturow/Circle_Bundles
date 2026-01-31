# circle_bundles/fiberwise_clustering.py
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "fiberwise_clustering",
    "plot_fiberwise_pca_grid",
    "plot_fiberwise_summary_bars",
    "get_cluster_persistence",
    "get_filtered_cluster_graph",
    "get_weights"
]


# ---------------------------------
# Internal: dependency guards
# ---------------------------------

def _require_networkx():
    try:
        import networkx as nx  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "fiberwise_clustering requires networkx. Install with `pip install networkx`."
        ) from e
    return nx


def _require_sklearn():
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        from sklearn.decomposition import PCA  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "fiberwise_clustering requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e
    return DBSCAN, PCA


# ---------------------------------
# Graph construction helpers
# ---------------------------------

def safe_add_edges(G, U: np.ndarray, cl: np.ndarray) -> None:
    """
    Add edges between cluster-nodes (j, c_j) and (k, c_k) when fibers j,k
    share at least one sample that is non-noise in BOTH fibers.

    Node convention:
      node = (fiber_index, cluster_label)

    Edge attributes:
      - indices_shared : sorted list of sample indices supporting the edge
    """
    U = np.asarray(U, dtype=bool)
    cl = np.asarray(cl)
    n_fibers, n_samples = U.shape
    if cl.shape != (n_fibers, n_samples):
        raise ValueError(f"cl must have shape {U.shape}, got {cl.shape}")

    # Accumulate shared indices per (u,v) edge deterministically
    shared: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[int]] = defaultdict(list)

    for k in range(n_fibers):
        Uk = U[k]
        for j in range(k):
            overlap = Uk & U[j]
            if not np.any(overlap):
                continue

            inds = np.where(overlap)[0]
            cj = cl[j, inds]
            ck = cl[k, inds]

            valid = (cj != -1) & (ck != -1)
            if not np.any(valid):
                continue

            inds = inds[valid]
            cj = cj[valid].astype(int)
            ck = ck[valid].astype(int)

            for idx, a, b in zip(inds, cj, ck):
                u = (j, int(a))
                v = (k, int(b))
                # Only connect clusters that exist as nodes
                if (u not in G) or (v not in G):
                    continue
                key = (u, v) if u <= v else (v, u)  # deterministic undirected key
                shared[key].append(int(idx))

    for (u, v), idxs in shared.items():
        idxs_sorted = sorted(set(idxs))
        if G.has_edge(u, v):
            prev = G.edges[u, v].get("indices_shared", [])
            merged = sorted(set(prev).union(idxs_sorted))
            G.edges[u, v]["indices_shared"] = merged
        else:
            G.add_edge(u, v, indices_shared=idxs_sorted)


def get_weights(G, method: str = "cardinality"):
    """
    Compute and attach edge weights for a fiberwise cluster graph.

    This function assigns/overwrites the edge attribute ``"weight"`` in-place, based on
    how much the endpoint clusters overlap in sample membership. It is typically used
    after :func:`fiberwise_clustering` to enable thresholding and simple persistence-style
    diagnostics via :func:`get_cluster_persistence` and :func:`get_filtered_cluster_graph`.

    Parameters
    ----------
    G:
        A ``networkx.Graph`` whose nodes represent fiber-clusters, and whose nodes store
        an ``"indices"`` attribute: the list of sample indices belonging to that cluster.
        If edges contain ``"indices_shared"``, those will be used directly (faster and
        consistent with the overlap construction in :func:`fiberwise_clustering`).
    method:
        Overlap-to-weight rule. One of:
          - ``"cardinality"``: weight = ``|A ∩ B|``
          - ``"rel_card"``: weight = ``|A ∩ B| / min(|A|, |B|)``
          - ``"rel_card2"``: weight = ``|A ∩ B| / ((|A| + |B|)/2)``

        Here ``A`` and ``B`` are the endpoint node membership sets.

    Returns
    -------
    G:
        The same graph instance, returned for convenience, with edge attribute ``"weight"``
        set on every edge.

    Notes
    -----
    - The graph is modified in-place.
    - If an edge has ``indices_shared``, we use ``len(indices_shared)`` as ``|A ∩ B|``.
      Otherwise we fall back to computing the intersection from node ``indices``.
    """
    if method not in {"cardinality", "rel_card", "rel_card2"}:
        raise ValueError("method must be one of: 'cardinality', 'rel_card', 'rel_card2'")

    for u, v, data in G.edges(data=True):
        # overlap size: prefer indices_shared (created by safe_add_edges)
        shared = data.get("indices_shared", None)
        if shared is not None:
            inter_size = float(len(shared))
        else:
            # fallback: compute from node indices
            Au = set(G.nodes[u].get("indices", []))
            Av = set(G.nodes[v].get("indices", []))
            inter_size = float(len(Au & Av))

        if method == "cardinality":
            data["weight"] = inter_size
            continue

        # endpoint sizes
        size_u = float(len(G.nodes[u].get("indices", [])))
        size_v = float(len(G.nodes[v].get("indices", [])))

        if method == "rel_card":
            denom = float(min(size_u, size_v))
            data["weight"] = inter_size / denom if denom > 0 else 0.0
        else:  # rel_card2
            denom = float((size_u + size_v) / 2.0)
            data["weight"] = inter_size / denom if denom > 0 else 0.0

    return G


# ---------------------------------
# Main compute routine
# ---------------------------------

def fiberwise_clustering(
    data: np.ndarray,
    U: np.ndarray,
    eps_values: np.ndarray,
    min_sample_values: np.ndarray,
    *,
    build_pca_embeddings: bool = True,
    pca_dim: int = 2,
    verbose: bool = True,
):
    """
    Cluster each fiber with DBSCAN and then merge clusters globally via an overlap graph.

    This is a useful diagnostic tool when you have a cover (fibers) over the dataset and
    want to see whether local cluster structure aligns across overlaps.

    Workflow
    --------
    1) For each fiber r (row of ``U``), run DBSCAN on the points ``data[U[r]]`` using
       ``eps_values[r]`` and ``min_sample_values[r]``.
    2) Create a graph whose nodes are *fiber-clusters* ``(r, label)`` (excluding label ``-1``).
       Each node stores the list of sample indices belonging to that DBSCAN cluster.
    3) Add an edge between two nodes if the corresponding fiber-clusters share at least one
       sample index that is non-noise in both fibers. Edges store ``indices_shared``.
    4) Define a global component label per sample by taking connected components of the graph
       and assigning each sample that appears in any node of that connected component.

    Parameters
    ----------
    data:
        Array of shape ``(n_samples, d)`` containing the ambient data vectors.
    U:
        Boolean membership matrix of shape ``(n_fibers, n_samples)``. Row ``r`` indicates
        which samples lie in fiber/cover set ``U_r``.
    eps_values:
        Array of shape ``(n_fibers,)`` with the DBSCAN ``eps`` parameter used for each fiber.
    min_sample_values:
        Array of shape ``(n_fibers,)`` with the DBSCAN ``min_samples`` parameter used for each fiber.
    build_pca_embeddings:
        If True, compute PCA embeddings (within each fiber) for quick plotting with
        :func:`plot_fiberwise_pca_grid`.
    pca_dim:
        Number of PCA components to compute per fiber (typically 2).

    Returns
    -------
    components:
        Integer array of shape ``(n_samples,)`` giving a global component label per sample.
        Samples not belonging to any non-noise fiber-cluster remain ``-1``.
    G:
        A ``networkx.Graph``. Nodes are tuples ``(fiber_idx, cluster_label)`` with node attribute
        ``indices`` (list of sample indices). Edges indicate overlap, and store
        ``indices_shared`` (the supporting sample indices).
    graph_dict:
        A simple serialization-friendly representation of the graph (nodes/links).
    cl:
        Integer array of shape ``(n_fibers, n_samples)`` giving the DBSCAN label of each sample
        within each fiber. Label ``-1`` denotes noise (or not present in that fiber).
    summary:
        Dict of helpful arrays for downstream plotting, including:
          - ``fiber_component_counts``: number of DBSCAN clusters per fiber (excluding noise)
          - ``global_component_counts``: number of graph nodes per global component
          - ``point_counts``: number of samples assigned to each global component
          - ``pca_store``: per-fiber PCA embeddings if requested

    Notes
    -----
    - This routine requires optional dependencies: ``networkx`` and ``scikit-learn``.
    - Edge weights are not computed here. If you want a weighted filtration on the cluster graph,
      call :func:`get_weights` (internal helper) or attach your own ``edge["weight"]`` values.

    Examples
    --------
    >>> components, G, graph_dict, cl, summary = fiberwise_clustering(
    ...     data, U,
    ...     eps_values=np.full(U.shape[0], 0.25),
    ...     min_sample_values=np.full(U.shape[0], 10),
    ... )
    >>> fig, _ = plot_fiberwise_summary_bars(summary)
    """
    from ..utils.status_utils import _status, _status_clear

    def _v(msg: str):
        if verbose:
            _status(msg)
    
    nx = _require_networkx()
    DBSCAN, PCA = _require_sklearn()

    U = np.asarray(U, dtype=bool)
    data = np.asarray(data)
    eps_values = np.asarray(eps_values, dtype=float)
    min_sample_values = np.asarray(min_sample_values, dtype=int)

    n_fibers, n_samples = U.shape
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_samples, d). Got shape {data.shape}.")
    if data.shape[0] != n_samples:
        raise ValueError(f"data must have n_samples={n_samples} rows, got {data.shape[0]}")
    if eps_values.shape != (n_fibers,):
        raise ValueError(f"eps_values must have shape ({n_fibers},), got {eps_values.shape}")
    if min_sample_values.shape != (n_fibers,):
        raise ValueError(f"min_sample_values must have shape ({n_fibers},), got {min_sample_values.shape}")

    cl = -1 * np.ones((n_fibers, n_samples), dtype=int)
    G = nx.Graph()

    fiber_component_counts = np.zeros(n_fibers, dtype=int)
    pca_store: Dict[int, Dict[str, Any]] = {}

            
    # --- Fiberwise DBSCAN + graph nodes ---
    for r in range(n_fibers):
        _v(f"Clustering set {r+1}/{n_fibers}...")

        row_inds = np.where(U[r])[0]
        if row_inds.size == 0:
            continue

        fiber_pts = data[row_inds]
        db = DBSCAN(eps=float(eps_values[r]), min_samples=int(min_sample_values[r]))
        labels = db.fit_predict(fiber_pts)
        cl[r, row_inds] = labels

        unique_clusters = set(labels.tolist())
        unique_clusters.discard(-1)
        fiber_component_counts[r] = len(unique_clusters)

        for lab in unique_clusters:
            mask = labels == lab
            cluster_indices = row_inds[mask]
            G.add_node((r, int(lab)), indices=cluster_indices.tolist())

        if build_pca_embeddings and fiber_pts.shape[0] > 1 and int(pca_dim) > 0:
            pca = PCA(n_components=int(pca_dim))
            pca_data = pca.fit_transform(fiber_pts)
            pca_store[r] = {"pca": pca_data, "clusters": labels, "row_inds": row_inds}

    _v(f"Computing global clusters...")
            
    # --- Overlap edges ---
    safe_add_edges(G, U, cl)

    # --- Global components: map nodes -> union of their sample indices ---
    components = -1 * np.ones(n_samples, dtype=int)
    global_component_counts: List[int] = []
    point_counts: List[int] = []

    connected_components = list(nx.connected_components(G))
    for comp_id, comp_nodes in enumerate(connected_components):
        global_component_counts.append(len(comp_nodes))

        all_inds: List[int] = []
        for node in comp_nodes:
            all_inds.extend(G.nodes[node].get("indices", []))
        all_inds = np.unique(all_inds).astype(int)

        components[all_inds] = comp_id
        point_counts.append(int(len(all_inds)))

    # --- graph_dict (simple serialization) ---
    nodes_dict = {
        f"fiber{node[0]}_cluster{node[1]}": G.nodes[node].get("indices", [])
        for node in G.nodes
    }
    links_dict: Dict[str, List[str]] = defaultdict(list)
    for u, v in G.edges:
        ku = f"fiber{u[0]}_cluster{u[1]}"
        kv = f"fiber{v[0]}_cluster{v[1]}"
        links_dict[ku].append(kv)
        links_dict[kv].append(ku)

    graph_dict = {
        "nodes": nodes_dict,
        "links": dict(links_dict),
        "simplices": [[f"fiber{node[0]}_cluster{node[1]}"] for node in G.nodes],
        "meta_data": {
            "projection": "custom",
            "n_cubes": int(n_fibers),
            "perc_overlap": 0.5,
            "clusterer": "DBSCAN()",
            "scaler": "None",
            "nerve_min_intersection": 1,
        },
        "meta_nodes": {},
    }

    summary = {
        "fiber_component_counts": fiber_component_counts,
        "global_component_counts": np.array(global_component_counts, dtype=int),
        "point_counts": np.array(point_counts, dtype=int),
        "pca_store": pca_store,
        "n_fibers": int(n_fibers),
        "n_samples": int(n_samples),
    }

    if verbose:
        _status_clear()
    
    return components, G, graph_dict, cl, summary


# ---------------------------------
# Plotting helpers
# ---------------------------------

def plot_fiberwise_pca_grid(
    summary: dict,
    *,
    to_view: Optional[Sequence[int]] = None,
    cmap: str = "viridis",
    point_size: float = 5.0,
    n_cols: int = 4,
    save_path: Optional[str] = None,
):
    """
    Plot per-fiber PCA scatter plots colored by DBSCAN cluster labels.

    This function consumes the ``summary`` returned by :func:`fiberwise_clustering`
    when called with ``build_pca_embeddings=True``.

    Parameters
    ----------
    summary:
        The ``summary`` dict returned by :func:`fiberwise_clustering`.
    to_view:
        Optional list of fiber indices to plot. If omitted, plots all fibers that
        have PCA data available.
    cmap:
        Matplotlib colormap name used to color points by cluster label.
    point_size:
        Scatter marker size passed to Matplotlib.
    n_cols:
        Number of columns in the subplot grid.
    save_path:
        If provided, save the figure to this path (PDF recommended).

    Returns
    -------
    fig, axes:
        The Matplotlib figure and a flattened array of axes. Returns ``(None, None)``
        if there is nothing to plot.

    Raises
    ------
    ValueError
        If PCA data is missing (i.e. ``summary["pca_store"]`` is empty).
    """
    import matplotlib.pyplot as plt

    pca_store = summary.get("pca_store", {})
    if not pca_store:
        raise ValueError("No PCA data found. Run fiberwise_clustering with build_pca_embeddings=True.")

    fibers_available = sorted(pca_store.keys())
    fibers = fibers_available if not to_view else [int(r) for r in to_view if int(r) in pca_store]
    if len(fibers) == 0:
        return None, None

    n_rows = int(np.ceil(len(fibers) / int(n_cols)))
    fig, axes = plt.subplots(n_rows, int(n_cols), figsize=(4 * int(n_cols), 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(fibers):]:
        ax.axis("off")

    for i, r in enumerate(fibers):
        ax = axes[i]
        pca_data = pca_store[r]["pca"]
        clusters = pca_store[r]["clusters"]
        ax.scatter(pca_data[:, 0], pca_data[:, 1], s=point_size, c=clusters, cmap=cmap)
        ax.set_title(f"Fiber {r}", fontsize=12)
        ax.set_aspect("equal", "box")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
    return fig, axes


def plot_fiberwise_summary_bars(
    summary: dict,
    *,
    hide_biggest: bool = False,
    save_path: Optional[str] = None,
    dpi: int = 200,
):
    """
    Plot quick bar-chart summaries of fiberwise clustering and global components.

    Produces three panels:
      1) number of DBSCAN clusters per fiber (excluding noise)
      2) number of cluster-nodes per global connected component (sorted)
      3) number of samples per global connected component (sorted)

    Parameters
    ----------
    summary:
        The ``summary`` dict returned by :func:`fiberwise_clustering`.
    hide_biggest:
        If True, drop the single largest global component in the "points per component"
        panel (useful when one component dominates the scale).
    save_path:
        If provided, saves a PDF with ``"_summary"`` appended to the filename.
    dpi:
        Figure DPI.

    Returns
    -------
    fig, axs:
        Matplotlib figure and axes array.
    """
    import matplotlib.pyplot as plt

    fiber_component_counts = np.asarray(summary["fiber_component_counts"])
    global_component_counts = np.asarray(summary["global_component_counts"])
    point_counts = np.asarray(summary["point_counts"])

    sorted_component_counts = sorted(global_component_counts.tolist())
    sorted_point_counts = sorted(point_counts.tolist())
    if hide_biggest and len(sorted_point_counts) > 0:
        sorted_point_counts = sorted_point_counts[:-1]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=int(dpi))

    axs[0].bar(np.arange(1, len(fiber_component_counts) + 1), fiber_component_counts, edgecolor="black", linewidth=1.0)
    axs[0].set_title("Number of Clusters per Fiber")
    axs[0].set_xlabel("Fiber Number")
    axs[0].set_ylabel("Number of Clusters")
    axs[0].grid(True, alpha=0.25)

    axs[1].bar(np.arange(1, len(sorted_component_counts) + 1), sorted_component_counts, edgecolor="black", linewidth=1.0)
    axs[1].set_title("Clusters per Global Component (Sorted)")
    axs[1].set_xlabel("Global Component Index")
    axs[1].set_ylabel("Cluster Count")
    axs[1].grid(True, alpha=0.25)

    axs[2].bar(np.arange(1, len(sorted_point_counts) + 1), sorted_point_counts, edgecolor="black", linewidth=1.0)
    axs[2].set_title("Points per Global Component (Sorted)")
    axs[2].set_xlabel("Global Component Index")
    axs[2].set_ylabel("Point Count")
    axs[2].grid(True, alpha=0.25)

    fig.tight_layout()

    if save_path is not None:
        out = save_path
        if out.lower().endswith(".pdf"):
            out = out[:-4] + "_summary.pdf"
        else:
            out = out + "_summary.pdf"
        fig.savefig(out, format="pdf", bbox_inches="tight")

    plt.show()
    return fig, axs


# ---------------------------------
# Edge-threshold "persistence" + filtering
# ---------------------------------

def _ensure_has_weights(G) -> None:
    # If there are edges but none have a 'weight', warn loudly (or raise).
    if G.number_of_edges() == 0:
        return
    has_any = any(("weight" in data) for _, _, data in G.edges(data=True))
    if not has_any:
        raise ValueError(
            "Graph has no edge weights. Run `get_weights(G, method=...)` first "
            "or add a 'weight' attribute to each edge."
        )


def get_cluster_persistence(
    G,
    *,
    show_results: bool = True,
    save_path: Optional[str] = None,
):
    """
    Track connected-component count as low-weight edges are removed.

    Interprets the cluster graph ``G`` as a weighted graph (edge attribute ``"weight"``),
    and forms a simple edge-threshold filtration:

    - Start with the full graph.
    - For each distinct weight value ``w`` in increasing order, remove all edges with weight
      exactly ``w`` (equivalently: keep only edges with weight > w).
    - Record the number of connected components after each removal step.

    Parameters
    ----------
    G:
        A ``networkx.Graph`` whose edges have a numeric ``"weight"`` attribute.
        (Use the internal helper ``get_weights(G, method=...)`` or attach weights yourself.)
    show_results:
        If True, plot the curve ``(# components) vs (threshold weight)``.
    save_path:
        If provided and ``show_results=True``, save the plot to this path (PDF recommended).

    Returns
    -------
    history:
        List of dicts with keys:
          - ``"weight"``: the threshold value (first entry uses ``-np.inf`` for the full graph)
          - ``"n_components"``: number of connected components at that stage
          - ``"components"``: the list of node-sets returned by ``nx.connected_components``

    Raises
    ------
    ValueError
        If the graph has edges but none have a ``"weight"`` attribute.
    """
    nx = _require_networkx()
    _ensure_has_weights(G)

    import matplotlib.pyplot as plt

    Gc = G.copy()
    history: List[Dict[str, Any]] = []

    comps0 = list(nx.connected_components(Gc))
    history.append({"weight": -np.inf, "n_components": len(comps0), "components": comps0})

    edges_by_w: Dict[float, List[Tuple[Any, Any]]] = defaultdict(list)
    for u, v, data in Gc.edges(data=True):
        w = float(data.get("weight", 0.0))
        edges_by_w[w].append((u, v))

    for w in sorted(edges_by_w.keys()):
        Gc.remove_edges_from(edges_by_w[w])
        comps = list(nx.connected_components(Gc))
        history.append({"weight": w, "n_components": len(comps), "components": comps})

    if show_results:
        ws = [h["weight"] for h in history[1:]]  # skip -inf
        ns = [h["n_components"] for h in history[1:]]

        plt.figure(figsize=(10, 6))
        plt.plot(ws, ns, marker="o", linewidth=2)
        plt.xlabel("Weight Threshold")
        plt.ylabel("Number of Connected Components")
        plt.title("Cluster Persistence: Weight vs. Number of Connected Components")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")

        plt.show()

    return history


def _separate_intersections_on_removed_edges(
    G,
    cl: np.ndarray,
    *,
    thresh: float,
    rule: str = "to_smaller_cluster",
):
    """
    For each edge with weight <= thresh, split intersection samples so that those samples
    belong to ONLY ONE endpoint cluster (chosen by rule).

    rule:
      - 'to_smaller_cluster': keep shared samples in smaller cluster, remove from larger
      - 'to_larger_cluster' : keep shared samples in larger cluster, remove from smaller
    """
    nx = _require_networkx()

    if rule not in ("to_smaller_cluster", "to_larger_cluster"):
        raise ValueError("rule must be 'to_smaller_cluster' or 'to_larger_cluster'")

    _ensure_has_weights(G)

    G_clean = G.copy()
    cl_clean = np.array(cl, copy=True)

    edges_to_remove: List[Tuple[Any, Any]] = []
    for u, v, data in G_clean.edges(data=True):
        w = float(data.get("weight", 0.0))
        if w <= float(thresh):
            edges_to_remove.append((u, v))

    removed_from_node: Dict[Tuple[int, int], set[int]] = defaultdict(set)

    for u, v in edges_to_remove:
        inds_u = set(G_clean.nodes[u].get("indices", []))
        inds_v = set(G_clean.nodes[v].get("indices", []))

        shared_edge = G_clean.edges[u, v].get("indices_shared", None) if G_clean.has_edge(u, v) else None
        inter = set(shared_edge) if shared_edge is not None else (inds_u & inds_v)
        if not inter:
            continue

        size_u = len(inds_u)
        size_v = len(inds_v)

        if rule == "to_smaller_cluster":
            remove_from = v if size_u <= size_v else u
        else:
            remove_from = v if size_u >= size_v else u

        r, label = remove_from
        to_remove = inter - removed_from_node[remove_from]
        if not to_remove:
            continue

        new_idx = [idx for idx in G_clean.nodes[remove_from].get("indices", []) if idx not in to_remove]
        G_clean.nodes[remove_from]["indices"] = new_idx
        removed_from_node[remove_from].update(to_remove)

        for idx in to_remove:
            if cl_clean[r, idx] == label:
                cl_clean[r, idx] = -1

    filtered_G = nx.Graph()
    filtered_G.add_nodes_from(G_clean.nodes(data=True))
    for u, v, data in G_clean.edges(data=True):
        if float(data.get("weight", 0.0)) > float(thresh):
            filtered_G.add_edge(u, v, **data)

    return G_clean, cl_clean, filtered_G


def get_filtered_cluster_graph(
    data: np.ndarray,
    G,
    cl: np.ndarray,
    *,
    thresh: float,
    rule: str = "to_smaller_cluster",
    show_results: bool = True,
    hide_biggest: bool = False,
    save_path: Optional[str] = None,
):
    """
    Threshold the weighted cluster graph and recompute global components.

    This is a “post-processing” step for :func:`fiberwise_clustering` once you have
    a weighted cluster graph ``G``:

    1) Remove edges with ``weight <= thresh`` (keep only edges with ``weight > thresh``).
    2) For each removed edge, resolve shared sample indices so that intersection samples are
       assigned to only one endpoint cluster (controlled by ``rule``). This updates the node
       membership lists and also produces a cleaned label matrix ``cl_clean``.
    3) Recompute connected components of the filtered graph and assign samples accordingly.

    Parameters
    ----------
    data:
        Original data array of shape ``(n_samples, d)`` (used here only for shape validation).
    G:
        A ``networkx.Graph`` whose edges have a numeric ``"weight"`` attribute and whose nodes
        store an ``"indices"`` list of member sample indices.
    cl:
        The per-fiber DBSCAN label matrix returned by :func:`fiberwise_clustering`
        (shape ``(n_fibers, n_samples)``).
    thresh:
        Edge threshold. Edges with ``weight <= thresh`` are removed.
    rule:
        How to resolve intersection samples on removed edges:
          - ``"to_smaller_cluster"``: keep shared samples in the smaller endpoint cluster
          - ``"to_larger_cluster"``: keep shared samples in the larger endpoint cluster
    show_results:
        If True, plot bar charts for clusters-per-component and points-per-component.
    hide_biggest:
        If True, drop the largest component from the points-per-component panel.
    save_path:
        If provided and ``show_results=True``, save a PDF with ``"_summary"`` appended.

    Returns
    -------
    components_filtered:
        Integer array of shape ``(n_samples,)`` giving the new global component label per sample.
        Unassigned samples remain ``-1``.
    filtered_G:
        The thresholded graph containing only edges with ``weight > thresh``.
    graph_dict_filtered:
        Serialization-friendly representation of the filtered graph.
    cl_clean:
        Cleaned copy of ``cl`` where samples removed from a cluster due to intersection resolution
        are set to ``-1`` for that fiber.
    comp_inds:
        Boolean array of shape ``(n_components, n_samples)`` indicating membership of each
        global component.

    Raises
    ------
    ValueError
        If the graph has edges but none have a ``"weight"`` attribute.
    """
    nx = _require_networkx()

    import matplotlib.pyplot as plt

    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (n_samples, d). Got {data.shape}.")
    n_samples = data.shape[0]

    G_clean, cl_clean, filtered_G = _separate_intersections_on_removed_edges(
        G, cl, thresh=float(thresh), rule=rule
    )

    comps = list(nx.connected_components(filtered_G))
    components_filtered = -1 * np.ones(n_samples, dtype=int)

    global_component_counts: List[int] = []
    point_counts: List[int] = []

    for comp_id, component in enumerate(comps):
        global_component_counts.append(len(component))
        all_inds: List[int] = []
        for node in component:
            all_inds.extend(filtered_G.nodes[node].get("indices", []))
        all_inds = np.unique(all_inds).astype(int)
        components_filtered[all_inds] = comp_id
        point_counts.append(int(len(all_inds)))

    n_components = len(comps)
    comp_inds = np.zeros((n_components, n_samples), dtype=bool)
    for j in range(n_components):
        comp_inds[j, components_filtered == j] = True

    if show_results:
        sorted_component_counts = sorted(global_component_counts)
        sorted_point_counts = sorted(point_counts)
        if hide_biggest and len(sorted_point_counts) > 0:
            sorted_point_counts = sorted_point_counts[:-1]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].bar(range(1, len(sorted_component_counts) + 1), sorted_component_counts, edgecolor="black")
        axs[0].set_title("Clusters per Global Component")
        axs[0].set_xlabel("Global Component Index")
        axs[0].set_ylabel("Cluster Count")
        axs[0].grid(True, alpha=0.25)

        axs[1].bar(range(1, len(sorted_point_counts) + 1), sorted_point_counts, edgecolor="black")
        axs[1].set_title("Points per Global Component")
        axs[1].set_xlabel("Global Component Index")
        axs[1].set_ylabel("Point Count")
        axs[1].grid(True, alpha=0.25)

        fig.tight_layout()
        if save_path is not None:
            out = save_path[:-4] + "_summary.pdf" if save_path.lower().endswith(".pdf") else save_path + "_summary.pdf"
            fig.savefig(out, format="pdf", bbox_inches="tight")
            print(f"✅ Saved filtered summary figure to {out}")
        plt.show()

    links_dict: Dict[str, List[str]] = defaultdict(list)
    for u, v in filtered_G.edges:
        ku = f"fiber{u[0]}_cluster{u[1]}"
        kv = f"fiber{v[0]}_cluster{v[1]}"
        links_dict[ku].append(kv)
        links_dict[kv].append(ku)

    graph_dict_filtered = {
        "nodes": {
            f"fiber{node[0]}_cluster{node[1]}": filtered_G.nodes[node].get("indices", [])
            for node in filtered_G.nodes
        },
        "links": dict(links_dict),
        "simplices": [[f"fiber{node[0]}_cluster{node[1]}"] for node in filtered_G.nodes],
        "meta_data": {
            "projection": "custom",
            "perc_overlap": 0.5,
            "clusterer": "DBSCAN()",
            "scaler": "None",
            "nerve_min_intersection": 1,
            "edge_threshold": float(thresh),
            "intersection_rule": rule,
        },
        "meta_nodes": {},
    }

    return components_filtered, filtered_G, graph_dict_filtered, cl_clean, comp_inds
