# circle_bundles/fiberwise_clustering.py
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

__all__ = [
    "safe_add_edges",
    "get_weights",
    "fiberwise_clustering",
    "plot_fiberwise_pca_grid",
    "plot_fiberwise_summary_bars",
    "get_cluster_persistence",
    "get_filtered_cluster_graph",
]


# ---------------------------------
# Graph construction helpers
# ---------------------------------

def safe_add_edges(G: nx.Graph, U: np.ndarray, cl: np.ndarray) -> None:
    """
    Add edges between cluster-nodes (j, c_j) and (k, c_k) when fibers j,k
    share at least one sample that is non-noise in BOTH fibers.

    Edge attributes:
      - indices_shared : sorted list of sample indices supporting the edge
    """
    U = np.asarray(U, dtype=bool)
    cl = np.asarray(cl)
    n_fibers, n_samples = U.shape
    if cl.shape != (n_fibers, n_samples):
        raise ValueError(f"cl must have shape {U.shape}, got {cl.shape}")

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


def get_weights(G: nx.Graph, method: str = "cardinality") -> nx.Graph:
    """
    Assign/overwrite edge 'weight' based on overlap of endpoint node memberships.

    method:
      - 'cardinality' : |A ∩ B|
      - 'rel_card'    : |A ∩ B| / min(|A|,|B|)
      - 'rel_card2'   : |A ∩ B| / ((|A|+|B|)/2)
    """
    if method not in {"cardinality", "rel_card", "rel_card2"}:
        raise ValueError("method must be one of: 'cardinality', 'rel_card', 'rel_card2'")

    for u, v, data in G.edges(data=True):
        Au = set(G.nodes[u].get("indices", []))
        Av = set(G.nodes[v].get("indices", []))
        inter = Au & Av
        inter_size = float(len(inter))

        if method == "cardinality":
            data["weight"] = inter_size

        elif method == "rel_card":
            denom = float(min(len(Au), len(Av)))
            data["weight"] = inter_size / denom if denom > 0 else 0.0

        elif method == "rel_card2":
            denom = float((len(Au) + len(Av)) / 2.0)
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
) -> tuple[np.ndarray, nx.Graph, dict, np.ndarray, dict]:
    """
    Fiberwise DBSCAN clustering, then a global cluster-graph built from overlaps.

    Parameters
    ----------
    data : (n_samples, d)
    U : (n_fibers, n_samples) bool
    eps_values : (n_fibers,) float
    min_sample_values : (n_fibers,) int

    Returns
    -------
    components : (n_samples,) int
        Global component label per sample (based on connected components of the cluster-graph).
    G : nx.Graph
        Nodes are (fiber_idx, cluster_label) with attribute 'indices' (sample indices).
        Edges indicate overlap; may include 'indices_shared' and later 'weight'.
    graph_dict : dict
        Simple serialization-friendly summary.
    cl : (n_fibers, n_samples) int
        DBSCAN labels within each fiber (-1 noise).
    summary : dict
        Useful arrays and optional PCA embeddings for plotting.
    """
    U = np.asarray(U, dtype=bool)
    data = np.asarray(data)
    eps_values = np.asarray(eps_values, dtype=float)
    min_sample_values = np.asarray(min_sample_values, dtype=int)

    n_fibers, n_samples = U.shape
    if data.shape[0] != n_samples:
        raise ValueError(f"data must have n_samples={n_samples} rows, got {data.shape[0]}")
    if eps_values.shape[0] != n_fibers or min_sample_values.shape[0] != n_fibers:
        raise ValueError("eps_values and min_sample_values must have length n_fibers")

    cl = -1 * np.ones((n_fibers, n_samples), dtype=int)
    G = nx.Graph()

    fiber_component_counts = np.zeros(n_fibers, dtype=int)
    pca_store: dict[int, dict[str, Any]] = {}

    # --- Fiberwise DBSCAN + graph nodes ---
    for r in range(n_fibers):
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

        if build_pca_embeddings and fiber_pts.shape[0] > 1 and pca_dim > 0:
            pca = PCA(n_components=int(pca_dim))
            pca_data = pca.fit_transform(fiber_pts)
            pca_store[r] = {"pca": pca_data, "clusters": labels, "row_inds": row_inds}

    # --- Overlap edges ---
    safe_add_edges(G, U, cl)

    # --- Global components ---
    components = -1 * np.ones(n_samples, dtype=int)
    global_component_counts: list[int] = []
    point_counts: list[int] = []

    connected_components = list(nx.connected_components(G))
    for comp_id, comp_nodes in enumerate(connected_components):
        global_component_counts.append(len(comp_nodes))

        all_inds: list[int] = []
        for node in comp_nodes:
            all_inds.extend(G.nodes[node]["indices"])
        all_inds = np.unique(all_inds).astype(int)

        components[all_inds] = comp_id
        point_counts.append(int(len(all_inds)))

    # --- graph_dict (fix links accumulation!) ---
    nodes_dict = {
        f"fiber{node[0]}_cluster{node[1]}": G.nodes[node]["indices"]
        for node in G.nodes
    }
    links_dict: dict[str, list[str]] = defaultdict(list)
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
        "n_fibers": n_fibers,
        "n_samples": n_samples,
    }

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
    Plot PCA scatter for selected fibers using summary['pca_store'].
    Requires fiberwise_clustering(..., build_pca_embeddings=True).
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

def get_cluster_persistence(
    G: nx.Graph,
    *,
    show_results: bool = True,
    save_path: Optional[str] = None,
):
    """
    Track number of connected components as we remove edges by nondecreasing weight.

    Returns a list of dicts:
      {'weight': w, 'n_components': int, 'components': [set(nodes), ...]}
    """
    import matplotlib.pyplot as plt

    Gc = G.copy()
    history = []

    comps0 = list(nx.connected_components(Gc))
    history.append({"weight": -np.inf, "n_components": len(comps0), "components": comps0})

    edges_by_w = defaultdict(list)
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
    G: nx.Graph,
    cl: np.ndarray,
    *,
    thresh: float,
    rule: str = "to_smaller_cluster",
) -> tuple[nx.Graph, np.ndarray, nx.Graph]:
    """
    For each edge with weight <= thresh, split intersection samples so that those samples
    belong to ONLY ONE endpoint cluster (chosen by rule).

    rule:
      - 'to_smaller_cluster': keep shared samples in smaller cluster, remove from larger
      - 'to_larger_cluster' : keep shared samples in larger cluster, remove from smaller
    """
    if rule not in ("to_smaller_cluster", "to_larger_cluster"):
        raise ValueError("rule must be 'to_smaller_cluster' or 'to_larger_cluster'")

    G_clean = G.copy()
    cl_clean = np.array(cl, copy=True)

    edges_to_remove = []
    for u, v, data in G_clean.edges(data=True):
        w = float(data.get("weight", 0.0))
        if w <= float(thresh):
            edges_to_remove.append((u, v))

    removed_from_node: dict[tuple[int, int], set[int]] = defaultdict(set)

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
    G: nx.Graph,
    cl: np.ndarray,
    *,
    thresh: float,
    rule: str = "to_smaller_cluster",
    show_results: bool = True,
    hide_biggest: bool = False,
    save_path: Optional[str] = None,
):
    """
    Threshold the weighted cluster graph, separate intersections for removed edges,
    and recompute global components.

    Returns
    -------
    components_filtered : (n_samples,) int
    filtered_G : nx.Graph
    graph_dict_filtered : dict
    cl_clean : (n_fibers, n_samples) int
    comp_inds : (n_components, n_samples) bool
    """
    import matplotlib.pyplot as plt

    data = np.asarray(data)
    n_samples = data.shape[0]

    G_clean, cl_clean, filtered_G = _separate_intersections_on_removed_edges(
        G, cl, thresh=float(thresh), rule=rule
    )

    comps = list(nx.connected_components(filtered_G))
    components_filtered = -1 * np.ones(n_samples, dtype=int)

    global_component_counts: list[int] = []
    point_counts: list[int] = []

    for comp_id, component in enumerate(comps):
        global_component_counts.append(len(component))
        all_inds: list[int] = []
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

    links_dict: dict[str, list[str]] = defaultdict(list)
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
