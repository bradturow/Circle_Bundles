from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx

from ..combinatorics import canon_edge

Edge = Tuple[int, int]

__all__ = ["graph_to_st", "create_st_dicts"]


def graph_to_st(G: nx.Graph, *, max_dim: int = 2, use_weights: bool = False):
    """
    Convert a NetworkX graph to a Gudhi SimplexTree by inserting:
      - vertices at filtration 0
      - edges at filtration = weight (if use_weights) else 0
      - cliques up to dimension max_dim (optional)
    """
    try:
        import gudhi as gd  # type: ignore
    except Exception as e:
        raise ImportError("graph_to_st requires gudhi to be installed.") from e

    st = gd.SimplexTree()

    for node in G.nodes():
        st.insert([node], filtration=0.0)

    for u, v, data in G.edges(data=True):
        filt = float(data.get("weight", 0.0)) if use_weights else 0.0
        st.insert([u, v], filtration=filt)

    if int(max_dim) > 1:
        from networkx.algorithms.clique import find_cliques
        for clique in find_cliques(G):
            if len(clique) <= int(max_dim) + 1:
                st.insert(list(clique), filtration=0.0)

    return st


def create_st_dicts(
    G0: nx.Graph,
    filtered_G0: nx.Graph,
    *,
    max_dim: int = 1,
):
    """
    Convert graphs to simplex trees using integer vertex ids, and build:
      - vertex_dict: vertex id -> component id in filtered_G0 (or -1)
      - edge_dict: canonical (u,v) -> component id if endpoints in same component, else -1
      - node_to_index: original node label -> integer id
    """
    try:
        import gudhi as gd  # type: ignore
    except Exception as e:
        raise ImportError("create_st_dicts requires gudhi to be installed.") from e

    nodes = list(G0.nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}

    def build_st(G: nx.Graph):
        st = gd.SimplexTree()
        for node in G.nodes():
            st.insert([node_to_index[node]], filtration=0.0)
        for u, v in G.edges():
            iu, iv = node_to_index[u], node_to_index[v]
            if iu == iv:
                continue
            a, b = canon_edge(iu, iv)
            st.insert([a, b], filtration=0.0)

        # If you ever want cliques here, we can mirror graph_to_st behavior.
        if int(max_dim) > 1:
            # optional: insert triangles etc from cliques
            from networkx.algorithms.clique import find_cliques
            for clique in find_cliques(G):
                if len(clique) <= int(max_dim) + 1:
                    st.insert([node_to_index[x] for x in clique], filtration=0.0)

        return st

    G0_st = build_st(G0)
    filtered_st = build_st(filtered_G0)

    components = list(nx.connected_components(filtered_G0))
    node_to_comp = {}
    for cid, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = cid

    vertex_dict: Dict[int, int] = {}
    for simplex, _ in G0_st.get_skeleton(0):
        idx = int(simplex[0])
        orig_node = nodes[idx]
        vertex_dict[idx] = int(node_to_comp.get(orig_node, -1))

    edge_dict: Dict[Edge, int] = {}
    for simplex, _ in G0_st.get_skeleton(1):
        if len(simplex) != 2:
            continue
        u, v = int(simplex[0]), int(simplex[1])
        a, b = canon_edge(u, v)
        node_u, node_v = nodes[a], nodes[b]
        cu = int(node_to_comp.get(node_u, -1))
        cv = int(node_to_comp.get(node_v, -1))
        edge_dict[(a, b)] = cu if (cu == cv and cu != -1) else -1

    return G0_st, filtered_st, vertex_dict, edge_dict, node_to_index
