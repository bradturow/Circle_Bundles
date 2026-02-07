# circle_bundles/viz/nerve_circle.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..nerve.combinatorics import canon_edge

__all__ = [
    "show_circle_nerve",
    "is_single_cycle_graph",
    "cycle_order_from_edges",
    "reindex_edges",
    "reindex_vertex_dict",
    "reindex_edge_dict",
]



def show_circle_nerve(
    *,
    n_vertices: int,
    edges: list[tuple[int, int]],
    kept_edges: list[tuple[int, int]] | None = None,
    omega: dict[tuple[int, int], int] | None = None,
    weights: dict[tuple[int, int], float] | None = None,
    phi: dict[int, int] | None = None,
    title: str | None = None,

    # layout / display
    ax=None,
    figsize: tuple[float, float] = (8.0, 8.0),
    dpi: Optional[int] = None,

    # styling
    r: float = 1.0,
    node_size: float = 600,
    node_facecolor: str = "lightblue",
    node_edgecolor: str = "k",
    node_label_color: str = "k",
    removed_edge_color: str = "lightgray",
    removed_edge_lw: float = 1.5,
    kept_edge_color: str = "black",
    kept_edge_lw: float = 4.0,
    omega_color: str = "blue",
    phi_color: str = "red",
    weights_color: str = "black",
    fontsize_node: int = 12,
    fontsize_omega: int = 12,
    fontsize_phi: int = 12,
    fontsize_weights: int = 9,
    omega_offset: float = 0.09,
    weights_offset: float = 0.09,
    phi_offset: float = 0.14,
    save_path: str | None = None,
):
    """
    Circle-layout nerve plot.

    Contract
    --------
    - If ax is None: creates a brand-new figure/axes (fig.add_axes).
    - If ax is provided: draws into that axes only.
    - Never calls plt.show() or display().
    """

    n_vertices = int(n_vertices)
    if n_vertices <= 0:
        raise ValueError("n_vertices must be positive.")

    # ---- canonicalize edges ----
    E_all = sorted({canon_edge(int(i), int(j)) for (i, j) in edges if i != j})
    if kept_edges is None:
        E_keep = set(E_all)
    else:
        E_keep = {canon_edge(int(i), int(j)) for (i, j) in kept_edges if i != j}
    E_rem = [e for e in E_all if e not in E_keep]

    def canon_dict(d):
        if d is None:
            return None
        return {canon_edge(int(i), int(j)): v for (i, j), v in d.items() if i != j}

    omega = canon_dict(omega)
    weights = canon_dict(weights)

    # ---- create figure/axes safely ----
    created_fig = False
    if ax is None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # 🔒 isolated axes
        created_fig = True
    else:
        fig = ax.figure

    # ---- geometry ----
    ang = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    P = np.c_[r * np.cos(ang), r * np.sin(ang)]

    def draw_edges(edgelist, *, color, lw, z):
        for i, j in edgelist:
            ax.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]],
                    color=color, linewidth=lw, zorder=z)

    draw_edges(E_rem, color=removed_edge_color, lw=removed_edge_lw, z=1)
    draw_edges(E_keep, color=kept_edge_color, lw=kept_edge_lw, z=2)

    ax.scatter(
        P[:, 0], P[:, 1],
        s=node_size,
        c=node_facecolor,
        edgecolors=node_edgecolor,
        zorder=3,
    )

    for i in range(n_vertices):
        ax.text(
            P[i, 0], P[i, 1], f"$U_{{{i+1}}}$",
            ha="center", va="center",
            fontsize=fontsize_node,
            color=node_label_color,
            fontweight="bold",
            zorder=4,
        )

    if omega:
        for (i, j), val in omega.items():
            mid = 0.5 * (P[i] + P[j])
            u = mid / (np.linalg.norm(mid) + 1e-12)
            pos = mid + omega_offset * r * u
            ax.text(*pos, str(int(val)),
                    fontsize=fontsize_omega, color=omega_color,
                    ha="center", va="center", zorder=5)

    if weights:
        for (i, j), w in weights.items():
            mid = 0.5 * (P[i] + P[j])
            u = -mid / (np.linalg.norm(mid) + 1e-12)
            pos = mid + weights_offset * r * u
            ax.text(*pos, f"{float(w):.3g}",
                    fontsize=fontsize_weights, color=weights_color,
                    ha="center", va="center", zorder=5)

    if phi:
        for v, val in phi.items():
            u = P[v] / (np.linalg.norm(P[v]) + 1e-12)
            pos = P[v] + phi_offset * r * u
            ax.text(*pos, str(int(val)),
                    fontsize=fontsize_phi, color=phi_color,
                    ha="center", va="center", zorder=6)

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, pad=18)

    if save_path and created_fig:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def _edges_to_adj(n: int, edges: list[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for (a, b) in edges:
        a, b = int(a), int(b)
        if a == b:
            continue
        if not (0 <= a < n and 0 <= b < n):
            continue
        adj[a].add(b)
        adj[b].add(a)
    return adj


def is_single_cycle_graph(n: int, edges: list[tuple[int, int]]) -> tuple[bool, str]:
    """Check whether the undirected graph is a single cycle on all n vertices."""
    n = int(n)
    E = {canon_edge(int(a), int(b)) for (a, b) in edges if int(a) != int(b)}
    if n == 0:
        return False, "n=0"
    if len(E) != n:
        return False, f"|E|={len(E)} != n={n} (cycle needs exactly n edges)"

    adj = _edges_to_adj(n, list(E))
    degs = [len(adj[i]) for i in range(n)]
    if any(d != 2 for d in degs):
        bad = [i for i, d in enumerate(degs) if d != 2]
        return False, f"not 2-regular (bad vertices: {bad[:10]}...)"

    seen = set()
    stack = [0]
    while stack:
        v = stack.pop()
        if v in seen:
            continue
        seen.add(v)
        stack.extend(list(adj[v] - seen))
    if len(seen) != n:
        return False, f"graph not connected (reached {len(seen)}/{n})"

    return True, "ok"


def cycle_order_from_edges(n: int, edges: list[tuple[int, int]], start: int = 0) -> list[int]:
    """Given a single-cycle graph on {0..n-1}, return a cyclic order of vertices."""
    n = int(n)
    E = {canon_edge(int(a), int(b)) for (a, b) in edges if int(a) != int(b)}
    adj = _edges_to_adj(n, list(E))

    start = int(start)
    if not (0 <= start < n):
        start = 0

    order = [start]
    prev = None
    cur = start

    for _ in range(n - 1):
        nbrs = sorted(adj[cur])
        if len(nbrs) != 2:
            raise ValueError(f"Expected degree 2 at vertex {cur}, got neighbors {nbrs}")
        nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        order.append(nxt)
        prev, cur = cur, nxt

    return order


def reindex_edges(
    edges: Optional[list[tuple[int, int]]],
    old_to_new: dict[int, int],
) -> Optional[list[tuple[int, int]]]:
    if edges is None:
        return None
    out = []
    for (a, b) in edges:
        if int(a) not in old_to_new or int(b) not in old_to_new:
            raise KeyError(f"Missing vertex in old_to_new for edge {(a, b)}")
        out.append(canon_edge(old_to_new[int(a)], old_to_new[int(b)]))
    return out


def reindex_vertex_dict(d: dict[int, Any] | None, old_to_new: dict[int, int]) -> dict[int, Any] | None:
    if d is None:
        return None
    out: dict[int, Any] = {}
    for k, v in d.items():
        kk = int(k)
        if kk not in old_to_new:
            raise KeyError(f"Missing vertex in old_to_new for vertex {kk}")
        out[old_to_new[kk]] = v
    return out


def reindex_edge_dict(
    d: dict[tuple[int, int], Any] | None,
    old_to_new: dict[int, int],
) -> dict[tuple[int, int], Any] | None:
    if d is None:
        return None
    out: dict[tuple[int, int], Any] = {}
    for (a, b), v in d.items():
        aa, bb = int(a), int(b)
        if aa not in old_to_new or bb not in old_to_new:
            raise KeyError(f"Missing vertex in old_to_new for edge {(a, b)}")
        out[canon_edge(old_to_new[aa], old_to_new[bb])] = v
    return out
