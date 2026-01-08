# viz/nerve_circle.py
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt

from ..combinatorics import canon_edge  # project-wide canonical edge


# ============================================================
# Circle-layout nerve plot
# ============================================================

def show_circle_nerve(
    *,
    n_vertices: int,
    edges: list[tuple[int, int]],
    kept_edges: list[tuple[int, int]] | None = None,
    omega: dict[tuple[int, int], int] | None = None,
    weights: dict[tuple[int, int], float] | None = None,
    phi: dict[int, int] | None = None,
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
    # label offsets (relative to radius)
    omega_offset: float = 0.09,     # outside midpoint
    weights_offset: float = 0.09,   # inside midpoint
    phi_offset: float = 0.14,       # outside node
    title: str | None = None,
    ax=None,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Circle-layout nerve plot.

    Parameters
    ----------
    n_vertices:
        Number of cover sets / nerve vertices.
    edges:
        Full edge list (i,j) for the nerve.
    kept_edges:
        Subcomplex edges to highlight (thick). If None, all edges are kept.
    omega:
        Optional dict edge->int label (e.g., ±1) drawn outside midpoints.
    weights:
        Optional dict edge->float label drawn inside midpoints.
    phi:
        Optional dict vertex->int label drawn outside nodes.

    Notes
    -----
    - Canonicalizes edge keys via project-wide canon_edge(a,b).
    - Supports omega/weights dicts keyed by either orientation.
    """

    if n_vertices <= 0:
        raise ValueError("n_vertices must be positive.")

    # canonicalize edge lists
    E_all = sorted({canon_edge(int(i), int(j)) for (i, j) in edges if int(i) != int(j)})
    if kept_edges is None:
        E_keep = set(E_all)
    else:
        E_keep = {canon_edge(int(i), int(j)) for (i, j) in kept_edges if int(i) != int(j)}

    E_rem = [e for e in E_all if e not in E_keep]

    # canonicalize dict keys (if provided)
    def _canon_edge_dict(d):
        if d is None:
            return None
        out = {}
        for (i, j), v in d.items():
            ii, jj = int(i), int(j)
            if ii == jj:
                continue
            out[canon_edge(ii, jj)] = v
        return out

    omega = _canon_edge_dict(omega)
    weights = _canon_edge_dict(weights)

    # node positions on circle
    ang = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    P = np.c_[r * np.cos(ang), r * np.sin(ang)]  # (n,2)

    # set up axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # ---- draw edges ----
    def draw_edges(edgelist, *, color, lw, zorder):
        for (i, j) in edgelist:
            if not (0 <= i < n_vertices and 0 <= j < n_vertices):
                continue
            xs = [P[i, 0], P[j, 0]]
            ys = [P[i, 1], P[j, 1]]
            ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder)

    draw_edges(E_rem, color=removed_edge_color, lw=removed_edge_lw, zorder=1)
    draw_edges(sorted(E_keep), color=kept_edge_color, lw=kept_edge_lw, zorder=2)

    # ---- draw nodes ----
    ax.scatter(
        P[:, 0], P[:, 1],
        s=node_size,
        c=node_facecolor,
        edgecolors=node_edgecolor,
        zorder=3
    )

    # node labels U_i
    for i in range(n_vertices):
        ax.text(
            P[i, 0], P[i, 1],
            f"$U_{{{i+1}}}$",
            ha="center", va="center",
            fontsize=fontsize_node,
            color=node_label_color,
            fontweight="bold",
            zorder=4
        )

    # ---- omega labels (outside midpoints) ----
    if omega is not None:
        for (i, j) in E_all:
            if (i, j) not in omega:
                continue
            mid = 0.5 * (P[i] + P[j])
            norm = float(np.linalg.norm(mid))
            if norm == 0:
                continue
            u = mid / norm  # outward direction
            pos = mid + (omega_offset * r) * u
            ax.text(
                pos[0], pos[1],
                str(int(omega[(i, j)])),
                ha="center", va="center",
                fontsize=fontsize_omega,
                color=omega_color,
                zorder=5
            )

    # ---- phi labels (outside nodes) ----
    if phi is not None:
        for v, val in phi.items():
            v = int(v)
            if not (0 <= v < n_vertices):
                continue
            u = P[v] / (float(np.linalg.norm(P[v])) + 1e-12)
            pos = P[v] + (phi_offset * r) * u
            ax.text(
                pos[0], pos[1],
                str(int(val)),
                ha="center", va="center",
                fontsize=fontsize_phi,
                color=phi_color,
                zorder=6
            )

    # ---- weights labels (inside midpoints) ----
    if weights is not None:
        for (i, j) in E_all:
            if (i, j) not in weights:
                continue
            mid = 0.5 * (P[i] + P[j])
            norm = float(np.linalg.norm(mid))
            if norm == 0:
                continue
            u_in = -mid / norm  # inward direction
            pos = mid + (weights_offset * r) * u_in
            ax.text(
                pos[0], pos[1],
                f"{float(weights[(i, j)]):.3g}",
                ha="center", va="center",
                fontsize=fontsize_weights,
                color=weights_color,
                zorder=5
            )

    # ---- finalize ----
    ax.set_aspect("equal")
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
        ax.set_title(title, pad=18, y=1.03)  

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


# ============================================================
# Helpers (used by bundle wrappers)
# ============================================================

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
    """
    Check whether the undirected graph is a single cycle on all n vertices.
    Conditions:
      - connected
      - every vertex degree == 2
      - |E| == n
    """
    E = {canon_edge(a, b) for (a, b) in edges if int(a) != int(b)}
    if n == 0:
        return False, "n=0"
    if len(E) != n:
        return False, f"|E|={len(E)} != n={n} (cycle needs exactly n edges)"

    adj = _edges_to_adj(n, list(E))
    degs = [len(adj[i]) for i in range(n)]
    if any(d != 2 for d in degs):
        bad = [i for i, d in enumerate(degs) if d != 2]
        return False, f"not 2-regular (bad vertices: {bad[:10]}...)"

    # connectivity via DFS
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
    """
    Given a single cycle graph on {0..n-1}, return cyclic order of vertices.
    """
    E = {canon_edge(a, b) for (a, b) in edges if int(a) != int(b)}
    adj = _edges_to_adj(n, list(E))

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


def reindex_edges(edges: Optional[list[tuple[int, int]]], old_to_new: dict[int, int]) -> Optional[list[tuple[int, int]]]:
    if edges is None:
        return None
    out = []
    for (a, b) in edges:
        a2 = old_to_new[int(a)]
        b2 = old_to_new[int(b)]
        out.append(canon_edge(a2, b2))
    return out


def reindex_vertex_dict(d: dict[int, Any] | None, old_to_new: dict[int, int]) -> dict[int, Any] | None:
    if d is None:
        return None
    return {old_to_new[int(k)]: v for k, v in d.items()}


def reindex_edge_dict(
    d: dict[tuple[int, int], Any] | None,
    old_to_new: dict[int, int],
) -> dict[tuple[int, int], Any] | None:
    if d is None:
        return None
    out: dict[tuple[int, int], Any] = {}
    for (a, b), v in d.items():
        a2 = old_to_new[int(a)]
        b2 = old_to_new[int(b)]
        out[canon_edge(a2, b2)] = v
    return out
