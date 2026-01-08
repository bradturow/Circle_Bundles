# viz/nerve_plotly.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ..combinatorics import canon_edge, canon_tri

Edge = Tuple[int, int]
Tri = Tuple[int, int, int]

from ipywidgets import widgets, VBox
from IPython.display import display


def embed_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Embed landmarks into R^3:
      - (n,) or (n,1): interpret as angles (radians) -> unit circle
      - (n,2): pad to (n,3)
      - (n,3): unchanged
      - (n,d>3): PCA -> (n,3)
    """
    L = np.asarray(landmarks)
    if L.ndim == 1:
        L = L.reshape(-1, 1)

    n, d = L.shape
    if d == 1:
        ang = L[:, 0].astype(float)
        return np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    if d == 2:
        return np.c_[L[:, 0], L[:, 1], np.zeros(n)]
    if d == 3:
        return L.astype(float)

    pca = PCA(n_components=3)
    return pca.fit_transform(L.astype(float))


def make_nerve_figure(
    *,
    landmarks: np.ndarray,
    edges: List[Edge],
    triangles: Optional[List[Tri]] = None,
    show_labels: bool = True,
    show_axes: bool = False,
    node_size: int = 5,
    edge_width: float = 3.0,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    edge_weights: Optional[Dict[Edge, float]] = None,
    edge_cutoff: Optional[float] = None,
    highlight_edges: Optional[Set[Edge]] = None,
    highlight_color: str = "red",
    cochains: Optional[List[Dict[Tuple[int, ...], object]]] = None,
    fontsize: int = 16,
    title: Optional[str] = None,
) -> go.Figure:
    if title is None:
        title = "Geometric Realization of the Nerve"

    emb = embed_landmarks(landmarks)
    n = emb.shape[0]

    # canonicalize inputs
    E = [canon_edge(*e) for e in edges]
    T = [canon_tri(*t) for t in (triangles or [])]

    # canonicalize weights
    ew = None
    if edge_weights is not None:
        ew = {canon_edge(*e): float(w) for e, w in edge_weights.items()}

    # filter edges by cutoff if requested
    if edge_cutoff is not None and ew is not None:
        cutoff = float(edge_cutoff)
        E = [e for e in E if float(ew.get(e, np.inf)) <= cutoff]

    # --- filter triangles to respect current edge set ---
    Eset = set(E)
    if T:
        def tri_is_present(t: Tri) -> bool:
            i, j, k = t
            return (
                canon_edge(i, j) in Eset and
                canon_edge(i, k) in Eset and
                canon_edge(j, k) in Eset
            )
        T = [t for t in T if tri_is_present(t)]
    # --------------------------------------------------------

    # edge segments
    ex: List[Optional[float]] = []
    ey: List[Optional[float]] = []
    ez: List[Optional[float]] = []
    for (i, j) in E:
        if not (0 <= i < n and 0 <= j < n):
            continue
        ex += [float(emb[i, 0]), float(emb[j, 0]), None]
        ey += [float(emb[i, 1]), float(emb[j, 1]), None]
        ez += [float(emb[i, 2]), float(emb[j, 2]), None]

    fig = go.Figure()

    # base edges
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(width=edge_width, color="black"),
        hoverinfo="none",
        showlegend=False,
    ))

    # highlighted edges overlay
    if highlight_edges:
        H = sorted({canon_edge(*e) for e in highlight_edges})
        hx: List[Optional[float]] = []
        hy: List[Optional[float]] = []
        hz: List[Optional[float]] = []
        for (i, j) in H:
            if not (0 <= i < n and 0 <= j < n):
                continue
            hx += [float(emb[i, 0]), float(emb[j, 0]), None]
            hy += [float(emb[i, 1]), float(emb[j, 1]), None]
            hz += [float(emb[i, 2]), float(emb[j, 2]), None]
        fig.add_trace(go.Scatter3d(
            x=hx, y=hy, z=hz,
            mode="lines",
            line=dict(width=max(2 * edge_width, edge_width + 4), color=highlight_color),
            hoverinfo="none",
            showlegend=False,
        ))

    # triangles
    if T:
        tri = np.asarray(T, dtype=int)
        fig.add_trace(go.Mesh3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            opacity=float(tri_opacity),
            color=tri_color,
            hoverinfo="skip",
            showlegend=False,
        ))

    # nodes
    if show_labels:
        fig.add_trace(go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="markers+text",
            text=[f"U{i}" for i in range(n)],
            textposition="top center",
            marker=dict(size=node_size, color="blue"),
            textfont=dict(size=16, color="blue"),
            showlegend=False,
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="markers",
            marker=dict(size=node_size, color="blue"),
            showlegend=False,
        ))

    # cochain overlays
    if cochains:
        for cochain in cochains:
            for sig, val in cochain.items():
                sig = tuple(sorted(map(int, sig)))
                if len(sig) == 1:
                    i = sig[0]
                    if 0 <= i < n:
                        fig.add_trace(go.Scatter3d(
                            x=[emb[i, 0]], y=[emb[i, 1]], z=[emb[i, 2]],
                            mode="text",
                            text=[str(val)],
                            textfont=dict(size=fontsize, color="red"),
                            showlegend=False,
                        ))
                elif len(sig) == 2:
                    i, j = canon_edge(sig[0], sig[1])
                    if 0 <= i < n and 0 <= j < n:
                        mid = emb[[i, j]].mean(axis=0)
                        fig.add_trace(go.Scatter3d(
                            x=[mid[0]], y=[mid[1]], z=[mid[2]],
                            mode="text",
                            text=[str(val)],
                            textfont=dict(size=fontsize, color="red"),
                            showlegend=False,
                        ))
                elif len(sig) == 3:
                    i, j, k = canon_tri(sig[0], sig[1], sig[2])
                    if 0 <= i < n and 0 <= j < n and 0 <= k < n:
                        mid = emb[[i, j, k]].mean(axis=0)
                        fig.add_trace(go.Scatter3d(
                            x=[mid[0]], y=[mid[1]], z=[mid[2]],
                            mode="text",
                            text=[str(val)],
                            textfont=dict(size=fontsize, color="green"),
                            showlegend=False,
                        ))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data"),
        showlegend=False,
    )
    if not show_axes:
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
            )
        )

    return fig



def nerve_with_slider(
    *,
    cover,
    edge_weights: dict[tuple[int,int], float],
    show_labels: bool = True,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    show_axes: bool = False,
):
    landmarks = np.asarray(cover.landmarks)
    edges = list(cover.nerve_edges())
    triangles = list(cover.nerve_triangles())

    ew = {canon_edge(*e): float(w) for e, w in edge_weights.items()}
    vals = sorted(set(ew.values()))
    if not vals:
        fig = make_nerve_figure(
            landmarks=landmarks,
            edges=edges,
            triangles=triangles,
            show_labels=show_labels,
            show_axes=show_axes,
            tri_opacity=tri_opacity,
            tri_color=tri_color,
            title="Nerve",
        )
        fig.show()
        return fig

    options = [(f"{v:.6g}", v) for v in vals]
    slider = widgets.SelectionSlider(
        options=options,
        value=max(vals),
        description="Max w:",
        orientation="horizontal",
        layout={"width": "100%"},
        continuous_update=False,
    )

    out = widgets.Output()

    def update(change=None):
        out.clear_output(wait=True)
        with out:
            fig = make_nerve_figure(
                landmarks=landmarks,
                edges=edges,
                triangles=triangles,
                show_labels=show_labels,
                show_axes=show_axes,
                tri_opacity=tri_opacity,
                tri_color=tri_color,
                edge_weights=ew,
                edge_cutoff=float(slider.value),
                title=f"Nerve (w ≤ {float(slider.value):.6g})",
            )
            fig.show()

    slider.observe(update, names="value")
    display(VBox([out, slider]))
    update()
    return slider  # or return (out, slider)
