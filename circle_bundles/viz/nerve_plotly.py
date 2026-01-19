# circle_bundles/viz/nerve_plotly.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ..nerve.combinatorics import canon_edge, canon_tri

Edge = Tuple[int, int]
Tri = Tuple[int, int, int]


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------

def embed_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Embed landmarks into R^3:
      - (n,) or (n,1): interpret as angles (radians) -> unit circle in xy-plane
      - (n,2): pad to (n,3) with z=0
      - (n,3): unchanged
      - (n,d>3): PCA -> (n,3)
    """
    L = np.asarray(landmarks)
    if L.ndim == 1:
        L = L.reshape(-1, 1)
    if L.ndim != 2:
        raise ValueError(f"landmarks must be 1D or 2D. Got shape {L.shape}.")

    n, d = L.shape
    if n == 0:
        raise ValueError("landmarks is empty.")

    if d == 1:
        ang = L[:, 0].astype(float)
        return np.c_[np.cos(ang), np.sin(ang), np.zeros_like(ang)]
    if d == 2:
        return np.c_[L[:, 0].astype(float), L[:, 1].astype(float), np.zeros(n, dtype=float)]
    if d == 3:
        return L.astype(float)

    pca = PCA(n_components=3)
    return pca.fit_transform(L.astype(float))


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _canon_edges(edges: Iterable[Edge]) -> List[Edge]:
    out: List[Edge] = []
    for (i, j) in edges:
        i, j = int(i), int(j)
        if i == j:
            continue
        out.append(canon_edge(i, j))
    return out


def _canon_tris(triangles: Iterable[Tri]) -> List[Tri]:
    out: List[Tri] = []
    for (i, j, k) in triangles:
        i, j, k = int(i), int(j), int(k)
        if len({i, j, k}) < 3:
            continue
        out.append(canon_tri(i, j, k))
    return out


def _canon_edge_weights(edge_weights: Mapping[Edge, float]) -> Dict[Edge, float]:
    return {canon_edge(int(i), int(j)): float(w) for (i, j), w in edge_weights.items()}


def _filter_edges_by_cutoff(E: List[Edge], ew: Optional[Dict[Edge, float]], cutoff: Optional[float]) -> List[Edge]:
    if cutoff is None or ew is None:
        return E
    c = float(cutoff)
    # if an edge is missing a weight, treat it as +inf and drop it under any finite cutoff
    return [e for e in E if float(ew.get(e, np.inf)) <= c]


def _filter_tris_by_edge_set(T: List[Tri], Eset: Set[Edge]) -> List[Tri]:
    if not T:
        return T

    keep: List[Tri] = []
    for (i, j, k) in T:
        if (
            canon_edge(i, j) in Eset and
            canon_edge(i, k) in Eset and
            canon_edge(j, k) in Eset
        ):
            keep.append((i, j, k))
    return keep


def _segments_from_edges(emb: np.ndarray, edges: Sequence[Edge]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    ex: List[Optional[float]] = []
    ey: List[Optional[float]] = []
    ez: List[Optional[float]] = []
    n = emb.shape[0]

    for (i, j) in edges:
        if not (0 <= i < n and 0 <= j < n):
            continue
        ex += [float(emb[i, 0]), float(emb[j, 0]), None]
        ey += [float(emb[i, 1]), float(emb[j, 1]), None]
        ez += [float(emb[i, 2]), float(emb[j, 2]), None]
    return ex, ey, ez


# -----------------------------------------------------------------------------
# Main plot
# -----------------------------------------------------------------------------

def make_nerve_figure(
    *,
    landmarks: np.ndarray,
    edges: Sequence[Edge],
    triangles: Optional[Sequence[Tri]] = None,
    show_labels: bool = True,
    show_axes: bool = False,
    node_size: int = 5,
    edge_width: float = 3.0,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    edge_weights: Optional[Mapping[Edge, float]] = None,
    edge_cutoff: Optional[float] = None,
    highlight_edges: Optional[Set[Edge]] = None,
    highlight_color: str = "red",
    # cochains: list of dicts mapping simplex->value
    # expected simplex keys as tuple-like (i,), (i,j), (i,j,k) OR ints for vertices
    cochains: Optional[List[Dict[Tuple[int, ...], object]]] = None,
    fontsize: int = 16,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Plotly 3D visualization of the nerve (edges + optional filled triangles + optional labels).

    Notes
    -----
    - Nodes are always drawn (so isolated vertices still appear).
    - If edge_cutoff is provided along with edge_weights, edges are filtered by weight.
    - Triangles are filtered to only include those whose 3 edges are currently present.
    - highlight_edges are drawn on top as a thicker colored overlay.
    """
    if title is None:
        title = "2-Skeleton Of The Nerve"

    emb = embed_landmarks(landmarks)
    n = int(emb.shape[0])

    # canonicalize inputs
    E = _canon_edges(edges)
    T = _canon_tris(triangles or [])

    ew = _canon_edge_weights(edge_weights) if edge_weights is not None else None

    # filter edges by cutoff if requested
    E = _filter_edges_by_cutoff(E, ew, edge_cutoff)

    # filter triangles to respect current edge set
    Eset = set(E)
    T = _filter_tris_by_edge_set(T, Eset)

    fig = go.Figure()

    # ---- edges (base) ----
    ex, ey, ez = _segments_from_edges(emb, E)
    fig.add_trace(
        go.Scatter3d(
            x=ex, y=ey, z=ez,
            mode="lines",
            line=dict(width=float(edge_width), color="black"),
            hoverinfo="none",
            showlegend=False,
        )
    )

    # ---- highlighted edges ----
    if highlight_edges:
        H = _canon_edges(highlight_edges)
        hx, hy, hz = _segments_from_edges(emb, H)
        fig.add_trace(
            go.Scatter3d(
                x=hx, y=hy, z=hz,
                mode="lines",
                line=dict(width=float(max(2 * edge_width, edge_width + 4)), color=highlight_color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # ---- triangles ----
    if T:
        tri = np.asarray(T, dtype=int)
        fig.add_trace(
            go.Mesh3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
                opacity=float(tri_opacity),
                color=tri_color,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # ---- nodes (always!) ----
    if show_labels:
        fig.add_trace(
            go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers+text",
                text=[f"U{i}" for i in range(n)],
                textposition="top center",
                marker=dict(size=int(node_size), color="blue"),
                textfont=dict(size=16, color="blue"),
                showlegend=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
                mode="markers",
                marker=dict(size=int(node_size), color="blue"),
                showlegend=False,
            )
        )

    # ---- cochain overlays (text at simplex barycenters) ----
    if cochains:
        for cochain in cochains:
            for sig, val in cochain.items():
                sig_t = tuple(sorted(map(int, sig)))
                if len(sig_t) == 1:
                    i = sig_t[0]
                    if 0 <= i < n:
                        fig.add_trace(go.Scatter3d(
                            x=[float(emb[i, 0])], y=[float(emb[i, 1])], z=[float(emb[i, 2])],
                            mode="text",
                            text=[str(val)],
                            textfont=dict(size=int(fontsize), color="red"),
                            showlegend=False,
                        ))
                elif len(sig_t) == 2:
                    i, j = canon_edge(sig_t[0], sig_t[1])
                    if 0 <= i < n and 0 <= j < n:
                        mid = emb[[i, j]].mean(axis=0)
                        fig.add_trace(go.Scatter3d(
                            x=[float(mid[0])], y=[float(mid[1])], z=[float(mid[2])],
                            mode="text",
                            text=[str(val)],
                            textfont=dict(size=int(fontsize), color="red"),
                            showlegend=False,
                        ))
                elif len(sig_t) == 3:
                    i, j, k = canon_tri(sig_t[0], sig_t[1], sig_t[2])
                    if 0 <= i < n and 0 <= j < n and 0 <= k < n:
                        mid = emb[[i, j, k]].mean(axis=0)
                        fig.add_trace(go.Scatter3d(
                            x=[float(mid[0])], y=[float(mid[1])], z=[float(mid[2])],
                            mode="text",
                            text=[str(val)],
                            textfont=dict(size=int(fontsize), color="green"),
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


# -----------------------------------------------------------------------------
# Notebook slider wrapper (imports are local so the module is import-safe)
# -----------------------------------------------------------------------------

def nerve_with_slider(
    *,
    cover,
    edge_weights: Mapping[Edge, float],
    show_labels: bool = True,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    show_axes: bool = False,
):
    """
    Notebook helper: slider over an edge weight cutoff.

    Returns:
      - If there are weights: the slider widget
      - Else: the plotted figure
    """
    # local imports so non-notebook usage doesn't require ipywidgets
    from ipywidgets import widgets, VBox
    from IPython.display import display

    landmarks = np.asarray(cover.landmarks)
    edges = list(cover.nerve_edges())
    triangles = list(cover.nerve_triangles())

    ew = _canon_edge_weights(edge_weights)
    vals = sorted(set(float(v) for v in ew.values()))

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

    # Keep your SelectionSlider behavior (stable, discrete cutoffs)
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

    def update(_change=None):
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

    return slider
