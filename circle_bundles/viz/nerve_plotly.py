# circle_bundles/viz/nerve_plotly.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from ..nerve.combinatorics import canon_edge, canon_tri

Edge = Tuple[int, int]
Tri = Tuple[int, int, int]

__all__ = [
    "embed_landmarks",
    "make_nerve_figure",
    "nerve_plotly_from_U",
    "nerve_with_slider_from_U",
    # backwards compat
    "nerve_with_slider",
]


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


def _segments_from_edges(
    emb: np.ndarray,
    edges: Sequence[Edge],
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    ex: List[Optional[float]] = []
    ey: List[Optional[float]] = []
    ez: List[Optional[float]] = []
    n = int(emb.shape[0])

    for (i, j) in edges:
        i = int(i)
        j = int(j)
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
    # expected simplex keys as tuple-like (i,), (i,j), (i,j,k)
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

        # only highlight edges that are currently present after cutoff filtering
        H = [e for e in H if e in Eset]

        if H:
            hx, hy, hz = _segments_from_edges(emb, H)
            fig.add_trace(
                go.Scatter3d(
                    x=hx, y=hy, z=hz,
                    mode="lines",
                    line=dict(width=float(max(2 * edge_width, edge_width + 4)), color=str(highlight_color)),
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
                color=str(tri_color),
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
# Cover-free entry points (Bundle should use these)
# -----------------------------------------------------------------------------

def nerve_plotly_from_U(
    *,
    U: np.ndarray,
    landmarks: np.ndarray,
    edges: Sequence[Edge],
    triangles: Optional[Sequence[Tri]] = None,
    title: Optional[str] = None,
    show_labels: bool = True,
    show_axes: bool = False,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    cochains: Optional[List[Dict[Tuple[int, ...], object]]] = None,
    edge_weights: Optional[Mapping[Edge, float]] = None,
    edge_cutoff: Optional[float] = None,
    highlight_edges: Optional[Set[Edge]] = None,
    highlight_color: str = "red",
) -> go.Figure:
    """
    Static nerve plot that does NOT require a cover object.
    """
    _U = np.asarray(U, dtype=bool)
    if _U.ndim != 2:
        raise ValueError(f"U must be 2D (n_sets, n_samples). Got {_U.shape}.")

    return make_nerve_figure(
        landmarks=np.asarray(landmarks),
        edges=list(edges),
        triangles=list(triangles or []),
        show_labels=bool(show_labels),
        show_axes=bool(show_axes),
        tri_opacity=float(tri_opacity),
        tri_color=str(tri_color),
        edge_weights=edge_weights,
        edge_cutoff=edge_cutoff,
        highlight_edges=highlight_edges,
        highlight_color=str(highlight_color),
        cochains=cochains,
        title=title,
    )


def nerve_with_slider_from_U(
    *,
    U: np.ndarray,
    landmarks: np.ndarray,
    edges: Sequence[Edge],
    triangles: Sequence[Tri] | None = None,
    edge_weights: Mapping[Edge, float],
    show_labels: bool = True,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    show_axes: bool = False,
    highlight_edges: Optional[Set[Edge]] = None,
    highlight_color: str = "red",
    mark_cutoff: Optional[float] = None,   # e.g. max-trivial cutoff (a weight value)
    title: Optional[str] = None,
    show_title_value: bool = True,
    initial: Union[str, float] = "max",     # "max" | "min" | float (snap)
    show_jump: bool = True,                # if False, never show jump button
    jump_label: str = "Jump to max-trivial",
):
    """
    Notebook helper: slider over an edge weight cutoff, WITHOUT resetting the camera.
    Cover-free version used by Bundle.

    Returns:
      - if jump is shown: (jump_btn, slider, fig_widget)
      - else:            (slider, fig_widget)
    """
    from ipywidgets import widgets, VBox, HBox
    from IPython.display import display
    import plotly.graph_objects as go

    _U = np.asarray(U, dtype=bool)
    if _U.ndim != 2:
        raise ValueError(f"U must be 2D (n_sets, n_samples). Got {_U.shape}.")

    landmarks = np.asarray(landmarks)
    emb = embed_landmarks(landmarks)

    edges_all = _canon_edges(edges)
    tris_all = _canon_tris(triangles or [])

    ew = _canon_edge_weights(edge_weights)
    vals = sorted(set(float(v) for v in ew.values()))
    if not vals:
        fig = make_nerve_figure(
            landmarks=landmarks,
            edges=edges_all,
            triangles=tris_all,
            show_labels=show_labels,
            show_axes=show_axes,
            tri_opacity=tri_opacity,
            tri_color=tri_color,
            highlight_edges=highlight_edges,
            highlight_color=highlight_color,
            title=(title or "Nerve"),
        )
        fig.show()
        return fig

    base_title = title or "Nerve"

    # Discrete slider (filtration cutoffs)
    options = [(f"{v:.6g}", v) for v in vals]

    def _snap_to_nearest(x: float) -> float:
        return min(vals, key=lambda v: abs(v - float(x)))

    # pick initial value
    if isinstance(initial, str):
        if initial == "min":
            init_val = min(vals)
        elif initial == "max":
            init_val = max(vals)
        else:
            raise ValueError("initial must be 'min', 'max', or a float.")
    else:
        init_val = _snap_to_nearest(float(initial))

    # if mark_cutoff is provided, it takes precedence for initial snap
    if mark_cutoff is not None:
        init_val = _snap_to_nearest(float(mark_cutoff))

    slider = widgets.SelectionSlider(
        options=options,
        value=init_val,
        description="Max w:",
        orientation="horizontal",
        layout={"width": "100%"},
        continuous_update=False,
    )

    # --- Build the figure ONCE as a FigureWidget ---
    cutoff0 = float(slider.value)

    def _title_for_cutoff(cutoff: float) -> str:
        if show_title_value:
            return f"{base_title} (w ≤ {cutoff:.6g})"
        return base_title

    # initial filtered E,T
    E0 = _filter_edges_by_cutoff(list(edges_all), ew, cutoff0)
    E0set = set(E0)
    T0 = _filter_tris_by_edge_set(list(tris_all), E0set)

    fig = go.FigureWidget()

    # Trace 0: edges
    ex, ey, ez = _segments_from_edges(emb, E0)
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez,
        mode="lines",
        line=dict(width=3.0, color="black"),
        hoverinfo="none",
        showlegend=False,
    ))

    # Trace 1: highlighted edges overlay
    fig.add_trace(go.Scatter3d(
        x=[], y=[], z=[],
        mode="lines",
        line=dict(width=7.0, color=str(highlight_color)),
        hoverinfo="none",
        showlegend=False,
    ))

    # Trace 2: triangles (or placeholder)
    if T0:
        tri = np.asarray(T0, dtype=int)
        fig.add_trace(go.Mesh3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            i=tri[:, 0], j=tri[:, 1], k=tri[:, 2],
            opacity=float(tri_opacity),
            color=str(tri_color),
            hoverinfo="skip",
            showlegend=False,
        ))
    else:
        fig.add_trace(go.Mesh3d(
            x=[], y=[], z=[],
            i=[], j=[], k=[],
            opacity=0.0,
            showlegend=False,
        ))

    # Trace 3: nodes (+labels optional)
    n = int(emb.shape[0])
    if show_labels:
        fig.add_trace(go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="markers+text",
            text=[f"U{i}" for i in range(n)],
            textposition="top center",
            marker=dict(size=5, color="blue"),
            textfont=dict(size=16, color="blue"),
            showlegend=False,
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            mode="markers",
            marker=dict(size=5, color="blue"),
            showlegend=False,
        ))

    fig.update_layout(
        title=_title_for_cutoff(cutoff0),
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

    # canonical highlight edges once
    Hcanon = _canon_edges(highlight_edges or [])

    def _highlight_under_cutoff(Eset: Set[Edge]) -> List[Edge]:
        return [e for e in Hcanon if e in Eset]

    # set initial highlight
    H0 = _highlight_under_cutoff(E0set)
    hx, hy, hz = _segments_from_edges(emb, H0)
    fig.data[1].x = hx
    fig.data[1].y = hy
    fig.data[1].z = hz

    # Slider callback: update in-place (keeps camera)
    def update(_change=None):
        cutoff = float(slider.value)
        E = _filter_edges_by_cutoff(list(edges_all), ew, cutoff)
        Eset = set(E)
        T = _filter_tris_by_edge_set(list(tris_all), Eset)

        with fig.batch_update():
            # edges trace
            ex, ey, ez = _segments_from_edges(emb, E)
            fig.data[0].x = ex
            fig.data[0].y = ey
            fig.data[0].z = ez

            # highlight trace
            H = _highlight_under_cutoff(Eset)
            hx, hy, hz = _segments_from_edges(emb, H)
            fig.data[1].x = hx
            fig.data[1].y = hy
            fig.data[1].z = hz

            # triangles trace
            if T:
                tri = np.asarray(T, dtype=int)
                fig.data[2].x = emb[:, 0]
                fig.data[2].y = emb[:, 1]
                fig.data[2].z = emb[:, 2]
                fig.data[2].i = tri[:, 0]
                fig.data[2].j = tri[:, 1]
                fig.data[2].k = tri[:, 2]
                fig.data[2].opacity = float(tri_opacity)
                fig.data[2].color = str(tri_color)
            else:
                fig.data[2].x = []
                fig.data[2].y = []
                fig.data[2].z = []
                fig.data[2].i = []
                fig.data[2].j = []
                fig.data[2].k = []

            fig.layout.title = _title_for_cutoff(cutoff)

    slider.observe(update, names="value")
    update()

    # --- Jump button: ONLY create it if mark_cutoff is provided ---
    controls = [slider]

    if mark_cutoff is not None:
        jump_btn = widgets.Button(
            description="Jump to max-trivial",
            button_style="info",
            layout={"width": "220px"},
        )

        def _jump(_btn):
            m = float(mark_cutoff)
            nearest = min(vals, key=lambda v: abs(v - m))

            # If it's already at nearest, the observer might not fire; force update.
            slider.value = nearest
            update()

        jump_btn.on_click(_jump)
        controls = [jump_btn, slider]

    # Display: figure + controls
    display(VBox([fig, HBox(controls)]))

    # Return shape: if no jump button, return (None, slider, fig)
    if mark_cutoff is None:
        return (None, slider, fig)
    return (jump_btn, slider, fig)



# -----------------------------------------------------------------------------
# Backwards-compatible cover wrapper (optional)
# -----------------------------------------------------------------------------

def nerve_with_slider(
    *,
    cover: Any,
    edge_weights: Mapping[Edge, float],
    show_labels: bool = True,
    tri_opacity: float = 0.25,
    tri_color: str = "pink",
    show_axes: bool = False,
    highlight_edges: Optional[Set[Edge]] = None,
    highlight_color: str = "red",
    mark_cutoff: Optional[float] = None,
    title: Optional[str] = None,
    show_title_value: bool = True,
    initial: Union[str, float] = "max",
    show_jump: bool = True,
    jump_label: str = "Jump to max-trivial",
):
    """
    Backwards compatible wrapper for older cover-based notebooks.

    Internally calls the cover-free `nerve_with_slider_from_U`.
    """
    U = np.asarray(cover.U, dtype=bool)
    landmarks = np.asarray(getattr(cover, "landmarks"))
    edges = list(cover.nerve_edges())
    tris = list(cover.nerve_triangles())

    return nerve_with_slider_from_U(
        U=U,
        landmarks=landmarks,
        edges=edges,
        triangles=tris,
        edge_weights=edge_weights,
        show_labels=show_labels,
        tri_opacity=tri_opacity,
        tri_color=tri_color,
        show_axes=show_axes,
        highlight_edges=highlight_edges,
        highlight_color=highlight_color,
        mark_cutoff=mark_cutoff,
        title=title,
        show_title_value=show_title_value,
        initial=initial,
        show_jump=show_jump,
        jump_label=jump_label,
    )
