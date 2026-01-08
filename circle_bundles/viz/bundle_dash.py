# viz/bundle_dash.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import socket
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

__all__ = [
    "BundleVizInputs",
    "find_free_port",
    "prepare_bundle_viz_inputs",
    "prepare_bundle_viz_inputs_from_bundle",
    "make_bundle_app",
    "run_bundle_app",
    "show_bundle_vis",          # NEW public general entrypoint
    "save_bundle_snapshot",     # NEW optional offline saving helper
]


# ----------------------------
# Data container
# ----------------------------

@dataclass
class BundleVizInputs:
    base_points: np.ndarray               # (m, d_base)
    data: np.ndarray                      # (m, d_data)
    dist_mat: np.ndarray                  # (m, m)
    colors: Optional[np.ndarray] = None   # (m,)
    densities: Optional[np.ndarray] = None  # (m,)
    landmark_masks: Optional[List[np.ndarray]] = None  # list of (m,) bool arrays
    sample_inds: Optional[np.ndarray] = None           # (m,) indices into original


# ----------------------------
# Helpers
# ----------------------------

def find_free_port() -> int:
    """Pick an available local port (best-effort)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _embed_base_points_pca(base_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      emb: (n,3) embedding (pads with zeros if base dim < 3)
      explained: cumulative explained variance ratio, length <=3
    """
    base_points = np.asarray(base_points)
    if base_points.ndim != 2:
        raise ValueError("base_points must be 2D (n_points, dim).")
    d = int(base_points.shape[1])
    if d <= 0:
        raise ValueError("base_points has zero columns.")

    pca = PCA(n_components=min(3, d))
    emb = pca.fit_transform(base_points)
    explained = np.cumsum(pca.explained_variance_ratio_)[: min(3, d)]

    if emb.shape[1] < 3:
        emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])), mode="constant")
    return emb, explained


def _normalize_to_unit_interval(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        return np.zeros_like(vals, dtype=float)
    return (vals - vmin) / (vmax - vmin)


def _normalize_landmarks(landmark_inds: Any, n: int) -> Optional[List[np.ndarray]]:
    """
    Accept:
      - None
      - ndarray (n,)
      - ndarray (k,n) or (n,k)
      - list/tuple of arrays (n,)
    Return: list of bool masks [(n,), ...] or None
    """
    if landmark_inds is None:
        return None

    if isinstance(landmark_inds, np.ndarray):
        arr = np.asarray(landmark_inds)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"landmark_inds length mismatch: expected {n}, got {arr.shape[0]}")
            return [arr.astype(bool)]
        if arr.ndim == 2:
            # allow (n,k) or (k,n); canonicalize to (k,n)
            if arr.shape[0] == n and arr.shape[1] != n:
                arr = arr.T
            if arr.shape[1] != n:
                raise ValueError(f"2D landmark_inds must have one axis length {n}; got {arr.shape}")
            return [arr[i].astype(bool) for i in range(arr.shape[0])]
        raise ValueError("landmark_inds ndarray must be 1D or 2D.")

    masks: List[np.ndarray] = []
    for m in landmark_inds:
        mm = np.asarray(m).astype(bool)
        if mm.shape != (n,):
            raise ValueError(f"Each landmark mask must be shape ({n},), got {mm.shape}")
        masks.append(mm)
    return masks


def _subset_landmarks(masks: Optional[List[np.ndarray]], sample_inds: np.ndarray) -> Optional[List[np.ndarray]]:
    if masks is None:
        return None
    return [m[sample_inds] for m in masks]


def _parse_click_index(clickData: Any) -> Optional[int]:
    """
    Dash clickData typically looks like:
      {"points":[{"pointIndex": i, ...}]}  (sometimes pointNumber)
    """
    if not clickData or not isinstance(clickData, dict):
        return None
    pts = clickData.get("points", None)
    if not pts or not isinstance(pts, list):
        return None
    pt0 = pts[0] if pts else None
    if not isinstance(pt0, dict):
        return None
    idx = pt0.get("pointIndex", pt0.get("pointNumber", None))
    if idx is None:
        return None
    try:
        return int(idx)
    except Exception:
        return None


def _call_get_dist_mat(get_dist_mat: Callable[..., np.ndarray], bp: np.ndarray, base_metric: Any) -> np.ndarray:
    """
    Try get_dist_mat(bp, metric=...) but fall back to get_dist_mat(bp)
    to support user-provided callables with simpler signatures.
    """
    try:
        return np.asarray(get_dist_mat(bp, metric=base_metric))
    except TypeError:
        return np.asarray(get_dist_mat(bp))


# ----------------------------
# Prep (pure)
# ----------------------------

def prepare_bundle_viz_inputs(
    *,
    base_points: np.ndarray,
    data: np.ndarray,
    get_dist_mat: Callable[..., np.ndarray],
    full_dist_mat: Optional[np.ndarray] = None,
    base_metric: Any = None,
    same_metric: bool = False,
    max_samples: int = 10_000,
    colors: Optional[np.ndarray] = None,
    densities: Optional[np.ndarray] = None,
    landmark_inds: Any = None,
    rng: Optional[np.random.Generator] = None,
) -> BundleVizInputs:
    """
    Produce a downsampled view of base_points/data plus a distance matrix.
    """
    base_points = np.asarray(base_points)
    data = np.asarray(data)

    if base_points.ndim != 2:
        raise ValueError("base_points must be 2D.")
    n = int(base_points.shape[0])
    if data.ndim != 2 or data.shape[0] != n:
        raise ValueError(f"data and base_points must align: data {data.shape} vs base {base_points.shape}")

    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[0] != n:
            raise ValueError("colors must have length n.")
    if densities is not None:
        densities = np.asarray(densities)
        if densities.shape[0] != n:
            raise ValueError("densities must have length n.")

    if rng is None:
        rng = np.random.default_rng()

    landmark_masks_full = _normalize_landmarks(landmark_inds, n)

    if n > int(max_samples):
        sample_inds = rng.choice(n, size=int(max_samples), replace=False)
    else:
        sample_inds = np.arange(n)

    bp = base_points[sample_inds]
    X = data[sample_inds]
    c = colors[sample_inds] if colors is not None else None
    d = densities[sample_inds] if densities is not None else None
    lm = _subset_landmarks(landmark_masks_full, sample_inds)

    m = int(bp.shape[0])

    # Reuse full dist mat only if it's truly "the same" and we kept all samples
    if full_dist_mat is not None and same_metric and m == n:
        dist_mat = np.asarray(full_dist_mat)
    else:
        dist_mat = _call_get_dist_mat(get_dist_mat, bp, base_metric)

    if dist_mat.shape != (m, m):
        raise ValueError(f"dist_mat must be (m,m). Got {dist_mat.shape} for m={m}")

    return BundleVizInputs(
        base_points=bp,
        data=X,
        dist_mat=dist_mat,
        colors=c,
        densities=d,
        landmark_masks=lm,
        sample_inds=sample_inds,
    )


def prepare_bundle_viz_inputs_from_bundle(
    bundle,
    *,
    get_dist_mat: Callable[..., np.ndarray],
    max_samples: int = 10_000,
    base_metric: Any = None,
    colors: Optional[np.ndarray] = None,
    densities: Optional[np.ndarray] = None,
    landmark_inds: Any = None,
    rng: Optional[np.random.Generator] = None,
) -> BundleVizInputs:
    """
    BundleResult-aware prep:
      - uses bundle.cover.base_points
      - uses bundle.data
      - uses bundle.cover.full_dist_mat if present
      - uses bundle.cover.metric as default base_metric
    """
    cover = bundle.cover
    base_points = getattr(cover, "base_points", None)
    if base_points is None:
        raise AttributeError("bundle.cover.base_points is missing (needed for show_bundle).")

    cover_metric = getattr(cover, "metric", None)
    if base_metric is None:
        base_metric = cover_metric
    if base_metric is None:
        base_metric = "euclidean"

    same_metric = (base_metric is cover_metric)
    full_dist_mat = getattr(cover, "full_dist_mat", None)

    return prepare_bundle_viz_inputs(
        base_points=np.asarray(base_points),
        data=np.asarray(bundle.data),
        get_dist_mat=get_dist_mat,
        full_dist_mat=full_dist_mat,
        base_metric=base_metric,
        same_metric=same_metric,
        max_samples=max_samples,
        colors=colors,
        densities=densities,
        landmark_inds=landmark_inds,
        rng=rng,
    )


# ----------------------------
# Pure figure construction (reused by Dash + snapshot saving)
# ----------------------------

def _make_figures(
    *,
    base_embedded: np.ndarray,          # (n,3)
    explained_variance: np.ndarray,     # (<=3,)
    data: np.ndarray,                  # (n, d_data)
    dist_mat: np.ndarray,              # (n,n)
    colors: Optional[np.ndarray],
    normalized_colors: Optional[np.ndarray],
    densities: Optional[np.ndarray],
    landmark_masks: Optional[List[np.ndarray]],
    selected_index: Optional[int],
    r: float,
    density_threshold: Optional[float] = None,
) -> Tuple[go.Figure, go.Figure, str, str]:
    n = int(base_embedded.shape[0])

    # --- base plot ---
    fig_base = go.Figure()
    fig_base.add_trace(
        go.Scatter3d(
            x=base_embedded[:, 0], y=base_embedded[:, 1], z=base_embedded[:, 2],
            mode="markers",
            marker=dict(size=2, color="lightgray", opacity=0.5),
            hoverinfo="none",
            name="Base Points",
        )
    )

    # --- data plot ---
    fig_data = go.Figure()
    variance_text = f"PCA Variance (Base): {np.round(explained_variance, 3)}"

    label = "Selected Point: (none)"

    if selected_index is not None and 0 <= selected_index < n:
        label = f"Selected Point ({selected_index})"
        nearby_indices = np.where(dist_mat[selected_index] < float(r))[0]

        if densities is not None and density_threshold is not None:
            keep = densities[nearby_indices] > float(density_threshold)
            filtered = nearby_indices[keep]
        else:
            filtered = nearby_indices

        # highlight base neighborhood
        fig_base.add_trace(
            go.Scatter3d(
                x=[base_embedded[selected_index, 0]],
                y=[base_embedded[selected_index, 1]],
                z=[base_embedded[selected_index, 2]],
                mode="markers",
                marker=dict(size=7, color="red", opacity=1.0),
                name="Selected",
                hoverinfo="none",
            )
        )
        fig_base.add_trace(
            go.Scatter3d(
                x=base_embedded[filtered, 0],
                y=base_embedded[filtered, 1],
                z=base_embedded[filtered, 2],
                mode="markers",
                marker=dict(size=3, color="blue", opacity=0.8),
                name="Neighbors",
                hoverinfo="none",
            )
        )

        nearby_data = data[filtered]
        if nearby_data.shape[0] > 1:
            pca_fiber = PCA(n_components=min(3, nearby_data.shape[1]))
            fiber_pca = pca_fiber.fit_transform(nearby_data)
            if fiber_pca.shape[1] < 3:
                fiber_pca = np.pad(fiber_pca, ((0, 0), (0, 3 - fiber_pca.shape[1])), mode="constant")

            fiber_var = np.cumsum(pca_fiber.explained_variance_ratio_)
            variance_text = f"PCA Variance (Fiber): {np.round(fiber_var, 3)}"

            if normalized_colors is not None and colors is not None:
                cvals = normalized_colors[filtered]
                orig = colors[filtered]
                nonzero = orig != 0

                if np.any(nonzero):
                    fig_data.add_trace(
                        go.Scatter3d(
                            x=fiber_pca[nonzero, 0], y=fiber_pca[nonzero, 1], z=fiber_pca[nonzero, 2],
                            mode="markers",
                            marker=dict(size=3, opacity=0.6, color=cvals[nonzero], colorscale="hsv", cmin=0, cmax=1),
                            name="Fiber (colored)",
                            hoverinfo="none",
                        )
                    )
                if np.any(~nonzero):
                    fig_data.add_trace(
                        go.Scatter3d(
                            x=fiber_pca[~nonzero, 0], y=fiber_pca[~nonzero, 1], z=fiber_pca[~nonzero, 2],
                            mode="markers",
                            marker=dict(size=3, opacity=0.5, color="gray"),
                            name="Fiber (zero)",
                            hoverinfo="none",
                        )
                    )
            else:
                fig_data.add_trace(
                    go.Scatter3d(
                        x=fiber_pca[:, 0], y=fiber_pca[:, 1], z=fiber_pca[:, 2],
                        mode="markers",
                        marker=dict(size=3, opacity=0.6, color="blue"),
                        name="Fiber",
                        hoverinfo="none",
                    )
                )

            if landmark_masks is not None:
                landmark_colors = ["orange", "green", "purple", "cyan", "magenta", "yellow"]
                for i, mask in enumerate(landmark_masks):
                    local = np.where(np.asarray(mask, bool)[filtered])[0]
                    if local.size:
                        fig_data.add_trace(
                            go.Scatter3d(
                                x=fiber_pca[local, 0], y=fiber_pca[local, 1], z=fiber_pca[local, 2],
                                mode="markers",
                                marker=dict(size=4, color=landmark_colors[i % len(landmark_colors)], opacity=0.9),
                                name=f"Landmarks {i+1}",
                                hoverinfo="none",
                            )
                        )

    fig_base.update_layout(
        title="Base Points",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=30, b=30),
        showlegend=False,
    )
    fig_data.update_layout(
        title="Fiber Data",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, t=30, b=30),
        showlegend=True,
    )

    return fig_base, fig_data, label, variance_text


# ----------------------------
# Dash app (thin wrapper)
# ----------------------------

def make_bundle_app(
    viz: BundleVizInputs,
    *,
    initial_r: float = 0.1,
    r_max: float = 2.0,
) -> dash.Dash:
    base_points = np.asarray(viz.base_points)
    data = np.asarray(viz.data)
    dist_mat = np.asarray(viz.dist_mat)
    colors = viz.colors
    densities = viz.densities
    landmark_masks = viz.landmark_masks

    n = int(base_points.shape[0])
    if data.shape[0] != n or dist_mat.shape != (n, n):
        raise ValueError("viz inputs misaligned.")

    base_embedded, explained_variance = _embed_base_points_pca(base_points)
    normalized_colors = _normalize_to_unit_interval(colors) if colors is not None else None

    app = dash.Dash(__name__)

    layout_children = [
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="base-plot",
                            style={"width": "100%", "height": "400px", "margin-bottom": "25px"},
                            config={"displayModeBar": True},
                        ),
                        html.Div(
                            id="selected-point",
                            style={"fontSize": 16, "marginTop": "10px", "textAlign": "center"},
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="data-plot",
                            style={"width": "100%", "height": "400px", "margin-bottom": "25px"},
                            config={"displayModeBar": True},
                        ),
                        html.Div(
                            id="pca-variance",
                            style={"fontSize": 16, "marginTop": "10px", "textAlign": "center"},
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block", "verticalAlign": "top"},
                ),
            ],
            style={"display": "flex", "width": "100%", "justify-content": "center"},
        ),
        html.Div(
            [
                dcc.Slider(
                    id="radius-slider",
                    min=0.01,
                    max=float(r_max),
                    step=0.01,
                    value=float(initial_r),
                    marks={0.01: "0.01", round(r_max / 2, 2): str(round(r_max / 2, 2)), r_max: str(r_max)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="drag",
                )
            ],
            style={"width": "80%", "margin": "auto", "margin-top": "20px"},
        ),
        html.Div(
            "Adjust the neighborhood radius (r):",
            style={"fontSize": 16, "marginTop": "-10px", "textAlign": "center"},
        ),
    ]

    if densities is not None:
        dmin = float(np.min(densities))
        dmax = float(np.max(densities))
        layout_children.append(
            html.Div(
                [
                    dcc.Slider(
                        id="density-slider",
                        min=dmin,
                        max=dmax,
                        step=0.01,
                        value=dmin,
                        marks={round(dmin, 2): str(round(dmin, 2)), round(dmax, 2): str(round(dmax, 2))},
                        tooltip={"placement": "bottom", "always_visible": True},
                        updatemode="drag",
                    ),
                    html.Div(
                        "Density threshold: show points with density > d",
                        style={"fontSize": 16, "textAlign": "center"},
                    ),
                ],
                style={"width": "80%", "margin": "auto", "margin-top": "20px"},
            )
        )

    app.layout = html.Div(layout_children, style={"margin": "auto", "maxWidth": "95vw"})

    callback_inputs = [Input("base-plot", "clickData"), Input("radius-slider", "value")]
    if densities is not None:
        callback_inputs.append(Input("density-slider", "value"))

    @app.callback(
        [Output("base-plot", "figure"), Output("data-plot", "figure"),
         Output("selected-point", "children"), Output("pca-variance", "children")],
        callback_inputs,
    )
    def update_figures(clickData, r, density_threshold=None):
        selected_index = _parse_click_index(clickData)

        fig_base, fig_data, label, variance_text = _make_figures(
            base_embedded=base_embedded,
            explained_variance=explained_variance,
            data=data,
            dist_mat=dist_mat,
            colors=colors,
            normalized_colors=normalized_colors,
            densities=densities,
            landmark_masks=landmark_masks,
            selected_index=selected_index,
            r=float(r),
            density_threshold=None if density_threshold is None else float(density_threshold),
        )
        return fig_base, fig_data, label, variance_text

    return app


def run_bundle_app(app: dash.Dash, *, port: Optional[int] = None, debug: bool = False):
    if port is None:
        port = find_free_port()
    app.run(debug=debug, use_reloader=False, port=int(port))


# ----------------------------
# Public general entrypoint
# ----------------------------

def show_bundle_vis(
    *,
    base_points: np.ndarray,
    data: np.ndarray,
    get_dist_mat: Optional[Callable[..., np.ndarray]] = None,
    full_dist_mat: Optional[np.ndarray] = None,
    base_metric: Any = None,
    same_metric: bool = False,
    initial_r: float = 0.1,
    r_max: float = 2.0,
    colors: Optional[np.ndarray] = None,
    densities: Optional[np.ndarray] = None,
    landmark_inds: Any = None,
    max_samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    port: Optional[int] = None,
    debug: bool = False,
) -> dash.Dash:
    """
    General interactive viewer for (data, base_points) where base_points live in some metric space.

    - Neighborhoods come from dist_mat computed on base_points (via get_dist_mat).
    - "Fiber Data" shows PCA of data restricted to the selected neighborhood.
    """
    if get_dist_mat is None:
        # lazy import: avoids circular imports and keeps viz import light
        from ..metrics import get_dist_mat as _get_dist_mat  # adjust path if needed
        get_dist_mat = _get_dist_mat

    viz = prepare_bundle_viz_inputs(
        base_points=np.asarray(base_points),
        data=np.asarray(data),
        get_dist_mat=get_dist_mat,
        full_dist_mat=full_dist_mat,
        base_metric=base_metric,
        same_metric=bool(same_metric),
        max_samples=int(max_samples),
        colors=colors,
        densities=densities,
        landmark_inds=landmark_inds,
        rng=rng,
    )
    app = make_bundle_app(viz, initial_r=float(initial_r), r_max=float(r_max))
    run_bundle_app(app, port=port, debug=debug)
    return app


# ----------------------------
# Snapshot saving (offline, no Dash required)
# ----------------------------

def save_bundle_snapshot(
    viz: BundleVizInputs,
    *,
    selected_index: int,
    r: float,
    density_threshold: Optional[float] = None,
    base_html: Optional[str] = None,
    data_html: Optional[str] = None,
    base_image: Optional[str] = None,
    data_image: Optional[str] = None,
) -> Tuple[go.Figure, go.Figure]:
    """
    Create the two figures for a given (selected_index, r, density_threshold) and optionally save.

    Notes on saving:
      - HTML always works: fig.write_html("file.html")
      - Static images (PNG/SVG/PDF) require the 'kaleido' package:
          pip install -U kaleido
        Then:
          fig.write_image("out.pdf")  # yes, PDF is supported

    Returns (fig_base, fig_data).
    """
    base_points = np.asarray(viz.base_points)
    data = np.asarray(viz.data)
    dist_mat = np.asarray(viz.dist_mat)
    colors = viz.colors
    densities = viz.densities
    landmark_masks = viz.landmark_masks

    base_embedded, explained_variance = _embed_base_points_pca(base_points)
    normalized_colors = _normalize_to_unit_interval(colors) if colors is not None else None

    fig_base, fig_data, _, _ = _make_figures(
        base_embedded=base_embedded,
        explained_variance=explained_variance,
        data=data,
        dist_mat=dist_mat,
        colors=colors,
        normalized_colors=normalized_colors,
        densities=densities,
        landmark_masks=landmark_masks,
        selected_index=int(selected_index),
        r=float(r),
        density_threshold=None if density_threshold is None else float(density_threshold),
    )

    if base_html is not None:
        fig_base.write_html(base_html)
    if data_html is not None:
        fig_data.write_html(data_html)

    if base_image is not None:
        fig_base.write_image(base_image)
    if data_image is not None:
        fig_data.write_image(data_image)

    return fig_base, fig_data
