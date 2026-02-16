# circle_bundles/viz/bundle_dash.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import io
import socket
import numpy as np

__all__ = [
    "BundleVizInputs",
    "find_free_port",
    "prepare_bundle_viz_inputs",
    "prepare_bundle_viz_inputs_from_bundle",
    "make_bundle_app",
    "run_bundle_app",
    "show_bundle_vis",
    "save_bundle_snapshot",
]


# ----------------------------
# Data container
# ----------------------------

@dataclass
class BundleVizInputs:
    base_points: np.ndarray                 # (m, d_base)
    data: np.ndarray                        # (m, d_data)
    dist_mat: np.ndarray                    # (m, m)
    colors: Optional[np.ndarray] = None     # (m,)
    densities: Optional[np.ndarray] = None  # (m,)

    # Total-space / fiber landmarks: list of (m,) bool masks over *downsampled* points
    data_landmark_masks: Optional[List[np.ndarray]] = None

    # Base landmarks:
    # - base_landmark_masks: list of (m,) bool masks over *downsampled* points (subset of base_points)
    # - base_landmark_points: list of (L_i, d_base) arrays (extra points not necessarily in dataset)
    base_landmark_masks: Optional[List[np.ndarray]] = None
    base_landmark_points: Optional[List[np.ndarray]] = None

    sample_inds: Optional[np.ndarray] = None           # (m,) indices into original


# ----------------------------
# Helpers
# ----------------------------

def find_free_port() -> int:
    """Pick an available local port (best-effort)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def _embed_base_points_pca(base_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Return:
      emb: (n,3) embedding (pads with zeros if base dim < 3)
      explained: cumulative explained variance ratio, length <=3
      pca: fitted PCA object (so we can transform landmark points consistently)

    Notes
    -----
    - Imports scikit-learn lazily (only when this is called).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "bundle_dash PCA embedding requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e

    base_points = np.asarray(base_points)
    if base_points.ndim != 2:
        raise ValueError("base_points must be 2D (n_points, dim).")
    d = int(base_points.shape[1])
    if d <= 0:
        raise ValueError("base_points has zero columns.")

    pca = PCA(n_components=min(3, d))
    emb = pca.fit_transform(base_points.astype(float))
    explained = np.cumsum(pca.explained_variance_ratio_)[: min(3, d)]

    if emb.shape[1] < 3:
        emb = np.pad(emb, ((0, 0), (0, 3 - emb.shape[1])), mode="constant")
    return emb, explained, pca


def _normalize_to_unit_interval(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if vmax <= vmin:
        return np.zeros_like(vals, dtype=float)
    return (vals - vmin) / (vmax - vmin)


def _normalize_bool_masks_any(inds_or_masks: Any, n: int, *, name: str) -> Optional[List[np.ndarray]]:
    """
    Accept:
      - None
      - ndarray (n,) bool-ish
      - ndarray (k,n) or (n,k) bool-ish
      - list/tuple of arrays (n,)
    Return: list of bool masks [(n,), ...] or None
    """
    if inds_or_masks is None:
        return None

    if isinstance(inds_or_masks, np.ndarray):
        arr = np.asarray(inds_or_masks)
        if arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} length mismatch: expected {n}, got {arr.shape[0]}")
            return [arr.astype(bool)]
        if arr.ndim == 2:
            # allow (n,k) or (k,n); canonicalize to (k,n)
            if arr.shape[0] == n and arr.shape[1] != n:
                arr = arr.T
            if arr.shape[1] != n:
                raise ValueError(f"2D {name} must have one axis length {n}; got {arr.shape}")
            return [arr[i].astype(bool) for i in range(arr.shape[0])]
        raise ValueError(f"{name} ndarray must be 1D or 2D.")

    masks: List[np.ndarray] = []
    for m in inds_or_masks:
        mm = np.asarray(m).astype(bool)
        if mm.shape != (n,):
            raise ValueError(f"Each {name} mask must be shape ({n},), got {mm.shape}")
        masks.append(mm)
    return masks


def _subset_masks(masks: Optional[List[np.ndarray]], sample_inds: np.ndarray) -> Optional[List[np.ndarray]]:
    if masks is None:
        return None
    return [np.asarray(m, bool)[sample_inds] for m in masks]


def _normalize_base_landmarks(
    landmarks: Any,
    *,
    base_points: np.ndarray,
    name: str = "landmarks",
) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Base-landmarks input normalization.

    Accepts:
      - None
      - points: ndarray (L, d_base)
      - mask:   ndarray (n,) bool
      - inds:   ndarray (L,) int
      - list/tuple of any mixture of the above (each element is one "group")

    Returns:
      (base_landmark_masks_full, base_landmark_points_full)
        - base_landmark_masks_full: list of (n,) bool masks selecting points in base_points
        - base_landmark_points_full: list of (L_i, d_base) arrays (extra points)
    """
    if landmarks is None:
        return None, None

    bp = np.asarray(base_points)
    if bp.ndim != 2:
        raise ValueError("base_points must be 2D.")
    n, d_base = int(bp.shape[0]), int(bp.shape[1])

    def _one(obj: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if obj is None:
            return None, None

        arr = np.asarray(obj)

        # bool mask
        if arr.dtype == bool and arr.ndim == 1:
            if arr.shape[0] != n:
                raise ValueError(f"{name} bool mask length mismatch: expected {n}, got {arr.shape[0]}")
            return arr.astype(bool), None

        # index array
        if np.issubdtype(arr.dtype, np.integer) and arr.ndim == 1:
            idx = arr.astype(int).reshape(-1)
            if idx.size == 0:
                return np.zeros((n,), dtype=bool), None
            if np.any(idx < 0) or np.any(idx >= n):
                raise ValueError(f"{name} index out of bounds for n={n}.")
            mask = np.zeros((n,), dtype=bool)
            mask[idx] = True
            return mask, None

        # points array
        if arr.ndim == 2 and int(arr.shape[1]) == d_base:
            pts = np.asarray(arr, dtype=float)
            return None, pts

        raise ValueError(
            f"{name} entries must be bool mask (n,), int indices (L,), or points (L,d_base={d_base}). "
            f"Got array with shape {arr.shape} dtype {arr.dtype}."
        )

    masks: List[np.ndarray] = []
    pts_groups: List[np.ndarray] = []

    if isinstance(landmarks, (list, tuple)):
        for g in landmarks:
            m, p = _one(g)
            if m is not None:
                masks.append(m.astype(bool))
            if p is not None:
                pts_groups.append(np.asarray(p, dtype=float))
    else:
        m, p = _one(landmarks)
        if m is not None:
            masks.append(m.astype(bool))
        if p is not None:
            pts_groups.append(np.asarray(p, dtype=float))

    return (masks if masks else None), (pts_groups if pts_groups else None)


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


def _call_get_dist_mat(
    get_dist_mat: Optional[Callable[..., np.ndarray]],
    bp: np.ndarray,
    base_metric: Any,
) -> np.ndarray:
    """
    Priority:
      1) If base_metric has .pairwise, use base_metric.pairwise(bp) directly.
      2) Else if get_dist_mat provided, try get_dist_mat(bp, metric=base_metric) then fallback get_dist_mat(bp)
      3) Else fallback to circle_bundles.metrics.get_dist_mat(bp, metric=base_metric)
    """
    # 1) Metric object fast path
    if base_metric is not None and hasattr(base_metric, "pairwise"):
        return np.asarray(base_metric.pairwise(bp))

    # 2) User-provided callable
    if get_dist_mat is not None:
        try:
            return np.asarray(get_dist_mat(bp, metric=base_metric))
        except TypeError:
            return np.asarray(get_dist_mat(bp))

    # 3) Library fallback
    from ..metrics import get_dist_mat as _get_dist_mat
    return np.asarray(_get_dist_mat(bp, metric=base_metric))


def _fig_to_png_bytes(fig, *, scale: int = 1) -> bytes:
    """
    Convert a plotly figure to PNG bytes (requires kaleido).
    - scale=1 is much faster than scale=2 for 3D.
    """
    try:
        import plotly.io as pio
        # Avoid any attempt to fetch mathjax in headless chrome (can stall on some setups)
        pio.defaults.mathjax = None
        return fig.to_image(format="png", engine="kaleido", scale=int(scale))
    except Exception as e:
        msg = str(e)
        if "not compatible with this version of Kaleido" in msg or "compatible with this version of Kaleido" in msg:
            raise RuntimeError(
                "Plotly/Kaleido version mismatch.\n"
                "Fix by either:\n"
                "  pip install -U 'plotly>=6.1.1'   (recommended)\n"
                "or\n"
                "  pip install -U 'kaleido==0.2.1'  (for Plotly 5.x)\n"
            ) from e
        if "requires the kaleido package" in msg.lower():
            raise RuntimeError("Static export requires kaleido. Install with `pip install -U kaleido`.") from e
        raise


def _combine_pngs_to_pdf_bytes(
    png_left: bytes,
    png_right: bytes,
    *,
    gap: int = 24,   # pixels between panels
    pad: int = 0,    # outer padding
) -> bytes:
    """
    Combine two PNGs side-by-side into ONE tight PDF (no extra whitespace).
    Requires pillow: pip install pillow
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Combining snapshots requires pillow. Install with `pip install pillow`.") from e

    L = Image.open(io.BytesIO(png_left)).convert("RGB")
    R = Image.open(io.BytesIO(png_right)).convert("RGB")

    W = L.width + int(gap) + R.width + 2 * int(pad)
    H = max(L.height, R.height) + 2 * int(pad)
    out = Image.new("RGB", (W, H), (255, 255, 255))

    yL = int(pad) + (H - 2 * int(pad) - L.height) // 2
    yR = int(pad) + (H - 2 * int(pad) - R.height) // 2
    out.paste(L, (int(pad), yL))
    out.paste(R, (int(pad) + L.width + int(gap), yR))

    buf = io.BytesIO()
    out.save(buf, format="PDF")  # tight to the composed image bounds
    return buf.getvalue()


# ----------------------------
# Prep (pure)
# ----------------------------

def prepare_bundle_viz_inputs(
    *,
    base_points: np.ndarray,
    data: np.ndarray,
    get_dist_mat: Optional[Callable[..., np.ndarray]] = None,
    full_dist_mat: Optional[np.ndarray] = None,
    base_metric: Any = None,
    same_metric: bool = False,
    max_samples: int = 10_000,
    colors: Optional[np.ndarray] = None,
    densities: Optional[np.ndarray] = None,
    data_landmark_inds: Any = None,
    landmarks: Any = None,
    rng: Optional[np.random.Generator] = None,
) -> BundleVizInputs:
    """
    Produce a downsampled view of base_points/data plus a distance matrix.

    Landmarks
    ---------
    - data_landmark_inds: masks over points (n,) or list of such, used to highlight points in the *fiber* PCA panel.
    - landmarks: base-landmarks to highlight in the *base* panel.

    If full_dist_mat is provided and same_metric=True, we will:
      - use it directly if no downsampling happened
      - otherwise subset it via dist_mat = full_dist_mat[np.ix_(sample_inds, sample_inds)]
    """
    base_points = np.asarray(base_points)
    data = np.asarray(data)

    if base_points.ndim == 1:
        base_points = base_points.reshape(-1, 1)

    if base_points.ndim != 2:
        raise ValueError("base_points must be 2D.")
    n = int(base_points.shape[0])
    if data.ndim != 2 or data.shape[0] != n:
        raise ValueError(f"data and base_points must align: data {data.shape} vs base {base_points.shape}")

    if colors is not None:
        colors = np.asarray(colors).reshape(-1)
        if colors.shape[0] != n:
            raise ValueError("colors must have length n.")
    if densities is not None:
        densities = np.asarray(densities).reshape(-1)
        if densities.shape[0] != n:
            raise ValueError("densities must have length n.")

    if rng is None:
        rng = np.random.default_rng()

    # total-space landmark masks (over full n)
    data_landmark_masks_full = _normalize_bool_masks_any(data_landmark_inds, n, name="data_landmark_inds")

    # base landmarks (over full n masks and/or extra points)
    base_landmark_masks_full, base_landmark_points_full = _normalize_base_landmarks(
        landmarks, base_points=base_points, name="landmarks"
    )

    if n > int(max_samples):
        sample_inds = rng.choice(n, size=int(max_samples), replace=False)
        sample_inds.sort()  # stable order helps toggling / reproducibility
    else:
        sample_inds = np.arange(n, dtype=int)

    bp = base_points[sample_inds]
    X = data[sample_inds]
    c = colors[sample_inds] if colors is not None else None
    d = densities[sample_inds] if densities is not None else None

    data_lm = _subset_masks(data_landmark_masks_full, sample_inds)
    base_lm_masks = _subset_masks(base_landmark_masks_full, sample_inds)

    m = int(bp.shape[0])

    # Use / subset full dist matrix when available
    if full_dist_mat is not None and bool(same_metric):
        full_dist_mat = np.asarray(full_dist_mat)
        if full_dist_mat.shape != (n, n):
            raise ValueError(f"full_dist_mat must be (n,n) with n={n}. Got {full_dist_mat.shape}.")
        dist_mat = full_dist_mat[np.ix_(sample_inds, sample_inds)]
    else:
        dist_mat = _call_get_dist_mat(get_dist_mat, bp, base_metric)

    dist_mat = np.asarray(dist_mat)
    if dist_mat.shape != (m, m):
        raise ValueError(f"dist_mat must be (m,m). Got {dist_mat.shape} for m={m}")

    # extra base landmark points (not necessarily in the dataset)
    base_lm_pts: Optional[List[np.ndarray]] = None
    if base_landmark_points_full is not None:
        d_base = int(base_points.shape[1])
        base_lm_pts = []
        for pts in base_landmark_points_full:
            pts = np.asarray(pts, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != d_base:
                raise ValueError(f"Each landmarks points group must be (L_i, {d_base}), got {pts.shape}")
            base_lm_pts.append(pts)

    return BundleVizInputs(
        base_points=bp,
        data=X,
        dist_mat=dist_mat,
        colors=c,
        densities=d,
        data_landmark_masks=data_lm,
        base_landmark_masks=base_lm_masks,
        base_landmark_points=base_lm_pts,
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
    data_landmark_inds: Any = None,
    landmarks: Any = None,
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
        from ..metrics import EuclideanMetric
        base_metric = EuclideanMetric()

    same_metric = (base_metric is cover_metric)
    full_dist_mat = getattr(cover, "full_dist_mat", None)

    return prepare_bundle_viz_inputs(
        base_points=np.asarray(base_points),
        data=np.asarray(bundle.data),
        get_dist_mat=get_dist_mat,
        full_dist_mat=full_dist_mat,
        base_metric=base_metric,
        same_metric=same_metric,
        max_samples=int(max_samples),
        colors=colors,
        densities=densities,
        data_landmark_inds=data_landmark_inds,
        landmarks=landmarks,
        rng=rng,
    )


# ----------------------------
# Pure figure construction (reused by Dash + snapshot saving)
# ----------------------------

def _make_figures(
    *,
    base_embedded: np.ndarray,                        # (n,3)
    explained_variance: np.ndarray,                   # (<=3,)
    base_landmark_masks: Optional[List[np.ndarray]],  # list of (n,) bool masks (subset of base points)
    base_landmarks_embedded: Optional[List[np.ndarray]],  # list of (L_i,3) extra points
    data: np.ndarray,                                # (n, d_data)
    dist_mat: np.ndarray,                            # (n,n)
    colors: Optional[np.ndarray],
    normalized_colors: Optional[np.ndarray],
    densities: Optional[np.ndarray],
    data_landmark_masks: Optional[List[np.ndarray]],  # list of (n,) bool masks (for fiber panel)
    selected_index: Optional[int],
    r: float,
    density_threshold: Optional[float] = None,
) -> Tuple["go.Figure", "go.Figure", str, str]:
    """
    Returns (fig_base, fig_data, label, variance_text).

    Notes
    -----
    - Imports plotly + sklearn lazily (only when called).
    """
    try:
        import plotly.graph_objects as go
    except ImportError as e:
        raise ImportError(
            "bundle_dash figure construction requires plotly. Install with `pip install plotly`."
        ) from e

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "bundle_dash fiber PCA requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e

    n = int(base_embedded.shape[0])

    # Marker sizes (priority: red > landmark > blue > gray)
    size_gray = 2
    size_blue = 3
    size_landmark = 5
    size_red = 7

    # --- base plot ---
    fig_base = go.Figure()
    fig_base.add_trace(
        go.Scatter3d(
            x=base_embedded[:, 0], y=base_embedded[:, 1], z=base_embedded[:, 2],
            mode="markers",
            marker=dict(size=size_gray, color="lightgray", opacity=0.5),
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
            keep = np.asarray(densities[nearby_indices] > float(density_threshold), dtype=bool)
            filtered = nearby_indices[keep]
        else:
            filtered = nearby_indices

        # Base plot: neighbors first (blue), then landmarks, then selected (red) last.
        if filtered.size:
            fig_base.add_trace(
                go.Scatter3d(
                    x=base_embedded[filtered, 0],
                    y=base_embedded[filtered, 1],
                    z=base_embedded[filtered, 2],
                    mode="markers",
                    marker=dict(size=size_blue, color="blue", opacity=0.8),
                    name="Neighbors",
                    hoverinfo="none",
                )
            )

        # Base plot: dataset landmarks (masks over base points)
        if base_landmark_masks is not None:
            lm_colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "black"]
            for i, mask in enumerate(base_landmark_masks):
                mask = np.asarray(mask, bool)
                idx = np.where(mask)[0]
                if idx.size:
                    fig_base.add_trace(
                        go.Scatter3d(
                            x=base_embedded[idx, 0],
                            y=base_embedded[idx, 1],
                            z=base_embedded[idx, 2],
                            mode="markers",
                            marker=dict(size=size_landmark, color=lm_colors[i % len(lm_colors)], opacity=0.95),
                            name=f"Landmarks {i+1}",
                            hoverinfo="none",
                        )
                    )

        # Base plot: extra landmark points (not necessarily in dataset)
        if base_landmarks_embedded is not None:
            lm_colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "black"]
            for i, L3 in enumerate(base_landmarks_embedded):
                L3 = np.asarray(L3, dtype=float)
                if L3.size == 0:
                    continue
                fig_base.add_trace(
                    go.Scatter3d(
                        x=L3[:, 0], y=L3[:, 1], z=L3[:, 2],
                        mode="markers",
                        marker=dict(size=size_landmark, color=lm_colors[i % len(lm_colors)], opacity=0.95),
                        name=f"Landmarks (extra) {i+1}",
                        hoverinfo="none",
                    )
                )

        # Base plot: selected point last (red)
        fig_base.add_trace(
            go.Scatter3d(
                x=[base_embedded[selected_index, 0]],
                y=[base_embedded[selected_index, 1]],
                z=[base_embedded[selected_index, 2]],
                mode="markers",
                marker=dict(size=size_red, color="red", opacity=1.0),
                name="Selected",
                hoverinfo="none",
            )
        )

        # Fiber PCA
        nearby_data = data[filtered] if filtered.size else np.zeros((0, data.shape[1]), dtype=float)

        if nearby_data.shape[0] >= 2:
            pca_fiber = PCA(n_components=min(3, nearby_data.shape[1]))
            fiber_pca = pca_fiber.fit_transform(nearby_data)
            if fiber_pca.shape[1] < 3:
                fiber_pca = np.pad(fiber_pca, ((0, 0), (0, 3 - fiber_pca.shape[1])), mode="constant")

            fiber_var = np.cumsum(pca_fiber.explained_variance_ratio_)
            variance_text = f"PCA Variance (Fiber): {np.round(fiber_var, 3)}"

            if normalized_colors is not None and colors is not None:
                cvals = normalized_colors[filtered]
                orig = colors[filtered]
                nonzero = np.asarray(orig != 0, dtype=bool)

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

            # Total-space landmarks in the fiber panel (masks are over downsampled indexing)
            if data_landmark_masks is not None:
                lm_colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "black"]
                for i, mask in enumerate(data_landmark_masks):
                    mask = np.asarray(mask, bool)
                    local = np.where(mask[filtered])[0]
                    if local.size:
                        fig_data.add_trace(
                            go.Scatter3d(
                                x=fiber_pca[local, 0], y=fiber_pca[local, 1], z=fiber_pca[local, 2],
                                mode="markers",
                                marker=dict(size=4, color=lm_colors[i % len(lm_colors)], opacity=0.9),
                                name=f"Data Landmarks {i+1}",
                                hoverinfo="none",
                            )
                        )
        elif nearby_data.shape[0] == 1:
            fig_data.add_trace(
                go.Scatter3d(
                    x=[0.0], y=[0.0], z=[0.0],
                    mode="markers",
                    marker=dict(size=5, opacity=0.9, color="blue"),
                    name="Fiber (1 point)",
                    hoverinfo="none",
                )
            )
            variance_text = "PCA Variance (Fiber): (neighborhood has 1 point)"
        else:
            variance_text = "PCA Variance (Fiber): (empty neighborhood)"

    else:
        # Even with no selection, still show base landmarks (so you can orient yourself)
        if base_landmark_masks is not None:
            lm_colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "black"]
            for i, mask in enumerate(base_landmark_masks):
                mask = np.asarray(mask, bool)
                idx = np.where(mask)[0]
                if idx.size:
                    fig_base.add_trace(
                        go.Scatter3d(
                            x=base_embedded[idx, 0],
                            y=base_embedded[idx, 1],
                            z=base_embedded[idx, 2],
                            mode="markers",
                            marker=dict(size=size_landmark, color=lm_colors[i % len(lm_colors)], opacity=0.95),
                            name=f"Landmarks {i+1}",
                            hoverinfo="none",
                        )
                    )

        if base_landmarks_embedded is not None:
            lm_colors = ["orange", "green", "purple", "cyan", "magenta", "yellow", "black"]
            for i, L3 in enumerate(base_landmarks_embedded):
                L3 = np.asarray(L3, dtype=float)
                if L3.size == 0:
                    continue
                fig_base.add_trace(
                    go.Scatter3d(
                        x=L3[:, 0], y=L3[:, 1], z=L3[:, 2],
                        mode="markers",
                        marker=dict(size=size_landmark, color=lm_colors[i % len(lm_colors)], opacity=0.95),
                        name=f"Landmarks (extra) {i+1}",
                        hoverinfo="none",
                    )
                )

    # Show the 3D axes box/grid/backdrop, but hide tick labels (numbers).
    _axis_style = dict(
        showbackground=True,
        showgrid=True,
        zeroline=False,
        showticklabels=False,
        title="",  # no axis label text
    )
    _scene_style = dict(
        xaxis=_axis_style,
        yaxis=_axis_style,
        zaxis=_axis_style,
    )

    fig_base.update_layout(
        title="Base Points",
        scene=_scene_style,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        uirevision="bundle-viewer",  # preserve camera between updates
        width=650,
        height=450,
    )
    fig_data.update_layout(
        title="Fiber Data",
        scene=_scene_style,
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        uirevision="bundle-viewer",  # preserve camera between updates
        width=650,
        height=450,
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
):
    """
    Build the Dash app.

    Notes
    -----
    - Imports dash lazily (only when this is called).
    - Plotly/sklearn are also lazy via helper functions called here.
    """
    try:
        import dash
        from dash import dcc, html
        from dash.dependencies import Input, Output, State
    except ImportError as e:
        raise ImportError(
            "make_bundle_app requires dash. Install with `pip install dash`."
        ) from e

    base_points = np.asarray(viz.base_points)
    data = np.asarray(viz.data)
    dist_mat = np.asarray(viz.dist_mat)
    colors = viz.colors
    densities = viz.densities

    data_landmark_masks = viz.data_landmark_masks
    base_landmark_masks = viz.base_landmark_masks
    base_landmark_points = viz.base_landmark_points

    n = int(base_points.shape[0])
    if data.shape[0] != n or dist_mat.shape != (n, n):
        raise ValueError("viz inputs misaligned.")

    base_embedded, explained_variance, pca_base = _embed_base_points_pca(base_points)
    normalized_colors = _normalize_to_unit_interval(colors) if colors is not None else None

    base_landmarks_embedded: Optional[List[np.ndarray]] = None
    if base_landmark_points is not None:
        base_landmarks_embedded = []
        for g in base_landmark_points:
            g = np.asarray(g, dtype=float)
            L = pca_base.transform(g)
            if L.shape[1] < 3:
                L = np.pad(L, ((0, 0), (0, 3 - L.shape[1])), mode="constant")
            base_landmarks_embedded.append(L)

    app = dash.Dash(__name__)

    layout_children = [
        # persistent selection + download
        dcc.Store(id="selected-index-store", data=None),
        dcc.Download(id="download-pdf"),
        html.Div(
            [
                html.Button(
                    "Save snapshot (PDF)",
                    id="save-snapshot-btn",
                    n_clicks=0,
                    style={"fontSize": 14, "padding": "8px 14px"},
                ),
            ],
            style={"textAlign": "center", "marginTop": "8px", "marginBottom": "8px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(
                            id="base-plot",
                            style={"width": "100%", "height": "400px", "margin-bottom": "25px"},
                            config={"displayModeBar": True},
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
                ],
                style={"width": "80%", "margin": "auto", "margin-top": "20px"},
            )
        )

    app.layout = html.Div(layout_children, style={"margin": "auto", "maxWidth": "95vw"})

    # Persist selection
    @app.callback(
        Output("selected-index-store", "data"),
        Input("base-plot", "clickData"),
        State("selected-index-store", "data"),
    )
    def _remember_selected_index(clickData, current):
        idx = _parse_click_index(clickData)
        return current if idx is None else int(idx)

    # Update figures (use stored selection)
    if densities is not None:
        @app.callback(
            [Output("base-plot", "figure"), Output("data-plot", "figure")],
            [Input("selected-index-store", "data"), Input("radius-slider", "value"), Input("density-slider", "value")],
        )
        def update_figures(selected_index_store, r, density_threshold):
            selected_index = int(selected_index_store) if selected_index_store is not None else None
            fig_base, fig_data, _, _ = _make_figures(
                base_embedded=base_embedded,
                explained_variance=explained_variance,
                base_landmark_masks=base_landmark_masks,
                base_landmarks_embedded=base_landmarks_embedded,
                data=data,
                dist_mat=dist_mat,
                colors=colors,
                normalized_colors=normalized_colors,
                densities=densities,
                data_landmark_masks=data_landmark_masks,
                selected_index=selected_index,
                r=float(r),
                density_threshold=None if density_threshold is None else float(density_threshold),
            )
            return fig_base, fig_data

        # Download combined PDF snapshot (side-by-side)
        @app.callback(
            Output("download-pdf", "data"),
            Input("save-snapshot-btn", "n_clicks"),
            State("selected-index-store", "data"),
            State("radius-slider", "value"),
            State("density-slider", "value"),
            prevent_initial_call=True,
        )
        def _download_snapshot_pdf(n_clicks, selected_index_store, r, density_threshold):
            if selected_index_store is None:
                return dash.no_update

            selected_index = int(selected_index_store)
            fig_base, fig_data, _, _ = _make_figures(
                base_embedded=base_embedded,
                explained_variance=explained_variance,
                base_landmark_masks=base_landmark_masks,
                base_landmarks_embedded=base_landmarks_embedded,
                data=data,
                dist_mat=dist_mat,
                colors=colors,
                normalized_colors=normalized_colors,
                densities=densities,
                data_landmark_masks=data_landmark_masks,
                selected_index=selected_index,
                r=float(r),
                density_threshold=None if density_threshold is None else float(density_threshold),
            )

            png_left = _fig_to_png_bytes(fig_base, scale=1)
            png_right = _fig_to_png_bytes(fig_data, scale=1)
            pdf_bytes = _combine_pngs_to_pdf_bytes(png_left, png_right, gap=24, pad=0)

            fname = f"bundle_snapshot_i{selected_index}_r{float(r):.3f}_d{float(density_threshold):.3f}.pdf"
            return dcc.send_bytes(pdf_bytes, fname)

    else:
        @app.callback(
            [Output("base-plot", "figure"), Output("data-plot", "figure")],
            [Input("selected-index-store", "data"), Input("radius-slider", "value")],
        )
        def update_figures(selected_index_store, r):
            selected_index = int(selected_index_store) if selected_index_store is not None else None
            fig_base, fig_data, _, _ = _make_figures(
                base_embedded=base_embedded,
                explained_variance=explained_variance,
                base_landmark_masks=base_landmark_masks,
                base_landmarks_embedded=base_landmarks_embedded,
                data=data,
                dist_mat=dist_mat,
                colors=colors,
                normalized_colors=normalized_colors,
                densities=None,
                data_landmark_masks=data_landmark_masks,
                selected_index=selected_index,
                r=float(r),
                density_threshold=None,
            )
            return fig_base, fig_data

        @app.callback(
            Output("download-pdf", "data"),
            Input("save-snapshot-btn", "n_clicks"),
            State("selected-index-store", "data"),
            State("radius-slider", "value"),
            prevent_initial_call=True,
        )
        def _download_snapshot_pdf(n_clicks, selected_index_store, r):
            if selected_index_store is None:
                return dash.no_update

            selected_index = int(selected_index_store)
            fig_base, fig_data, _, _ = _make_figures(
                base_embedded=base_embedded,
                explained_variance=explained_variance,
                base_landmark_masks=base_landmark_masks,
                base_landmarks_embedded=base_landmarks_embedded,
                data=data,
                dist_mat=dist_mat,
                colors=colors,
                normalized_colors=normalized_colors,
                densities=None,
                data_landmark_masks=data_landmark_masks,
                selected_index=selected_index,
                r=float(r),
                density_threshold=None,
            )

            png_left = _fig_to_png_bytes(fig_base, scale=1)
            png_right = _fig_to_png_bytes(fig_data, scale=1)
            pdf_bytes = _combine_pngs_to_pdf_bytes(png_left, png_right, gap=24, pad=0)

            fname = f"bundle_snapshot_i{selected_index}_r{float(r):.3f}.pdf"
            return dcc.send_bytes(pdf_bytes, fname)

    return app


def run_bundle_app(app, *, port: Optional[int] = None, debug: bool = False):
    if port is None:
        port = find_free_port()
    url = f"http://127.0.0.1:{int(port)}/"
    print(f"Bundle viewer running at: {url}")
    app.run(debug=bool(debug), use_reloader=False, port=int(port))


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
    data_landmark_inds: Any = None,
    landmarks: Any = None,
    max_samples: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    port: Optional[int] = None,
    debug: bool = False,
):
    """
    General interactive viewer for (data, base_points) where base_points live in some metric space.

    - Neighborhoods come from dist_mat computed on base_points (via get_dist_mat/base_metric).
    - "Fiber Data" shows PCA of data restricted to the selected neighborhood.

    Snapshots
    ---------
    The save button downloads a combined PDF (side-by-side) containing ONLY the two Plotly figures
    (no sliders, no UI).
    """
    try:
        from dash import dcc  # noqa: F401
    except Exception:
        pass

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
        data_landmark_inds=data_landmark_inds,
        landmarks=landmarks,
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
) -> Tuple["go.Figure", "go.Figure"]:
    """
    Create the two figures for a given (selected_index, r, density_threshold) and optionally save.

    Notes on saving:
      - HTML always works: fig.write_html("file.html")
      - Static images require 'kaleido':
          pip install -U kaleido
    """
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "save_bundle_snapshot requires plotly. Install with `pip install plotly`."
        ) from e

    base_points = np.asarray(viz.base_points)
    data = np.asarray(viz.data)
    dist_mat = np.asarray(viz.dist_mat)
    colors = viz.colors
    densities = viz.densities

    data_landmark_masks = viz.data_landmark_masks
    base_landmark_masks = viz.base_landmark_masks
    base_landmark_points = viz.base_landmark_points

    base_embedded, explained_variance, pca_base = _embed_base_points_pca(base_points)
    normalized_colors = _normalize_to_unit_interval(colors) if colors is not None else None

    base_landmarks_embedded: Optional[List[np.ndarray]] = None
    if base_landmark_points is not None:
        base_landmarks_embedded = []
        for g in base_landmark_points:
            g = np.asarray(g, dtype=float)
            L = pca_base.transform(g)
            if L.shape[1] < 3:
                L = np.pad(L, ((0, 0), (0, 3 - L.shape[1])), mode="constant")
            base_landmarks_embedded.append(L)

    fig_base, fig_data, _, _ = _make_figures(
        base_embedded=base_embedded,
        explained_variance=explained_variance,
        base_landmark_masks=base_landmark_masks,
        base_landmarks_embedded=base_landmarks_embedded,
        data=data,
        dist_mat=dist_mat,
        colors=colors,
        normalized_colors=normalized_colors,
        densities=densities,
        data_landmark_masks=data_landmark_masks,
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
