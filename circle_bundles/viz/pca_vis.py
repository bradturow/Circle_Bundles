# circle_bundles/viz/pca_viz.py
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np


def show_pca(
    data: np.ndarray,
    *,
    colors: Optional[Sequence] = None,
    U: Optional[np.ndarray] = None,
    size: float = 2.0,
    titles: Optional[List[str]] = None,
    # --- speed controls ---
    max_points: int = 2000,
    max_components: int = 50,
    use_randomized: bool = False,
    n_iter: int = 4,
    # --- numerical / plot stability ---
    jitter_eps: float = 1e-3,
    tol_flat_z: float = 1e-6,
    random_state: int = 0,
    # --- appearance ---
    set_cmap: str = "viridis",
    set_cmap_range: Tuple[float, float] = (0.10, 0.90),
    elev = 20,
    azim = 35,
    # --- behavior ---
    interactive: bool = False,
    show: bool = True,
    return_figs: bool = False,
    # --- plotly UX ---
    plotly_scroll_zoom: bool = True,
    plotly_double_click: Union[bool, str] = "reset",
) -> Union["go.Figure", "plt.Figure", None]:
    """
    PCA visualization (3D scatter + cumulative explained variance), side-by-side.

    - Fits PCA on the full dataset (up to max_components).
    - Plots first 3 PCs (subsampled to max_points).
    - Plots cumulative explained variance (CEV) vs k.
    - No printing.

    Parameters
    ----------
    interactive:
        If True, uses Plotly and returns a Plotly Figure (single combined figure).
        If False, uses Matplotlib for BOTH panels and returns a Matplotlib Figure.
    plotly_scroll_zoom:
        If interactive, enables scroll zoom.
    plotly_double_click:
        Plotly double-click behavior. Use "reset" (default), "autosize", False, etc.

    Returns
    -------
    If return_figs=True:
        - interactive=True  -> plotly.graph_objects.Figure
        - interactive=False -> matplotlib.figure.Figure
    Else:
        None (but displays if show=True).
    """

    # ------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
    except ImportError as e:
        raise ImportError(
            "show_pca requires matplotlib. Install with `pip install matplotlib`."
        ) from e

    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError(
            "show_pca requires scikit-learn. Install with `pip install scikit-learn`."
        ) from e

    # Plotly only if interactive
    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError as e:
            raise ImportError(
                "interactive show_pca requires plotly. Install with `pip install plotly`."
            ) from e

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _set_axes_equal_3d(ax, xyz: np.ndarray) -> None:
        """
        Make Matplotlib 3D axis scale equal in x/y/z (as close as possible),
        so point clouds don't look squashed compared to Plotly's aspectmode='data'.
        """
        xyz = np.asarray(xyz, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            return

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        xmid = 0.5 * (x.max() + x.min())
        ymid = 0.5 * (y.max() + y.min())
        zmid = 0.5 * (z.max() + z.min())

        max_range = max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min())
        if not np.isfinite(max_range) or max_range <= 0:
            max_range = 1.0
        r = 0.5 * max_range

        ax.set_xlim(xmid - r, xmid + r)
        ax.set_ylim(ymid - r, ymid + r)
        ax.set_zlim(zmid - r, zmid + r)

        # Newer Matplotlib supports explicit 3D box aspect
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1, 1, 1))

    def _make_static_3d_prettier(ax) -> None:
        """Make Matplotlib 3D look cleaner without deleting the axis back panes."""
        # Plotly-ish camera vibe
        try:
            ax3d.view_init(elev=elev, azim=azim)
        except Exception:
            pass
    
        # Keep grid, but make it subtle
        try:
            ax.grid(True)
        except Exception:
            pass
    
        # Light, visible panes (the "backdrop")
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
            if axis is None:
                continue
            try:
                axis.pane.set_alpha(0.12)  # <--- backdrop visible again
                axis.pane.set_edgecolor((0, 0, 0, 0.15))
            except Exception:
                pass
    
        # Subtle grid line styling (works on most mpl versions)
        try:
            ax.xaxis._axinfo["grid"]["linewidth"] = 0.6
            ax.yaxis._axinfo["grid"]["linewidth"] = 0.6
            ax.zaxis._axinfo["grid"]["linewidth"] = 0.6
            ax.xaxis._axinfo["grid"]["linestyle"] = "-"
            ax.yaxis._axinfo["grid"]["linestyle"] = "-"
            ax.zaxis._axinfo["grid"]["linestyle"] = "-"
            ax.xaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.12)
            ax.yaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.12)
            ax.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0.12)
        except Exception:
            pass

    # ------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("show_pca expects a 2D array of shape (n_samples, d).")
    n, d = X.shape
    if n <= 0:
        raise ValueError("show_pca received empty data (n_samples=0).")

    # Ensure at least 3 dims for plotting
    if d < 3:
        X = np.hstack([X, np.zeros((n, 3 - d), dtype=float)])
        d = 3

    n_components = int(min(max_components, n, d))
    if n_components < 1:
        raise ValueError("Not enough samples/dimensions to compute PCA.")

    # ------------------------------------------------------------
    # PCA fit
    # ------------------------------------------------------------
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized" if use_randomized else "auto",
        random_state=random_state if use_randomized else None,
        iterated_power=n_iter if use_randomized else "auto",
    )
    pca.fit(X)

    ev = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cev = np.cumsum(ev)

    # ------------------------------------------------------------
    # Subsample for plotting
    # ------------------------------------------------------------
    if max_points is not None and n > int(max_points):
        rng = np.random.default_rng(random_state)
        idx_plot = rng.choice(n, size=int(max_points), replace=False)
        idx_plot.sort()
    else:
        idx_plot = np.arange(n)

    X_plot = X[idx_plot]
    Z = pca.transform(X_plot)

    if Z.shape[1] < 3:
        Z = np.hstack([Z, np.zeros((Z.shape[0], 3 - Z.shape[1]), dtype=float)])
    Z3 = Z[:, :3].copy()

    # Jitter flat z-axis
    if np.std(Z3[:, 2]) < float(tol_flat_z):
        rng = np.random.default_rng(random_state)
        Z3[:, 2] += float(jitter_eps) * rng.standard_normal(Z3.shape[0])

    # ------------------------------------------------------------
    # Handle U / colors
    # ------------------------------------------------------------
    use_U = U is not None and np.size(U) > 0
    colors_plot = None
    set_colors = None

    if use_U:
        U_bool = np.asarray(U, dtype=bool)
        if U_bool.ndim != 2 or U_bool.shape[1] != n:
            raise ValueError(f"U must have shape (n_sets, n_samples) = (?, {n}).")

        U_plot = U_bool[:, idx_plot]
        n_sets = U_plot.shape[0]

        if titles is None:
            titles = [f"U_{j}" for j in range(n_sets)]
        elif len(titles) != n_sets:
            raise ValueError("titles must have length equal to U.shape[0].")

        if colors is None:
            cmap = plt.get_cmap(set_cmap)
            a, b = map(float, set_cmap_range)
            ts = np.linspace(a, b, n_sets) if n_sets > 1 else [(a + b) / 2.0]
            set_colors = [to_hex(cmap(t)) for t in ts]
        else:
            colors_arr = np.asarray(colors, dtype=object)
            if colors_arr.shape[0] != n:
                raise ValueError("colors must have length n_samples.")
            colors_plot = colors_arr[idx_plot]
    else:
        if colors is not None:
            colors_arr = np.asarray(colors, dtype=object)
            if colors_arr.shape[0] != n:
                raise ValueError("colors must have length n_samples.")
            colors_plot = colors_arr[idx_plot]

    # ------------------------------------------------------------
    # CEV helper arrays
    # ------------------------------------------------------------
    ks = np.arange(1, n_components + 1)

    title_str = (
        f"PCA Summary"
    )

    # ============================================================
    # INTERACTIVE (Plotly): single figure with 2 columns
    # ============================================================
    if interactive:
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.62, 0.38],
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            horizontal_spacing=0.06,
            subplot_titles=("3D PCA Projection", "Cumulative Explained Variance"),
        )

        if use_U:
            for j in range(U_plot.shape[0]):
                idx = np.where(U_plot[j])[0]
                if idx.size == 0:
                    continue
                c = set_colors[j] if colors is None else colors_plot[idx]
                fig.add_trace(
                    go.Scatter3d(
                        x=Z3[idx, 0],
                        y=Z3[idx, 1],
                        z=Z3[idx, 2],
                        mode="markers",
                        marker=dict(size=float(size), color=c, opacity=0.85),
                        name=titles[j],
                        showlegend=True,
                    ),
                    row=1,
                    col=1,
                )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=Z3[:, 0],
                    y=Z3[:, 1],
                    z=Z3[:, 2],
                    mode="markers",
                    marker=dict(size=float(size), color=colors_plot, opacity=0.85),
                    name="data",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # CEV line (2D)
        fig.add_trace(
            go.Scatter(
                x=ks,
                y=cev,
                mode="lines+markers",
                name="Explained Variance",
                showlegend=False,
                hovertemplate="k=%{x}<br>CEV=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="k", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance", range=[0.0, 1.01], row=1, col=2)

        # IMPORTANT: preserve UI state so clicking/zooming right panel doesn't
        # reset/squish the 3D camera (and vice versa).
        fig.update_layout(
            title=title_str,
            scene=dict(aspectmode="data", uirevision="pca_scene"),
            margin=dict(l=0, r=0, t=60, b=0),
            legend=dict(x=0.01, y=0.99),
            uirevision="pca",  # global uirevision
        )
        fig.update_xaxes(uirevision="pca_axes", row=1, col=2)
        fig.update_yaxes(uirevision="pca_axes", row=1, col=2)

        if show:
            fig.show(
                config={
                    "scrollZoom": bool(plotly_scroll_zoom),
                    "doubleClick": plotly_double_click,
                }
            )

        if return_figs:
            return fig
        return None

    # ============================================================
    # STATIC (Matplotlib): one figure with 1 row, 2 cols
    # ============================================================
    fig = plt.figure(figsize=(12, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axev = fig.add_subplot(1, 2, 2)

    # 3D scatter
    if use_U:
        for j in range(U_plot.shape[0]):
            idx = np.where(U_plot[j])[0]
            if idx.size == 0:
                continue
            if colors is None:
                c = set_colors[j]
                ax3d.scatter(
                    Z3[idx, 0],
                    Z3[idx, 1],
                    Z3[idx, 2],
                    s=float(size) * 6.0,
                    c=c,
                    alpha=0.85,
                    label=titles[j],
                    depthshade=True,
                )
            else:
                c = colors_plot[idx]
                ax3d.scatter(
                    Z3[idx, 0],
                    Z3[idx, 1],
                    Z3[idx, 2],
                    s=float(size) * 6.0,
                    c=c,
                    alpha=0.85,
                    label=titles[j],
                    depthshade=True,
                )
        ax3d.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    else:
        ax3d.scatter(
            Z3[:, 0],
            Z3[:, 1],
            Z3[:, 2],
            s=float(size) * 6.0,
            c=colors_plot,
            alpha=0.85,
            depthshade=True,
        )

    ax3d.set_title("3D PCA Projection")

    # Make Matplotlib 3D look less squashed (closer to Plotly)
    _set_axes_equal_3d(ax3d, Z3)
    _make_static_3d_prettier(ax3d)

    # CEV plot
    axev.plot(ks, cev, marker="o")
    axev.set_title("Cumulative Explained Variance")
    axev.set_xlabel("Number of Components (k)")
    axev.set_ylabel("Explained Variance")
    axev.set_ylim(0.0, 1.01)
    axev.grid(True)

    fig.suptitle(title_str)
    fig.tight_layout()

    if show:
        plt.show()

    if return_figs:
        return fig
    return None
