# circle_bundles/viz/pca_viz.py
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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
    # --- numerical/plot stability ---
    jitter_eps: float = 1e-3,
    tol_flat_z: float = 1e-6,
    random_state: int = 0,
    # --- behavior ---
    show: bool = True,
    return_figs: bool = False,
) -> Tuple[go.Figure, plt.Figure] | None:
    """
    PCA visualization:

    - Fit PCA on full data (up to max_components).
    - Plot first 3 PCs in Plotly, using up to max_points points.
    - Plot cumulative explained variance (up to computed components) in Matplotlib.
    - Print CEV for k=1,2,3 (if available).

    If U is provided, renders one trace per set (toggleable).
    """

    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("show_pca expects a 2D array of shape (n_samples, d).")
    n, d = X.shape
    if n <= 0:
        raise ValueError("show_pca received empty data (n_samples=0).")

    # pad to at least 3 dims so we can always plot 3 PCs
    if d < 3:
        X = np.hstack([X, np.zeros((n, 3 - d), dtype=float)])
        d = 3

    n_components = int(min(max_components, n, d))
    if n_components < 1:
        raise ValueError("Not enough samples/dimensions to compute PCA.")

    pca = PCA(
        n_components=n_components,
        svd_solver="randomized" if use_randomized else "auto",
        random_state=random_state if use_randomized else None,
        iterated_power=n_iter if use_randomized else "auto",
    )
    pca.fit(X)

    ev = np.asarray(pca.explained_variance_ratio_, dtype=float)
    cev = np.cumsum(ev)

    # ---------- choose plot subset ----------
    if max_points is not None and n > int(max_points):
        rng = np.random.default_rng(random_state)
        idx_plot = rng.choice(n, size=int(max_points), replace=False)
        idx_plot.sort()
    else:
        idx_plot = np.arange(n)

    X_plot = X[idx_plot]
    Z = pca.transform(X_plot)
    # ensure we have at least 3 columns
    if Z.shape[1] < 3:
        Z = np.hstack([Z, np.zeros((Z.shape[0], 3 - Z.shape[1]), dtype=float)])
    Z3 = Z[:, :3]

    # Jitter if z is basically constant
    if np.std(Z3[:, 2]) < float(tol_flat_z):
        rng = np.random.default_rng(random_state)
        Z3[:, 2] += float(jitter_eps) * rng.standard_normal(Z3.shape[0])

    # ---------- handle U / colors ----------
    use_U = U is not None and np.size(U) > 0
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
            palette = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            ]
            set_colors = [palette[j % len(palette)] for j in range(n_sets)]
            colors_plot = None
        else:
            colors_arr = np.asarray(colors, dtype=object)
            if colors_arr.shape[0] != n:
                raise ValueError("colors must have length n_samples when provided.")
            colors_plot = colors_arr[idx_plot]
            set_colors = None
    else:
        U_plot = None
        set_colors = None
        if colors is not None:
            colors_arr = np.asarray(colors, dtype=object)
            if colors_arr.shape[0] != n:
                raise ValueError("colors must have length n_samples when provided.")
            colors_plot = colors_arr[idx_plot]
        else:
            colors_plot = None

    # ---------- Plotly 3D scatter ----------
    fig3d = go.Figure()

    if use_U and U_plot is not None:
        for j in range(U_plot.shape[0]):
            idx = np.where(U_plot[j])[0]
            if idx.size == 0:
                continue

            c = set_colors[j] if colors is None else colors_plot[idx]
            fig3d.add_trace(
                go.Scatter3d(
                    x=Z3[idx, 0], y=Z3[idx, 1], z=Z3[idx, 2],
                    mode="markers",
                    marker=dict(size=float(size), color=c, opacity=0.85),
                    name=titles[j],
                )
            )
    else:
        fig3d.add_trace(
            go.Scatter3d(
                x=Z3[:, 0], y=Z3[:, 1], z=Z3[:, 2],
                mode="markers",
                marker=dict(size=float(size), color=colors_plot, opacity=0.85),
                name="data",
            )
        )

    fig3d.update_layout(
        scene=dict(aspectmode="data"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"PCA (first 3 PCs) | fit k={n_components} | plotted n={len(idx_plot)}",
    )

    # ---------- Matplotlib CEV ----------
    ks = np.arange(1, n_components + 1)
    fig_ev = plt.figure()
    plt.plot(ks, cev)
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Cumulative Explained Variance")
    plt.ylim(0.0, 1.01)
    plt.title("Cumulative Explained Variance vs k")
    plt.grid(True)

    def _cev_at(k: int) -> float:
        return float(cev[k - 1]) if len(cev) >= k else float("nan")

    print("Cumulative Explained Variance:")
    print(f"  k=1: {_cev_at(1):.4f}")
    print(f"  k=2: {_cev_at(2):.4f}")
    print(f"  k=3: {_cev_at(3):.4f}")
    if n_components < 3:
        print(f"(Note: only computed up to k={n_components} components.)")

    if show:
        fig3d.show()
        plt.show()

    if return_figs:
        return fig3d, fig_ev

    return None
