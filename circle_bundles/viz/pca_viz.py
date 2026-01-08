from __future__ import annotations

from typing import List, Optional, Sequence

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
) -> None:
    """
    Fast-ish PCA visualization:

    1) Fits PCA (optionally randomized) on the full dataset, but only up to
       `max_components` components (or fewer if limited by n_samples/dim).
    2) 3D Plotly scatter of the first 3 PCs, plotting at most `max_points` points
       (downsampled uniformly at random if needed).
    3) Minimal Matplotlib plot: k vs cumulative explained variance (for the
       computed components only).
    4) Prints cumulative explained variance for k=1,2,3 (within computed range).

    Notes
    -----
    - If d is huge (e.g. 32^3 = 32768), exact PCA is expensive. Randomized PCA
      is usually much faster and accurate for the top components.
    - The cumulative explained variance curve is only computed up to
      `max_components`, not all the way to 100% unless you set max_components
      very large (which will be slow).

    Parameters
    ----------
    data : (n_samples, d)
    colors : optional length-n_samples, Plotly-accepted colors (numbers or strings).
             If U is provided and colors is None, we color per-set.
    U : optional (n_sets, n_samples) boolean mask for subset traces.
        If provided, trace toggling works; downsampling is applied consistently.
    titles : optional list of trace names (length n_sets).
    max_points : maximum number of points to plot in the 3D scatter.
    max_components : maximum number of PCA components to compute (for EV curve).
    use_randomized : if True, uses randomized SVD solver.
    n_iter : power iterations for randomized SVD (higher = more accurate, slower).
    """
    X = np.asarray(data, dtype=float)
    if X.ndim != 2:
        raise ValueError("show_pca expects a 2D array of shape (n_samples, d).")
    n, d = X.shape
    if n == 0:
        raise ValueError("show_pca received empty data (n_samples=0).")

    # Ensure at least 3 columns for plotting
    if d < 3:
        X = np.hstack([X, np.zeros((n, 3 - d), dtype=float)])
        d = 3

    # Decide number of components to compute
    n_components = int(min(max_components, n, d))
    if n_components < 1:
        raise ValueError("Not enough samples/dimensions to compute PCA.")

    # Fit PCA on full X
    if use_randomized:
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=random_state,
            iterated_power=n_iter,
        )
    else:
        pca = PCA(n_components=n_components)

    pca.fit(X)
    ev = pca.explained_variance_ratio_
    cev = np.cumsum(ev)

    # Downsample for plotting (but keep EV computed on full data)
    if max_points is not None and n > max_points:
        rng = np.random.default_rng(random_state)
        idx_plot = rng.choice(n, size=int(max_points), replace=False)
        idx_plot.sort()  # stable ordering for nicer toggling behavior
    else:
        idx_plot = np.arange(n)

    X_plot = X[idx_plot]
    Z = pca.transform(X_plot)[:, :3]

    # Jitter if z is basically constant
    if np.std(Z[:, 2]) < tol_flat_z:
        rng = np.random.default_rng(random_state)
        Z[:, 2] += jitter_eps * rng.standard_normal(Z.shape[0])

    # Handle U / titles / colors for plotting (on the downsampled subset)
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
        if colors is not None:
            colors_arr = np.asarray(colors, dtype=object)
            if colors_arr.shape[0] != n:
                raise ValueError("colors must have length n_samples when provided.")
            colors_plot = colors_arr[idx_plot]
        else:
            colors_plot = None

    # ----- Plotly 3D scatter -----
    fig = go.Figure()

    if use_U and U_plot is not None:
        for j in range(U_plot.shape[0]):
            idx = np.where(U_plot[j])[0]
            if idx.size == 0:
                continue

            if colors is None:
                c = set_colors[j]
            else:
                c = colors_plot[idx]

            fig.add_trace(
                go.Scatter3d(
                    x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2],
                    mode="markers",
                    marker=dict(size=size, color=c, opacity=0.85),
                    name=titles[j],
                )
            )
    else:
        fig.add_trace(
            go.Scatter3d(
                x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
                mode="markers",
                marker=dict(size=size, color=colors_plot, opacity=0.85),
                name="data",
            )
        )

    fig.update_layout(
        scene=dict(aspectmode="data"),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"PCA (first 3 components) | fit k={n_components} | plotted n={len(idx_plot)}",
    )
    fig.show()

    # ----- Minimal cumulative EV plot (up to n_components) -----
    ks = np.arange(1, n_components + 1)
    plt.figure()
    plt.plot(ks, cev)
    plt.xlabel("Number of Components (k)")
    plt.ylabel("Cumulative Explained Variance")
    plt.ylim(0.0, 1.01)
    plt.title("Cumulative Explained Variance vs k")
    plt.grid(True)
    plt.show()

    # ----- Print cumulative EV for first three components (if available) -----
    def _cev_at(k: int) -> float:
        return float(cev[k - 1]) if len(cev) >= k else float("nan")

    print("Cumulative Explained Variance:")
    print(f"  k=1: {_cev_at(1):.4f}")
    print(f"  k=2: {_cev_at(2):.4f}")
    print(f"  k=3: {_cev_at(3):.4f}")

    if n_components < 3:
        print(f"(Note: only computed up to k={n_components} components.)")
