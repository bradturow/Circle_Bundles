# local_analysis.py
from __future__ import annotations

from typing import Sequence, Tuple, List

import numpy as np

__all__ = [
    "get_local_rips",
    "plot_local_rips",
    "get_local_pca",
]


# ============================================================
# Small internal helpers
# ============================================================

def _as_bool_U(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=bool)
    if U.ndim != 2:
        raise ValueError(f"U must be 2D (n_fibers, n_points). Got shape {U.shape}.")
    return U


def _as_p_values(p_values: np.ndarray | Sequence[float] | None, n_fibers: int) -> np.ndarray:
    if p_values is None:
        return np.ones(n_fibers, dtype=float)
    p = np.asarray(p_values, dtype=float).reshape(-1)
    if p.shape[0] != n_fibers:
        raise ValueError("p_values must have length equal to number of fibers (rows of U).")
    return p


def _as_fiber_ids(to_view: Sequence[int] | None, n_fibers: int) -> list[int]:
    if to_view is None or len(to_view) == 0:
        return list(range(n_fibers))
    fiber_ids = list(map(int, to_view))
    for j in fiber_ids:
        if j < 0 or j >= n_fibers:
            raise ValueError(f"to_view contains out-of-range fiber index {j} (n_fibers={n_fibers}).")
    return fiber_ids


def _subplots_grid(n_items: int, *, n_cols: int, figsize_per: float = 4.0):
    """
    Create a grid of subplots and return (fig, axes_flat).

    Lazy-imports matplotlib so importing this module doesn't pull it in.
    """
    import matplotlib.pyplot as plt  # lazy

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_items / n_cols)) if n_items > 0 else 1
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per * n_cols, figsize_per * n_rows),
    )
    axes_flat = np.atleast_1d(axes).ravel()
    return fig, axes_flat


def _default_titles(fiber_ids: Sequence[int]) -> list[str]:
    return [rf"$\pi^{{-1}}(U_{{{int(j)}}})$" for j in fiber_ids]


# ============================================================
# Public functions
# ============================================================

def get_dense_fiber_indices(
    U: np.ndarray,
    p_values: np.ndarray | Sequence[float] | None = None,
    to_view: Sequence[int] | None = None,
    random_state: int | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """
    Choose a (possibly subsampled) set of point indices in each fiber.

    Parameters
    ----------
    U : (n_fibers, n_points) bool indicator matrix
    p_values : per-fiber sampling fractions in [0,1]; default 1 for all
    to_view : optional subset of fiber indices to process
    random_state : RNG seed for subsampling

    Returns
    -------
    fiber_ids : list of fiber indices processed
    idx_list  : list of integer index arrays into the point set, one per fiber
    """
    U = _as_bool_U(U)
    n_fibers, _ = U.shape

    p_values_arr = _as_p_values(p_values, n_fibers)
    fiber_ids = _as_fiber_ids(to_view, n_fibers)

    U_sub = U[fiber_ids]
    p_sub = p_values_arr[fiber_ids]

    rng = np.random.default_rng(random_state)
    idx_list: list[np.ndarray] = []

    for row, p in zip(U_sub, p_sub):
        fiber_indices = np.where(row)[0].astype(int)
        m = int(fiber_indices.size)

        if m == 0:
            idx_list.append(np.array([], dtype=int))
            continue
        if m <= 2:
            idx_list.append(fiber_indices)
            continue

        p = float(np.clip(p, 0.0, 1.0))
        N = int(np.floor(p * m))
        N = max(2, min(N, m))  # keep at least 2 when possible

        chosen = rng.choice(fiber_indices, size=N, replace=False)
        idx_list.append(np.asarray(chosen, dtype=int))

    return fiber_ids, idx_list


def old_get_local_pca(
    data: np.ndarray,
    U: np.ndarray,
    p_values: np.ndarray | Sequence[float] | None = None,
    to_view: Sequence[int] | None = None,
    n_components: int = 2,
    random_state: int | None = None,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray | None]]:
    """
    Compute PCA embeddings separately on each fiber of a cover.

    This is a lightweight diagnostic tool: for each fiber (row of ``U``), we take the
    subset of samples belonging to that fiber, optionally subsample it, and run PCA on
    those points only. The result is a per-fiber low-dimensional embedding that can be
    visualized with :func:`plot_local_pca`.

    Parameters
    ----------
    data:
        Array of shape ``(n_points, d)`` containing the ambient data vectors.
    U:
        Boolean indicator matrix of shape ``(n_fibers, n_points)``. Entry ``U[r, i]``
        should be True exactly when point ``i`` lies in fiber/patch ``r``.
    p_values:
        Optional per-fiber sampling fractions in ``[0, 1]``. If provided, must have
        length ``n_fibers``. A value ``p_values[r] = p`` means “use approximately
        ``floor(p * m_r)`` points from fiber ``r``”, where ``m_r`` is that fiber’s
        membership size. If None, uses all points in each fiber.
    to_view:
        Optional list of fiber indices to process. If None, processes all fibers.
    n_components:
        Number of PCA components to compute per fiber.
    random_state:
        Seed used for subsampling and passed to PCA (for reproducibility).

    Returns
    -------
    fiber_ids:
        The list of fiber indices actually processed (either all fibers, or ``to_view``).
    idx_list:
        List of integer index arrays, one per processed fiber. These are the point indices
        used for PCA in that fiber (after subsampling, if any).
    proj_list:
        List of PCA projections, one per processed fiber. Entry ``proj_list[k]`` is an
        array of shape ``(m_k, n_components)`` for fiber ``fiber_ids[k]``, or None if that
        fiber contained fewer than 2 usable points.

    Notes
    -----
    - This function lazily imports scikit-learn so importing this module does not require
      scikit-learn unless you call it.
    - If you want to control the subsampling behavior directly, see
      ``get_dense_fiber_indices`` (currently not exported).
    """
    # lazy import so this module doesn't require sklearn unless used
    from sklearn.decomposition import PCA  # type: ignore

    data = np.asarray(data)
    U = _as_bool_U(U)

    fiber_ids, idx_list = get_dense_fiber_indices(
        U, p_values=p_values, to_view=to_view, random_state=random_state
    )

    proj_list: list[np.ndarray | None] = []
    for idx in idx_list:
        if idx.size < 2:
            proj_list.append(None)
            continue
        fiber_pts = data[idx]
        pca = PCA(n_components=int(n_components), random_state=random_state)
        proj_list.append(pca.fit_transform(fiber_pts))

    return fiber_ids, idx_list, proj_list


def plot_local_pca(
    fiber_ids: Sequence[int],
    proj_list: Sequence[np.ndarray | None],
    *,
    n_cols: int = 4,
    titles: str | Sequence[str] | None = "default",
    font_size: int = 16,
    point_size: float = 10.0,
    save_path: str | None = None,
):
    """
    Plot a grid of 2D PCA scatterplots for multiple fibers.

    This function is typically used with the output of :func:`get_local_pca`.
    Each subplot corresponds to one fiber; fibers whose PCA projection is missing
    (None or too small) are hidden.

    Parameters
    ----------
    fiber_ids:
        Fiber indices (for labeling). Usually the first output of :func:`get_local_pca`.
    proj_list:
        Per-fiber PCA coordinates. Usually the third output of :func:`get_local_pca`.
        Each entry should be an array of shape ``(m_i, 2)`` (at least 2 points),
        or None to skip that fiber.
    n_cols:
        Number of columns in the subplot grid.
    titles:
        Title behavior:
          - ``"default"``: uses LaTeX titles of the form ``\\pi^{-1}(U_j)``.
          - None: no titles.
          - sequence of strings: custom titles aligned with ``fiber_ids``.
    font_size:
        Title font size.
    point_size:
        Scatter marker size.
    save_path:
        If provided, save the figure to this path (any extension supported by matplotlib).

    Returns
    -------
    fig, axes:
        The matplotlib figure and a flat array of axes.
    """
    # lazy import
    import matplotlib.pyplot as plt  # noqa: F401  # (used via axes methods)

    fiber_ids = list(map(int, fiber_ids))
    n_fibers = len(fiber_ids)

    if titles == "default":
        titles_list = _default_titles(fiber_ids)
    elif titles is None:
        titles_list = None
    else:
        titles_list = list(titles)

    fig, axes = _subplots_grid(n_fibers, n_cols=int(n_cols), figsize_per=4.0)

    last_k = -1
    for k, (fiber_idx, proj) in enumerate(zip(fiber_ids, proj_list)):
        last_k = k
        ax = axes[k]
        if proj is None or proj.shape[0] < 2:
            ax.set_axis_off()
            continue

        ax.scatter(proj[:, 0], proj[:, 1], s=float(point_size))
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.3)

        if titles_list is not None:
            title = titles_list[k] if k < len(titles_list) else f"Fiber {fiber_idx}"
            ax.set_title(title, fontsize=int(font_size))

    for i in range(last_k + 1, len(axes)):
        axes[i].set_axis_off()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved local PCA figure to {save_path}")

    return fig, axes


def get_local_rips(
    data: np.ndarray,
    U: np.ndarray,
    p_values: np.ndarray | Sequence[float] | None = None,
    to_view: Sequence[int] | None = None,
    *,
    maxdim: int = 0,
    n_perm: int = 500,
    random_state: int | None = None,
    **ripser_kwargs,
) -> tuple[list[int], list[np.ndarray], list[dict | None]]:
    """
    Compute Ripser persistence separately on each fiber of a cover.

    For each fiber (row of ``U``), this function extracts the corresponding points
    from ``data`` (optionally subsampling them), then runs ``ripser`` on that fiber’s
    point cloud. The resulting persistence diagrams can be plotted with
    :func:`plot_local_rips`.

    Parameters
    ----------
    data:
        Array of shape ``(n_points, d)`` containing the ambient data vectors.
    U:
        Boolean indicator matrix of shape ``(n_fibers, n_points)`` indicating fiber
        membership (same convention as :func:`get_local_pca`).
    p_values:
        Optional per-fiber sampling fractions in ``[0, 1]`` (same convention as
        :func:`get_local_pca`).
    to_view:
        Optional list of fiber indices to process. If None, processes all fibers.
    maxdim:
        Maximum homology dimension for Ripser. If ``maxdim < 0``, returns None results
        for all fibers (useful as a quick “skip” switch).
    n_perm:
        Upper bound on the number of points passed to ripser per fiber. For each fiber,
        we set ``n_use = min(n_perm, m_fiber)`` and pass ``n_perm=n_use`` to ripser.
        (This leverages ripser’s internal subsampling/permutation behavior.)
    random_state:
        Seed used for the initial per-fiber subsampling (via ``p_values``).
    **ripser_kwargs:
        Additional keyword arguments forwarded directly to ``ripser(...)``.

    Returns
    -------
    fiber_ids:
        The list of fiber indices processed.
    idx_list:
        List of integer index arrays (points used in each fiber after subsampling).
    rips_list:
        List of ripser result dicts, one per fiber, or None for fibers with fewer than
        2 usable points.

    Notes
    -----
    - This function lazily imports ``ripser`` so importing this module does not require
      ripser unless you call it.
    - If you want exact control over which points go into each fiber computation, see
      ``get_dense_fiber_indices`` (currently not exported).
    """
    from ripser import ripser  # type: ignore  # lazy import

    data = np.asarray(data)
    U = _as_bool_U(U)

    fiber_ids, idx_list = get_dense_fiber_indices(
        U, p_values=p_values, to_view=to_view, random_state=random_state
    )

    if int(maxdim) < 0:
        return fiber_ids, idx_list, [None] * len(idx_list)

    rips_list: list[dict | None] = []
    for idx in idx_list:
        if idx.size < 2:
            rips_list.append(None)
            continue

        fiber_pts = data[idx]
        n_use = int(min(int(n_perm), fiber_pts.shape[0]))
        res = ripser(fiber_pts, maxdim=int(maxdim), n_perm=n_use, **ripser_kwargs)
        rips_list.append(res)

    return fiber_ids, idx_list, rips_list


def plot_local_rips(
    fiber_ids: Sequence[int],
    rips_list: Sequence[dict | None],
    *,
    n_cols: int = 4,
    titles: str | Sequence[str] | None = "default",
    font_size: int = 16,
    save_path: str | None = None,
):
    """
    Plot a grid of persistence diagrams for multiple fibers.

    This function is typically used with the output of :func:`get_local_rips`.
    Each subplot corresponds to one fiber; fibers with missing ripser output are hidden.

    Parameters
    ----------
    fiber_ids:
        Fiber indices (for labeling). Usually the first output of :func:`get_local_rips`.
    rips_list:
        Per-fiber ripser outputs. Usually the third output of :func:`get_local_rips`.
        Each entry should be a dict containing a ``"dgms"`` key, or None to skip.
    n_cols:
        Number of columns in the subplot grid.
    titles:
        Title behavior:
          - ``"default"``: uses LaTeX titles of the form ``\\pi^{-1}(U_j)``.
          - None: no titles.
          - sequence of strings: custom titles aligned with ``fiber_ids``.
    font_size:
        Title font size.
    save_path:
        If provided, save the figure to this path.

    Returns
    -------
    fig, axes:
        The matplotlib figure and a flat array of axes.

    Notes
    -----
    - This function lazily imports ``persim`` (for ``plot_diagrams``) so importing this
      module does not require persim unless you call it.
    """
    from persim import plot_diagrams  # type: ignore  # lazy import

    fiber_ids = list(map(int, fiber_ids))
    n_fibers = len(fiber_ids)
    if n_fibers == 0:
        raise ValueError("No fibers to plot.")

    if titles == "default":
        titles_list = _default_titles(fiber_ids)
    elif titles is None:
        titles_list = None
    else:
        titles_list = list(titles)

    fig, axes = _subplots_grid(n_fibers, n_cols=int(n_cols), figsize_per=4.0)

    last_k = -1
    for k, (fiber_idx, res) in enumerate(zip(fiber_ids, rips_list)):
        last_k = k
        ax = axes[k]
        if res is None or ("dgms" not in res):
            ax.set_axis_off()
            continue

        plot_diagrams(res["dgms"], ax=ax, show=False)

        if titles_list is not None:
            title = titles_list[k] if k < len(titles_list) else f"Fiber {fiber_idx}"
            ax.set_title(title, fontsize=int(font_size))

    for i in range(last_k + 1, len(axes)):
        axes[i].set_axis_off()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved local Rips figure to {save_path}")

    return fig, axes


def get_local_pca(
    data: np.ndarray,
    U: np.ndarray,
    *,
    f: np.ndarray | None = None,
    to_view: Sequence[int] | None = None,
    n_components: int = 2,
    n_cols: int = 3,
    titles: str | Sequence[str] | None = "default",
    font_size: int = 16,
    point_size: float = 10.0,
    cmap: str = "hsv",
    save_path: str | None = None,
    show: bool = True,
):
    """
    Compute and plot local PCA embeddings for fibers of a cover.

    Each fiber j is plotted using:
        data[U[j]]          (points)
        f[j, U[j]]          (colors, if provided)

    No subsampling. No index ambiguity. If dimensions don't match, this errors.

    Parameters
    ----------
    data : (n_points, d) array
        Ambient data.
    U : (n_fibers, n_points) bool array
        Cover membership matrix.
    f : (n_fibers, n_points) array, optional
        Local circular coordinates. Used for coloring if provided.
    to_view : list[int], optional
        Fibers to plot. Default: all.
    n_components : int
        PCA dimension (must be >= 2 for plotting).
    n_cols : int
        Number of subplot columns.
    titles : "default", None, or list[str]
        Title style.
    cmap : str
        Matplotlib colormap for angle coloring.
    save_path : str, optional
        Save figure path.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig, axes
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    data = np.asarray(data)
    U = _as_bool_U(U)

    n_fibers, n_points = U.shape

    if f is not None:
        f = np.asarray(f, dtype=float)
        if f.shape != U.shape:
            raise ValueError(f"f must have shape {U.shape}, got {f.shape}")

    fiber_ids = _as_fiber_ids(to_view, n_fibers)

    if titles == "default":
        titles_list = _default_titles(fiber_ids)
    elif titles is None:
        titles_list = None
    else:
        titles_list = list(titles)

    fig, axes = _subplots_grid(len(fiber_ids), n_cols=int(n_cols))
    last_k = -1

    for k, j in enumerate(fiber_ids):
        last_k = k
        ax = axes[k]

        idx = np.where(U[j])[0].astype(int)
        if idx.size < 2:
            ax.set_axis_off()
            continue

        Xj = data[idx]
        proj = PCA(n_components=int(n_components)).fit_transform(Xj)

        if proj.shape[1] < 2:
            raise ValueError("Need at least 2 PCA components to plot.")

        if f is None:
            ax.scatter(proj[:, 0], proj[:, 1], s=float(point_size))
        else:
            angles = np.mod(f[j, idx], 2 * np.pi)
            sc = ax.scatter(
                proj[:, 0],
                proj[:, 1],
                s=float(point_size),
                c=angles,
                cmap=cmap,
            )
#            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.3)

        if titles_list is not None:
            ax.set_title(titles_list[k], fontsize=int(font_size))

    for i in range(last_k + 1, len(axes)):
        axes[i].set_axis_off()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved local PCA figure to {save_path}")

    if show:
        plt.show()

    return fig, axes