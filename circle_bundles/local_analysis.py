from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

__all__ = [
    "get_dense_fiber_indices",
    "get_local_pca",
    "plot_local_pca",
    "get_local_rips",
    "plot_local_rips",
]


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
    U = np.asarray(U, dtype=bool)
    n_fibers, _ = U.shape

    if p_values is None:
        p_values_arr = np.ones(n_fibers, dtype=float)
    else:
        p_values_arr = np.asarray(p_values, dtype=float)
        if p_values_arr.shape[0] != n_fibers:
            raise ValueError("p_values must have length equal to number of fibers (rows of U).")

    if to_view is not None and len(to_view) > 0:
        fiber_ids = list(map(int, to_view))
        U_sub = U[fiber_ids]
        p_sub = p_values_arr[fiber_ids]
    else:
        fiber_ids = list(range(n_fibers))
        U_sub = U
        p_sub = p_values_arr

    rng = np.random.default_rng(random_state)
    idx_list: list[np.ndarray] = []

    for row, p in zip(U_sub, p_sub):
        fiber_indices = np.where(row)[0].astype(int)
        m = fiber_indices.size

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


def get_local_pca(
    data: np.ndarray,
    U: np.ndarray,
    p_values: np.ndarray | Sequence[float] | None = None,
    to_view: Sequence[int] | None = None,
    n_components: int = 2,
    random_state: int | None = None,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray | None]]:
    """
    Local PCA on each fiber (after optional subsampling).

    Returns
    -------
    fiber_ids : list[int]
    idx_list  : list[np.ndarray]          indices used in each fiber
    proj_list : list[np.ndarray | None]   PCA projections (m_i, n_components) or None
    """
    data = np.asarray(data)

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
    Grid of PCA scatterplots. Returns (fig, axes).
    """
    fiber_ids = list(map(int, fiber_ids))
    n_fibers = len(fiber_ids)

    if titles == "default":
        titles_list = [rf"$\pi^{{-1}}(U_{{{j}}})$" for j in fiber_ids]
    elif titles is None:
        titles_list = None
    else:
        titles_list = list(titles)

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_fibers / n_cols)) if n_fibers > 0 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

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
            # titles_list is either aligned with fiber_ids ("default") or provided explicitly
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
    Run Ripser on each fiber after optional subsampling.

    Notes
    -----
    Imports ripser lazily so importing this module doesn't require ripser.
    """
    # lazy import
    from ripser import ripser  # type: ignore

    data = np.asarray(data)

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
        n_use = int(min(n_perm, fiber_pts.shape[0]))
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
    Grid of persistence diagrams. Returns (fig, axes).

    Notes
    -----
    Imports persim lazily so importing this module doesn't require persim.
    """
    from persim import plot_diagrams  # type: ignore

    fiber_ids = list(map(int, fiber_ids))
    n_fibers = len(fiber_ids)
    if n_fibers == 0:
        raise ValueError("No fibers to plot.")

    if titles == "default":
        titles_list = [rf"$\pi^{{-1}}(U_{{{j}}})$" for j in fiber_ids]
    elif titles is None:
        titles_list = None
    else:
        titles_list = list(titles)

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n_fibers / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

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
