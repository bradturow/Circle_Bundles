# viz/angles.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def set_pi_ticks(ax, fontsize=12):
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=fontsize)


def fit_o2_on_circle(angles_ref: np.ndarray, angles_mov: np.ndarray):
    """
    Fit A in O(2) so that (cos,sin)(angles_mov) @ A ≈ (cos,sin)(angles_ref).
    Returns:
      A : (2,2) orthogonal matrix
      mean_err : mean Euclidean error in R^2 after alignment
      rms_err  : RMS Euclidean error in R^2 after alignment
    """
    a1 = np.asarray(angles_ref).reshape(-1)
    a2 = np.asarray(angles_mov).reshape(-1)
    if a1.shape != a2.shape:
        raise ValueError("angles_ref and angles_mov must have same shape.")
    if a1.size == 0:
        raise ValueError("Empty angle arrays.")

    X = np.c_[np.cos(a2), np.sin(a2)]  # moving
    Y = np.c_[np.cos(a1), np.sin(a1)]  # reference

    M = X.T @ Y
    U, _, Vt = np.linalg.svd(M)
    A = U @ Vt  # in O(2)

    R = X @ A
    diff = R - Y
    per_pt = np.linalg.norm(diff, axis=1)
    mean_err = float(np.mean(per_pt))
    rms_err = float(np.sqrt(np.mean(per_pt**2)))
    return A, mean_err, rms_err


def align_angles_to(angles_ref: np.ndarray, angles_mov: np.ndarray) -> np.ndarray:
    """Return angles_mov after optimal O(2) alignment to angles_ref (mod 2pi)."""
    A, _, _ = fit_o2_on_circle(angles_ref, angles_mov)
    X = np.c_[np.cos(angles_mov), np.sin(angles_mov)]
    X2 = X @ A
    return np.mod(np.arctan2(X2[:, 1], X2[:, 0]), 2 * np.pi)


def _set_angle_ticks(ax, *, which: str, ticks, labels, fontsize: int):
    if which == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=fontsize)
    elif which == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=fontsize)
    else:
        raise ValueError("which must be 'x' or 'y'.")


def _pi_ticks_0_to_2pi():
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    return ticks, labels


def _pi_ticks_0_to_pi():
    ticks = [0, np.pi / 2, np.pi]
    labels = [r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    return ticks, labels


def compare_angle_pairs(
    angle_arrays: list[np.ndarray],
    pairs: list[tuple[int, int]],
    *,
    labels: list[str] | None = None,
    align: bool = False,
    s: float = 1.0,
    fontsize: int = 14,
    ncols: int | str = "auto",
    titles: list[str] | None = None,
    titlesize: int = 14,
    show_metrics: bool = True,
    metric: str = "mean",  # "mean" or "rms"

    x_range: tuple[float, float] = (0.0, 2 * np.pi),
    y_range: tuple[float, float] = (0.0, 2 * np.pi),
    x_ticks: tuple[list[float], list[str]] | None = "auto",
    y_ticks: tuple[list[float], list[str]] | None = "auto",
):
    """
    Scatter plots angle_arrays[i] vs angle_arrays[j] for each (i,j) in pairs.

    If align=True, aligns the *second* array in each pair to the first using O(2).

    New:
      - x_range / y_range let you use domains like [0,pi] for RP^1 angles.
      - tick control via x_ticks/y_ticks:
          * "auto" (default): choose nice pi-ticks for [0,pi] or [0,2pi] if matched.
          * None: leave matplotlib defaults.
          * (ticks, labels): explicit.
    """
    if labels is None:
        labels = [f"angle[{i}]" for i in range(len(angle_arrays))]

    if ncols == "auto":
        nrows = int(np.ceil(np.sqrt(len(pairs))))
        ncols = int(np.ceil(len(pairs) / nrows))
    else:
        ncols = int(ncols)
        nrows = int(np.ceil(len(pairs) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    # Auto ticks based on ranges (only if user left as "auto")
    def resolve_ticks(ticks_setting, rng):
        if ticks_setting is None:
            return None
        if ticks_setting != "auto":
            return ticks_setting  # explicit (ticks, labels)

        lo, hi = float(rng[0]), float(rng[1])
        # Only do the pretty pi-ticks if it matches *exactly* these common ranges
        if np.isclose(lo, 0.0) and np.isclose(hi, 2 * np.pi):
            return _pi_ticks_0_to_2pi()
        if np.isclose(lo, 0.0) and np.isclose(hi, np.pi):
            return _pi_ticks_0_to_pi()
        return None  # fall back to matplotlib defaults

    xt = resolve_ticks(x_ticks, x_range)
    yt = resolve_ticks(y_ticks, y_range)

    for t, (i, j) in enumerate(pairs):
        a = np.asarray(angle_arrays[i]).reshape(-1)
        b = np.asarray(angle_arrays[j]).reshape(-1)

        if a.shape != b.shape:
            raise ValueError(f"Pair {t}: shapes differ: {a.shape} vs {b.shape}")

        _, mean_err, rms_err = fit_o2_on_circle(a, b)
        b_plot = align_angles_to(a, b) if align else b

        ax = axes[t]
        ax.scatter(a, b_plot, s=s, alpha=0.7)

        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)

        if xt is not None:
            ticks, ticklabels = xt
            _set_angle_ticks(ax, which="x", ticks=ticks, labels=ticklabels, fontsize=fontsize)
        if yt is not None:
            ticks, ticklabels = yt
            _set_angle_ticks(ax, which="y", ticks=ticks, labels=ticklabels, fontsize=fontsize)

        ax.set_xlabel(labels[i], fontsize=fontsize)
        ax.set_ylabel(labels[j] + (" (aligned)" if align else ""), fontsize=fontsize)

        if titles is not None:
            title = titles[t]
        else:
            if show_metrics:
                val = mean_err if metric == "mean" else rms_err
                title = f"{metric} err (circle): {val:.4g}"
            else:
                title = ""
        ax.set_title(title, fontsize=titlesize)

        ax.grid(True, linestyle="--", alpha=0.4)

        # keep square panels
        ax.set_aspect("auto")
        ax.set_box_aspect(1)

    for k in range(len(pairs), len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    return fig


def _select_edges_by_error(
    U: np.ndarray,
    f: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    metric: str = "mean",          # "mean" or "rms"
    max_pairs: int = 25,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    """
    Decide which edges to plot.

    Returns:
      selected_edges
      err_by_edge: edge -> error
      overlap_by_edge: edge -> |Uj ∩ Uk|
    """
    if metric not in ("mean", "rms"):
        raise ValueError("metric must be 'mean' or 'rms'.")

    # First collect only nonempty overlaps + their errors
    err_by_edge: dict[tuple[int, int], float] = {}
    ov_by_edge: dict[tuple[int, int], int] = {}

    for (j, k) in edges:
        j, k = int(j), int(k)
        mask = U[j] & U[k]
        ov = int(mask.sum())
        if ov == 0:
            continue

        a = f[j, mask]
        b = f[k, mask]
        _, mean_err, rms_err = fit_o2_on_circle(a, b)
        err = float(mean_err if metric == "mean" else rms_err)

        e = (j, k)
        err_by_edge[e] = err
        ov_by_edge[e] = ov

    candidates = list(err_by_edge.keys())
    if not candidates:
        return [], err_by_edge, ov_by_edge

    # If within cap, keep them all in original input order (stable)
    if len(candidates) <= int(max_pairs):
        selected = [e for e in edges if e in err_by_edge]
        return selected, err_by_edge, ov_by_edge

    # Otherwise pick best / worst / median by error
    candidates_sorted = sorted(candidates, key=lambda e: err_by_edge[e])
    best = candidates_sorted[0]
    worst = candidates_sorted[-1]
    median = candidates_sorted[len(candidates_sorted) // 2]

    # Ensure uniqueness if degenerate
    selected = []
    for e in (worst, median, best):  # I like worst→median→best order for story
        if e not in selected:
            selected.append(e)

    return selected, err_by_edge, ov_by_edge


def compare_trivs(
    cover,
    f: np.ndarray,
    *,
    edges: list[tuple[int, int]] | None = None,
    ncols: int | str = "auto",
    title_size: int = 14,
    align: bool = False,
    s: float = 1.0,
    save_path: str | None = None,
    show: bool = True,
    # NEW:
    max_pairs: int = 25,
    metric: str = "mean",  # "mean" or "rms" (used for selection + display)
    return_selected: bool = False,
):
    """
    Compare local trivializations on overlaps for each nerve edge (j,k).

    cover.U: (n_sets, n_samples) bool
    f: (n_sets, n_samples) angles (radians or cycles—doesn't matter for alignment as long as consistent)

    New behavior:
      - If number of nonempty overlaps <= max_pairs: plot all.
      - Otherwise: plot 3 examples only: WORST / MEDIAN / BEST (by chosen metric).
    """
    U = np.asarray(cover.U, dtype=bool)
    if U.ndim != 2:
        raise ValueError("cover.U must be 2D (n_sets, n_samples).")

    n_sets, n_samples = U.shape
    f = np.asarray(f)
    if f.shape != (n_sets, n_samples):
        raise ValueError(f"f must have shape {(n_sets, n_samples)}, got {f.shape}")

    if edges is None:
        edges = list(cover.nerve_edges())

    selected_edges, err_by_edge, ov_by_edge = _select_edges_by_error(
        U,
        f,
        edges,
        metric=metric,
        max_pairs=int(max_pairs),
    )

    if not selected_edges:
        raise ValueError("No nonempty overlaps found on the provided edges.")

    # Determine whether we subsampled (for nicer titles)
    subsampled = (len(selected_edges) < len([e for e in edges if (int(e[0]), int(e[1])) in err_by_edge]))
    tag_by_edge: dict[tuple[int, int], str] = {}
    if subsampled and len(selected_edges) >= 2:
        # selected_edges is [worst, median, best] in that order (with de-dup)
        # compute ranks by error to decide labeling robustly
        sorted_edges = sorted(err_by_edge.keys(), key=lambda e: err_by_edge[e])
        best = sorted_edges[0]
        worst = sorted_edges[-1]
        median = sorted_edges[len(sorted_edges) // 2]
        tag_by_edge[best] = "BEST"
        tag_by_edge[worst] = "WORST"
        tag_by_edge[median] = "MEDIAN"

    angle_arrays: list[np.ndarray] = []
    labels: list[str] = []
    titles: list[str] = []
    pairs: list[tuple[int, int]] = []

    for (j, k) in selected_edges:
        j, k = int(j), int(k)
        mask = U[j] & U[k]
        if not np.any(mask):
            continue

        a = f[j, mask]
        b = f[k, mask]

        angle_arrays.append(a)
        angle_arrays.append(b)
        labels.append(fr"$f_{{{j}}}$")
        labels.append(fr"$f_{{{k}}}$")

        ov = int(mask.sum())
        err = float(err_by_edge[(j, k)])

        tag = tag_by_edge.get((j, k), "")
        tag_str = f" [{tag}]" if tag else ""
        titles.append(
            fr"$f_{{{j}}}$ vs $f_{{{k}}}$"
            + tag_str
            + fr"  ($|U_j\cap U_k|={ov}$, {metric} err={err:.4g})"
        )

        pairs.append((2 * len(pairs), 2 * len(pairs) + 1))

    fig = compare_angle_pairs(
        angle_arrays,
        pairs,
        labels=labels,
        align=align,
        s=s,
        ncols=ncols,
        titles=titles,
        titlesize=title_size,
        metric=metric,
        show_metrics=False,  # titles already include the metric
    )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    if return_selected:
        return fig, selected_edges, err_by_edge

    return fig
