# circle_bundles/viz/thumb_grids.py
from __future__ import annotations

from typing import Optional, Sequence, Callable, Any, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .image_utils import render_to_rgba


def show_data_vis(
    data: Sequence[Any],
    vis_func: Callable[[Any], Union[np.ndarray, plt.Figure]],
    *,
    angles: Optional[Sequence[float]] = None,
    max_samples: int = 100,
    n_cols: int = 10,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    label_func: Optional[Union[Callable[[Any], Any], Sequence[Any]]] = None,
    label_position: str = "below",
    sampling_method: str = "angle",
    font_size: int = 15,
    transparent_border: bool = True,
    white_thresh: int = 250,
    wspace: float = 0.25,
    hspace: float = 0.10,
    figsize_per_cell: float = 2.0,
    dpi: int = 150,
    pad_frac: float = 0.1,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Show a thumbnail grid of rendered data.

    IMPORTANT:
    - `data` is treated as a Sequence. We do NOT coerce it to np.asarray(data),
      because that can break for object-like payloads (meshes, images with varying shape, etc.)
    """
    n_total = len(data)
    if n_total == 0:
        raise ValueError("show_data_vis received empty data.")

    rng = np.random.default_rng(seed)
    n_take = int(min(max_samples, n_total))
    n_cols = max(1, int(n_cols))

    # ---- choose indices ----
    if sampling_method == "angle" and angles is not None:
        ang = np.asarray(list(angles), dtype=float).reshape(-1)
        if ang.shape[0] != n_total:
            raise ValueError(f"angles must have length {n_total}, got {ang.shape[0]}.")

        angle_min, angle_max = float(np.min(ang)), float(np.max(ang))
        centers = np.linspace(angle_min, angle_max, n_take)

        used: set[int] = set()
        selected: list[int] = []
        for c in centers:
            idx = int(np.argmin(np.abs(ang - c)))
            if idx not in used:
                selected.append(idx)
                used.add(idx)

        if len(selected) < n_take:
            remaining = np.array([i for i in range(n_total) if i not in used], dtype=int)
            if remaining.size > 0:
                filler = rng.choice(remaining, size=(n_take - len(selected)), replace=False)
                selected.extend([int(x) for x in filler])

        selected = sorted(selected, key=lambda i: float(ang[i]))

    elif sampling_method in ("random", None) or (sampling_method == "angle" and angles is None):
        selected = [int(x) for x in rng.choice(n_total, size=n_take, replace=False)]

    elif sampling_method == "first":
        selected = list(range(n_take))

    else:
        raise ValueError(f"Unknown sampling_method={sampling_method!r}")

    # ---- labels ----
    labels_sel = None
    if isinstance(label_func, (list, tuple, np.ndarray)):
        if len(label_func) != n_total:
            raise ValueError("If label_func is a sequence, it must have length len(data).")
        labels_sel = [label_func[i] for i in selected]

    # ---- layout via GridSpec ----
    n = len(selected)
    n_rows = int(np.ceil(n / n_cols))

    fig_w = float(figsize_per_cell) * n_cols
    fig_h = float(figsize_per_cell) * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))

    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        wspace=float(wspace),
        hspace=float(hspace),
        left=0.02, right=0.98,
        bottom=0.02, top=0.98,
    )

    axes: list[plt.Axes] = []

    for cell in range(n_rows * n_cols):
        r, c = divmod(cell, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
        axes.append(ax)

        if cell >= n:
            continue

        idx = selected[cell]
        rendered = vis_func(data[idx])
        img_u8 = render_to_rgba(
            rendered,
            transparent_border=bool(transparent_border),
            trim=True,
            white_thresh=int(white_thresh),
        )

        ax.imshow(img_u8, interpolation="nearest")

        if pad_frac and float(pad_frac) > 0:
            pad = float(pad_frac)
            ax.set_xlim(-pad * img_u8.shape[1], img_u8.shape[1] * (1 + pad))
            ax.set_ylim(img_u8.shape[0] * (1 + pad), -pad * img_u8.shape[0])

        label = None
        if callable(label_func):
            label = label_func(data[idx])
        elif labels_sel is not None:
            label = labels_sel[cell]

        if label is not None:
            y = -0.10 if label_position == "below" else 1.05
            ax.text(
                0.5, y, str(label),
                transform=ax.transAxes,
                ha="center",
                va="top" if label_position == "below" else "bottom",
                fontsize=int(font_size),
            )

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, np.array(axes, dtype=object)
