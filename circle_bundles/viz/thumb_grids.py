from __future__ import annotations

from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from .image_utils import render_to_rgba


def show_data_vis(
    data,
    vis_func,
    *,
    angles=None,
    max_samples: int = 100,
    n_cols: int = 10,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    label_func=None,
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
):
    data = np.asarray(data)
    n_total = data.shape[0]

    # ---- choose indices ----
    if sampling_method == "angle" and angles is not None:
        angles = np.asarray(angles).reshape(-1)
        angle_min, angle_max = float(angles.min()), float(angles.max())
        bin_centers = np.linspace(angle_min, angle_max, min(max_samples, n_total))

        used = set()
        selected = []
        for c in bin_centers:
            idx = int(np.argmin(np.abs(angles - c)))
            if idx not in used:
                selected.append(idx)
                used.add(idx)

        if len(selected) < min(max_samples, n_total):
            remaining = np.array([i for i in range(n_total) if i not in used], dtype=int)
            if remaining.size > 0:
                rng = np.random.default_rng(seed)
                filler = rng.choice(
                    remaining,
                    size=min(max_samples, n_total) - len(selected),
                    replace=False,
                )
                selected.extend(list(filler))

        selected = sorted(selected, key=lambda i: angles[i])

    elif sampling_method in ("random", None) or (sampling_method == "angle" and angles is None):
        rng = np.random.default_rng(seed)
        selected = rng.choice(n_total, size=min(max_samples, n_total), replace=False).tolist()

    elif sampling_method == "first":
        selected = list(range(min(max_samples, n_total)))

    else:
        raise ValueError(f"Unknown sampling_method={sampling_method}")

    data_sel = data[selected]
    labels_sel = None
    if isinstance(label_func, (list, np.ndarray)):
        labels_sel = [label_func[i] for i in selected]

    # ---- layout via GridSpec ----
    n = len(data_sel)
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(n / n_cols))

    fig_w = figsize_per_cell * n_cols
    fig_h = figsize_per_cell * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=int(dpi))

    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        wspace=float(wspace),
        hspace=float(hspace),
        left=0.02, right=0.98,
        bottom=0.02, top=0.98,
    )

    axes = []
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
        axes.append(ax)

        if i >= n:
            continue

        rendered = vis_func(data_sel[i])
        img_u8 = render_to_rgba(
            rendered,
            transparent_border=bool(transparent_border),
            trim=True,
            white_thresh=int(white_thresh),
        )

        ax.imshow(img_u8, interpolation="nearest")

        if pad_frac and pad_frac > 0:
            ax.set_xlim(-pad_frac * img_u8.shape[1], img_u8.shape[1] * (1 + pad_frac))
            ax.set_ylim(img_u8.shape[0] * (1 + pad_frac), -pad_frac * img_u8.shape[0])

        label = None
        if callable(label_func):
            label = label_func(data_sel[i])
        elif labels_sel is not None:
            label = labels_sel[i]

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

    plt.show()
    return fig, np.array(axes, dtype=object)
