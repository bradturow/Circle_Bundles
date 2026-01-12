from __future__ import annotations

from typing import Optional, Dict, Any, Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform as mpl_proj_transform

from .image_utils import render_to_rgba


def value_to_rgb(val, cmap, vmin=None, vmax=None):
    """
    Map a scalar/discrete value to an RGBA-ish color.

    - If cmap is dict: discrete mapping, returns cmap.get(val, "gray")
    - Else: cmap is a Matplotlib colormap name or Colormap object
    """
    if isinstance(cmap, dict):
        return cmap.get(val, "gray")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return cm_obj(norm(val))


def _as_simplex_tuple(simplex: Any) -> Tuple[int, ...]:
    """Coerce a simplex representation to a sorted tuple of ints."""
    if isinstance(simplex, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in simplex)
    return (int(simplex),)


def nerve_vis(
    nerve,
    landmarks,
    *,
    cochains: Optional[Dict[int, Dict[Any, Any]]] = None,
    base_colors: Optional[dict] = None,
    cochain_cmaps: Optional[dict] = None,
    opacity: float = 0.5,
    node_size: float = 40,
    line_width: float = 2,
    node_labels=None,
    fontsize: int = 12,
    font_color: str = "black",
    vis_func=None,
    data=None,
    image_zoom: float = 0.2,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    # NEW:
    draw_all_vertices: bool = True,
    show: bool = True,
):
    """
    Visualize a nerve (Gudhi SimplexTree-like) with optional cochains.

    IMPORTANT FIX:
      If draw_all_vertices=True (default), nodes are drawn for *all* landmark indices,
      even if they are isolated and do not appear in nerve.get_simplices().
    """
    cochains = cochains or {}
    base_colors = base_colors or {0: "lightblue", 1: "black", 2: "pink"}
    cochain_cmaps = cochain_cmaps or {}

    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.ndim != 2:
        raise ValueError(f"landmarks must be 2D (N,d). Got shape {landmarks.shape}.")
    N, dim = landmarks.shape
    is_3d = (dim == 3)

    # --- pull simplices from nerve ---
    simplices = [tuple(int(x) for x in s) for (s, _) in nerve.get_simplices()]
    edges = [s for s in simplices if len(s) == 2]
    tris  = [s for s in simplices if len(s) == 3]

    if draw_all_vertices:
        verts = [(i,) for i in range(N)]
    else:
        verts = [s for s in simplices if len(s) == 1]

    # --- vmin/vmax for numeric cochains ---
    vmin_vmax: dict[int, tuple[float, float]] = {}
    for k, cdict in cochains.items():
        if not cdict:
            continue
        vals = [v for v in cdict.values() if np.isscalar(v)]
        if vals:
            vmin_vmax[int(k)] = (float(min(vals)), float(max(vals)))

    def cochain_lookup(k: int, simplex):
        c = cochains.get(k)
        if not c:
            return None

        s = _as_simplex_tuple(simplex)

        if k == 0:
            i = int(s[0])
            # accept keys: i, (i,), etc.
            return c.get(i, c.get((i,), None))

        # k >= 1: accept either orientation or sorted
        if s in c:
            return c[s]
        s_sorted = tuple(sorted(s))
        if s_sorted in c:
            return c[s_sorted]
        if k == 1 and len(s) == 2:
            rev = (s[1], s[0])
            if rev in c:
                return c[rev]
        return None

    def color_for(k: int, simplex, default):
        val = cochain_lookup(k, simplex)
        if val is None:
            return default
        vmin, vmax = vmin_vmax.get(int(k), (None, None))
        return value_to_rgb(val, cochain_cmaps.get(k, default), vmin, vmax)

    # --- figure/axes ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    # --- triangles ---
    for s in tris:
        if any((i < 0 or i >= N) for i in s):
            continue
        pts = landmarks[list(s), :]
        col = color_for(2, s, base_colors.get(2, "pink"))
        if is_3d:
            tri = Poly3DCollection([pts], facecolor=col, alpha=float(opacity), edgecolor="k")
            ax.add_collection3d(tri)
        else:
            ax.add_patch(patches.Polygon(pts[:, :2], closed=True, color=col, alpha=float(opacity)))

    # --- edges ---
    for (i, j) in edges:
        i, j = int(i), int(j)
        if not (0 <= i < N and 0 <= j < N) or i == j:
            continue
        pts = landmarks[[i, j], :]
        col = color_for(1, (i, j), base_colors.get(1, "black"))
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, linewidth=float(line_width))
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=col, linewidth=float(line_width))

    # --- nodes ---
    for (i,) in verts:
        i = int(i)
        if not (0 <= i < N):
            continue
        p = landmarks[i]
        col = color_for(0, (i,), base_colors.get(0, "lightblue"))

        if is_3d:
            ax.scatter(*p, color=col, s=float(node_size) ** 2, edgecolors="black", linewidths=1)
            if node_labels is not None:
                lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
                ax.text(*p, str(lab), fontsize=int(fontsize), color=font_color, ha="center", va="center")
        else:
            ax.scatter(p[0], p[1], color=col, s=float(node_size) ** 2,
                       edgecolors="black", linewidths=1, zorder=10)
            if node_labels is not None:
                lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
                ax.text(p[0], p[1], str(lab), fontsize=int(fontsize), color=font_color,
                        ha="center", va="center", zorder=11)

    # --- image overlays ---
    if vis_func is not None and data is not None:
        fig.canvas.draw()
        if is_3d:
            for (i,) in verts:
                i = int(i)
                if not (0 <= i < N):
                    continue
                img = render_to_rgba(vis_func(data[i]), transparent_border=True, trim=True)
                im = OffsetImage(img, zoom=float(image_zoom))

                x3, y3, z3 = landmarks[i]
                x2, y2, _ = mpl_proj_transform(x3, y3, z3, ax.get_proj())
                disp = ax.transData.transform((x2, y2))
                fig_xy = fig.transFigure.inverted().transform(disp)

                ab = AnnotationBbox(im, fig_xy, xycoords="figure fraction",
                                    frameon=False, pad=0.0, zorder=9999)
                fig.add_artist(ab)
        else:
            for (i,) in verts:
                i = int(i)
                if not (0 <= i < N):
                    continue
                img = render_to_rgba(vis_func(data[i]), transparent_border=True, trim=True)
                im = OffsetImage(img, zoom=float(image_zoom))
                ab = AnnotationBbox(im, landmarks[i, :2], xycoords="data",
                                    frameon=False, pad=0.0, zorder=9999)
                ax.add_artist(ab)

    if is_3d:
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_aspect("equal")

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=int(fontsize) + 2)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig, ax
