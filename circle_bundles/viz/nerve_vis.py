from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform as mpl_proj_transform

from .image_utils import render_to_rgba


def value_to_rgb(val, cmap, vmin=None, vmax=None):
    """
    Map a value to an RGBA color.

    - If cmap is dict: discrete mapping, returns cmap[val] or 'gray'
    - Else: cmap is a Matplotlib colormap name or Colormap object
    """
    if isinstance(cmap, dict):
        return cmap.get(val, "gray")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return cm_obj(norm(val))


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
):
    cochains = cochains or {}
    base_colors = base_colors or {0: "lightblue", 1: "black", 2: "pink"}
    cochain_cmaps = cochain_cmaps or {}

    landmarks = np.asarray(landmarks)
    dim = landmarks.shape[1]
    is_3d = (dim == 3)

    simplices = [tuple(s) for (s, _) in nerve.get_simplices()]
    verts = [s for s in simplices if len(s) == 1]
    edges = [s for s in simplices if len(s) == 2]
    tris = [s for s in simplices if len(s) == 3]

    vmin_vmax = {}
    for k, cdict in cochains.items():
        if not cdict:
            continue
        vals = [v for v in cdict.values() if np.isscalar(v)]
        if vals:
            vmin_vmax[k] = (min(vals), max(vals))

    def cochain_lookup(k: int, simplex):
        c = cochains.get(k)
        if not c:
            return None
        s = tuple(simplex)
        if k == 0:
            return c.get(s[0], c.get((s[0],), None))
        s_sorted = tuple(sorted(s))
        if s in c:
            return c[s]
        if s_sorted in c:
            return c[s_sorted]
        if k == 1 and (s[1], s[0]) in c:
            return c[(s[1], s[0])]
        return None

    def color_for(k: int, simplex, default):
        val = cochain_lookup(k, simplex)
        if val is None:
            return default
        vmin, vmax = vmin_vmax.get(k, (None, None))
        return value_to_rgb(val, cochain_cmaps.get(k, default), vmin, vmax)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)

    # triangles
    for s in tris:
        pts = landmarks[list(s), :]
        col = color_for(2, s, base_colors.get(2, "pink"))
        if is_3d:
            tri = Poly3DCollection([pts], facecolor=col, alpha=opacity, edgecolor="k")
            ax.add_collection3d(tri)
        else:
            ax.add_patch(patches.Polygon(pts[:, :2], closed=True, color=col, alpha=opacity))

    # edges
    for (i, j) in edges:
        pts = landmarks[[i, j], :]
        col = color_for(1, (i, j), base_colors.get(1, "black"))
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, linewidth=line_width)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=col, linewidth=line_width)

    # nodes
    for (i,) in verts:
        p = landmarks[i]
        col = color_for(0, (i,), base_colors.get(0, "lightblue"))
        if is_3d:
            ax.scatter(*p, color=col, s=node_size**2, edgecolors="black", linewidths=1)
            if node_labels is not None:
                ax.text(*p, str(node_labels[i]), fontsize=fontsize, color=font_color,
                        ha="center", va="center")
        else:
            ax.scatter(p[0], p[1], color=col, s=node_size**2, edgecolors="black", linewidths=1, zorder=10)
            if node_labels is not None:
                ax.text(p[0], p[1], str(node_labels[i]), fontsize=fontsize, color=font_color,
                        ha="center", va="center", zorder=11)

    # image overlays
    if vis_func is not None and data is not None:
        fig.canvas.draw()
        if is_3d:
            for (i,) in verts:
                img = render_to_rgba(vis_func(data[i]), transparent_border=True, trim=True)
                im = OffsetImage(img, zoom=image_zoom)

                x3, y3, z3 = landmarks[i]
                x2, y2, _ = mpl_proj_transform(x3, y3, z3, ax.get_proj())
                disp = ax.transData.transform((x2, y2))
                fig_xy = fig.transFigure.inverted().transform(disp)

                ab = AnnotationBbox(im, fig_xy, xycoords="figure fraction",
                                    frameon=False, pad=0.0, zorder=9999)
                fig.add_artist(ab)
        else:
            for (i,) in verts:
                img = render_to_rgba(vis_func(data[i]), transparent_border=True, trim=True)
                im = OffsetImage(img, zoom=image_zoom)
                ab = AnnotationBbox(im, landmarks[i, :2], xycoords="data",
                                    frameon=False, pad=0.0, zorder=9999)
                ax.add_artist(ab)

    if is_3d:
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_aspect("equal")

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=fontsize + 2)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()
    return fig, ax


def nerve_vis_adv(
    node_coords_3d,
    edges,
    vis_func,
    *,
    vis_data=None,
    cochains=None,
    cochain_cmaps=None,
    zoom: float = 0.3,
    figsize=(10, 8),
    node_size: float = 4,
    line_width: float = 1,
    node_labels=None,
    label_fontsize: int = 10,
    label_color: str = "black",
    edge_width_map=None,
):
    node_coords_3d = np.asarray(node_coords_3d, dtype=float)
    cochains = cochains or {}
    cochain_cmaps = cochain_cmaps or {}

    # stable projection via temp 3D axes
    fig3d = plt.figure(figsize=(4, 4))
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.scatter(node_coords_3d[:, 0], node_coords_3d[:, 1], node_coords_3d[:, 2], s=1)

    mins = node_coords_3d.min(axis=0)
    maxs = node_coords_3d.max(axis=0)
    mid = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins) + 1e-12)
    ax3d.set_xlim(mid[0] - span / 2, mid[0] + span / 2)
    ax3d.set_ylim(mid[1] - span / 2, mid[1] + span / 2)
    ax3d.set_zlim(mid[2] - span / 2, mid[2] + span / 2)

    fig3d.canvas.draw()
    P = ax3d.get_proj()

    projected = []
    for x, y, z in node_coords_3d:
        x2, y2, _ = mpl_proj_transform(x, y, z, P)
        projected.append((x2, y2))
    plt.close(fig3d)
    projected = np.asarray(projected)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")

    # vmin/vmax for numeric node cochains if needed
    vmin_vmax = {}
    if 0 in cochains and cochains[0]:
        vals0 = [v for v in cochains[0].values() if np.isscalar(v)]
        if vals0:
            vmin_vmax[0] = (min(vals0), max(vals0))

    default_width = float(line_width) if line_width is not None else 1.0

    # edges
    for i, j in edges:
        pi = projected[i]
        pj = projected[j]

        col = "gray"
        if 1 in cochains and cochains[1]:
            key = (i, j) if (i, j) in cochains[1] else ((j, i) if (j, i) in cochains[1] else None)
            if key is not None:
                val = cochains[1][key]
                cmap = cochain_cmaps.get(1, {})
                col = cmap.get(val, "gray") if isinstance(cmap, dict) else value_to_rgb(val, cmap, None, None)

        width = default_width
        if edge_width_map is not None:
            width = edge_width_map.get((i, j), edge_width_map.get((j, i), width))

        ax.plot([pi[0], pj[0]], [pi[1], pj[1]], color=col, linewidth=width, zorder=1)

    # nodes + images + labels
    for i, (x2, y2) in enumerate(projected):
        if vis_data is not None:
            img = render_to_rgba(vis_func(vis_data[i]), transparent_border=True, trim=True)
            ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=float(zoom)), (x2, y2), frameon=False))

        dot_col = "black"
        if 0 in cochains and cochains[0]:
            val = cochains[0].get((i,), cochains[0].get(i, None))
            if val is not None:
                dot_col = value_to_rgb(val, cochain_cmaps.get(0), *vmin_vmax.get(0, (None, None)))

        ax.plot(x2, y2, "o", markersize=float(node_size), color=dot_col, zorder=10)

        if node_labels is not None:
            lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
            ax.text(
                x2, y2, str(lab),
                fontsize=int(label_fontsize),
                color=label_color,
                ha="center", va="center", zorder=30
            )

    ax.axis("off")
    plt.tight_layout()
    return fig, ax
