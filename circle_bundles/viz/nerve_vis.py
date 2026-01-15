from __future__ import annotations

from typing import Optional, Dict, Any, Iterable, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.proj3d import proj_transform as mpl_proj_transform

from .image_utils import render_to_rgba


def value_to_rgb(val, cmap, vmin=None, vmax=None):
    """Map a scalar/discrete value to a color."""
    if isinstance(cmap, dict):
        return cmap.get(val, "gray")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cm_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return cm_obj(norm(val))


def _as_simplex_tuple(simplex: Any) -> Tuple[int, ...]:
    """Coerce a simplex representation to a tuple of ints."""
    if isinstance(simplex, (list, tuple, np.ndarray)):
        return tuple(int(x) for x in simplex)
    return (int(simplex),)


def nerve_vis(
    cover,
    landmark_coords=None,
    *,
    # --- NEW: allow drawing an arbitrary subcomplex (optional overrides) ---
    vertices: Optional[Iterable[int]] = None,
    edges: Optional[Iterable[Tuple[int, int]]] = None,
    triangles: Optional[Iterable[Tuple[int, int, int]]] = None,
    # --- styling / data ---
    cochains: Optional[Dict[int, Dict[Any, Any]]] = None,
    base_colors: Optional[dict] = None,
    cochain_cmaps: Optional[dict] = None,
    max_dim: int = 1,  # default: show 1-skeleton
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
    draw_all_vertices: bool = True,
    show: bool = True,
    # --- NEW: allow composing multi-panel figures ---
    ax=None,
):
    """
    Visualize (a subcomplex of) the witnessed Čech nerve of a CoverBase-like object.

    Default behavior
    ----------------
    - Uses cover.nerve_edges() / cover.nerve_triangles()
    - Draws only up to max_dim=1 (vertices+edges)
    - If landmark_coords is None, uses cover.landmarks[:, :2]

    Subcomplex behavior
    -------------------
    Pass vertices/edges/triangles to draw an arbitrary subcomplex.
    - If edges/triangles are provided, we use exactly those (filtered by max_dim).
    - If vertices is provided, we draw exactly those vertices (unless draw_all_vertices=True and vertices is None).

    Axes behavior
    -------------
    If ax is provided, draw into that axes and DO NOT create a new figure.
    In that case, you probably want show=False (so the caller can plt.show() once).
    """
    cochains = cochains or {}
    base_colors = base_colors or {0: "lightblue", 1: "black", 2: "pink"}
    cochain_cmaps = cochain_cmaps or {}

    if int(max_dim) not in (0, 1, 2):
        raise ValueError(f"max_dim must be 0, 1, or 2. Got {max_dim}.")

    # build/validate
    if hasattr(cover, "ensure_built"):
        cover.ensure_built()

    landmarks = np.asarray(cover.landmarks, dtype=float)
    if landmarks.ndim != 2:
        raise ValueError(f"cover.landmarks must be 2D. Got shape {landmarks.shape}.")
    M = int(landmarks.shape[0])

    # default coords = first two coordinates
    if landmark_coords is None:
        if landmarks.shape[1] < 2:
            raise ValueError("cover.landmarks must have at least 2 columns for default landmark_coords.")
        landmark_coords = landmarks[:, :2]
    else:
        landmark_coords = np.asarray(landmark_coords, dtype=float)
        if landmark_coords.ndim != 2:
            raise ValueError(f"landmark_coords must be 2D (M,d). Got shape {landmark_coords.shape}.")
        if int(landmark_coords.shape[0]) != M:
            raise ValueError(
                f"landmark_coords has {landmark_coords.shape[0]} rows but cover.landmarks has {M}."
            )

    dim = int(landmark_coords.shape[1])
    is_3d = (dim == 3)

    # --- simplices: either overrides or full nerve ---
    if edges is None and max_dim >= 1:
        edges_list = [tuple(int(x) for x in e) for e in cover.nerve_edges()]
    else:
        edges_list = [tuple(int(x) for x in e) for e in (edges or [])]

    if triangles is None and max_dim >= 2:
        tris_list = [tuple(int(x) for x in t) for t in cover.nerve_triangles()]
    else:
        tris_list = [tuple(int(x) for x in t) for t in (triangles or [])]

    # enforce max_dim
    if max_dim < 2:
        tris_list = []
    if max_dim < 1:
        edges_list = []
        tris_list = []

    # --- vertex list ---
    if vertices is not None:
        verts = [(int(i),) for i in vertices]
    else:
        if draw_all_vertices:
            verts = [(i,) for i in range(M)]
        else:
            used = set()
            for (i, j) in edges_list:
                used.add(i)
                used.add(j)
            for (i, j, k) in tris_list:
                used.add(i)
                used.add(j)
                used.add(k)
            verts = [(i,) for i in sorted(used)]

    # If user provided vertices, restrict edges/tris to those vertices
    if vertices is not None:
        Vset = {int(i[0]) for i in verts}
        edges_list = [(i, j) for (i, j) in edges_list if i in Vset and j in Vset]
        tris_list = [(i, j, k) for (i, j, k) in tris_list if (i in Vset and j in Vset and k in Vset)]

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
            return c.get(i, c.get((i,), None))

        # k>=1: allow either exact, sorted, or reversed for edges
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

    # --- figure/axes (NEW: support drawing into user-provided ax) ---
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
    else:
        fig = ax.figure
        # If caller supplies a 2D ax but coords are 3D, fail loudly
        if is_3d and getattr(ax, "name", "") != "3d":
            raise ValueError("landmark_coords are 3D but provided ax is not a 3D axes.")
        if (not is_3d) and getattr(ax, "name", "") == "3d":
            raise ValueError("landmark_coords are 2D but provided ax is a 3D axes.")

    # --- triangles ---
    for s in tris_list:
        if any((i < 0 or i >= M) for i in s):
            continue
        pts = landmark_coords[list(s), :]
        col = color_for(2, s, base_colors.get(2, "pink"))
        if is_3d:
            tri = Poly3DCollection([pts], facecolor=col, alpha=float(opacity), edgecolor="k")
            ax.add_collection3d(tri)
        else:
            ax.add_patch(patches.Polygon(pts[:, :2], closed=True, color=col, alpha=float(opacity)))

    # --- edges ---
    for (i, j) in edges_list:
        i, j = int(i), int(j)
        if not (0 <= i < M and 0 <= j < M) or i == j:
            continue
        pts = landmark_coords[[i, j], :]
        col = color_for(1, (i, j), base_colors.get(1, "black"))
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, linewidth=float(line_width))
        else:
            ax.plot(pts[:, 0], pts[:, 1], color=col, linewidth=float(line_width))

    # --- nodes ---
    for (i,) in verts:
        i = int(i)
        if not (0 <= i < M):
            continue
        p = landmark_coords[i]
        col = color_for(0, (i,), base_colors.get(0, "lightblue"))

        if is_3d:
            ax.scatter(*p, color=col, s=float(node_size) ** 2, edgecolors="black", linewidths=1)
            if node_labels is not None:
                lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
                ax.text(*p, str(lab), fontsize=int(fontsize), color=font_color, ha="center", va="center")
        else:
            ax.scatter(
                p[0],
                p[1],
                color=col,
                s=float(node_size) ** 2,
                edgecolors="black",
                linewidths=1,
                zorder=10,
            )
            if node_labels is not None:
                lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
                ax.text(
                    p[0],
                    p[1],
                    str(lab),
                    fontsize=int(fontsize),
                    color=font_color,
                    ha="center",
                    va="center",
                    zorder=11,
                )

    # --- image overlays (on vertices) ---
    if vis_func is not None and data is not None:
        fig.canvas.draw()
        if is_3d:
            for (i,) in verts:
                i = int(i)
                img = render_to_rgba(vis_func(data[i]), transparent_border=True, trim=True)
                im = OffsetImage(img, zoom=float(image_zoom))

                x3, y3, z3 = landmark_coords[i]
                x2, y2, _ = mpl_proj_transform(x3, y3, z3, ax.get_proj())
                disp = ax.transData.transform((x2, y2))
                fig_xy = fig.transFigure.inverted().transform(disp)

                ab = AnnotationBbox(
                    im,
                    fig_xy,
                    xycoords="figure fraction",
                    frameon=False,
                    pad=0.0,
                    zorder=9999,
                )
                fig.add_artist(ab)
        else:
            for (i,) in verts:
                i = int(i)
                img = render_to_rgba(vis_func(data[i]), transparent_border=True, trim=True)
                im = OffsetImage(img, zoom=float(image_zoom))
                ab = AnnotationBbox(
                    im,
                    landmark_coords[i, :2],
                    xycoords="data",
                    frameon=False,
                    pad=0.0,
                    zorder=9999,
                )
                ax.add_artist(ab)

    # --- axes cosmetics ---
    if is_3d:
        ax.set_box_aspect([1, 1, 1])
    else:
        ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=int(fontsize) + 2)

    # --- save (guard against .pkl etc.) ---
    if save_path is not None:
        save_path = str(save_path)
        root, ext = os.path.splitext(save_path)
        if ext.lower() == ".pkl":
            # auto-fix to png (avoids the common notebook bug)
            save_path = root + ".png"
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show and ax is not None:
        plt.show()

    return fig, ax
