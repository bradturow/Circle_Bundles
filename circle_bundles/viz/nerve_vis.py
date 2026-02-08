from __future__ import annotations

from typing import Optional, Dict, Any, Iterable, Tuple, List
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


def _nerve_edges_from_U(U: np.ndarray) -> List[Tuple[int, int]]:
    """Edges (i,j) where U[i] ∩ U[j] is nonempty."""
    U = np.asarray(U, dtype=bool)
    M = int(U.shape[0])
    edges: List[Tuple[int, int]] = []
    for i in range(M):
        Ui = U[i]
        for j in range(i + 1, M):
            if np.any(Ui & U[j]):
                edges.append((i, j))
    return edges


def _nerve_triangles_from_U(U: np.ndarray) -> List[Tuple[int, int, int]]:
    """Triangles (i,j,k) where U[i] ∩ U[j] ∩ U[k] is nonempty."""
    U = np.asarray(U, dtype=bool)
    M = int(U.shape[0])
    tris: List[Tuple[int, int, int]] = []
    for i in range(M):
        Ui = U[i]
        for j in range(i + 1, M):
            Uij = Ui & U[j]
            if not np.any(Uij):
                continue
            for k in range(j + 1, M):
                if np.any(Uij & U[k]):
                    tris.append((i, j, k))
    return tris


def nerve_vis(
    landmarks: np.ndarray,
    *,
    # Provide either U OR explicit simplices
    U: Optional[np.ndarray] = None,
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
    ax=None,
):
    """
    Visualize (a subcomplex of) the witnessed Čech nerve.

    Parameters
    ----------
    landmarks:
        (M,2) or (M,3) coordinates of the nerve vertices (these are *the* plotting coords).
    U:
        Optional (M, n_samples) boolean membership matrix.
        Used only to infer edges/triangles when they are not explicitly provided.
    vertices / edges / triangles:
        Optional explicit simplices to draw. If provided, they take precedence over U.

    Rules
    -----
    - If `edges` is None and max_dim>=1: we infer edges from U (requires U).
    - If `triangles` is None and max_dim>=2: we infer triangles from U (requires U).
    - Vertex set:
        * If `vertices` is provided: use exactly those.
        * Else if `draw_all_vertices`: draw all 0..M-1.
        * Else: draw vertices that appear in edges/triangles (after they are determined).
    """
    cochains = cochains or {}
    base_colors = base_colors or {0: "lightblue", 1: "black", 2: "pink"}
    cochain_cmaps = cochain_cmaps or {}

    if int(max_dim) not in (0, 1, 2):
        raise ValueError(f"max_dim must be 0, 1, or 2. Got {max_dim}.")

    landmarks = np.asarray(landmarks, dtype=float)
    if landmarks.ndim != 2:
        raise ValueError(f"landmarks must be 2D (M,2) or (M,3). Got shape {landmarks.shape}.")

    M = int(landmarks.shape[0])
    dim = int(landmarks.shape[1])
    if dim not in (2, 3):
        raise ValueError(f"landmarks must have 2 or 3 columns. Got {dim}.")
    is_3d = (dim == 3)

    if U is not None:
        U = np.asarray(U, dtype=bool)
        if U.ndim != 2:
            raise ValueError(f"U must be 2D (M, n_samples). Got shape {U.shape}.")
        if int(U.shape[0]) != M:
            raise ValueError(f"n_sets mismatch: U has {U.shape[0]} rows but landmarks has {M} rows.")

    # --- simplices: explicit overrides OR infer from U ---
    if max_dim >= 1:
        if edges is None:
            if U is None:
                edges_list: List[Tuple[int, int]] = []
            else:
                edges_list = _nerve_edges_from_U(U)
        else:
            edges_list = [tuple(int(x) for x in e) for e in edges]
    else:
        edges_list = []

    if max_dim >= 2:
        if triangles is None:
            if U is None:
                tris_list: List[Tuple[int, int, int]] = []
            else:
                tris_list = _nerve_triangles_from_U(U)
        else:
            tris_list = [tuple(int(x) for x in t) for t in triangles]
    else:
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
                used.add(int(i)); used.add(int(j))
            for (i, j, k) in tris_list:
                used.add(int(i)); used.add(int(j)); used.add(int(k))
            verts = [(i,) for i in sorted(used)]

    # If user provided vertices, restrict edges/tris to those vertices
    if vertices is not None:
        Vset = {int(i[0]) for i in verts}
        edges_list = [(i, j) for (i, j) in edges_list if int(i) in Vset and int(j) in Vset]
        tris_list = [(i, j, k) for (i, j, k) in tris_list
                     if (int(i) in Vset and int(j) in Vset and int(k) in Vset)]

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
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
    else:
        fig = ax.figure
        if is_3d and getattr(ax, "name", "") != "3d":
            raise ValueError("landmarks are 3D but provided ax is not a 3D axes.")
        if (not is_3d) and getattr(ax, "name", "") == "3d":
            raise ValueError("landmarks are 2D but provided ax is a 3D axes.")

    # --- triangles ---
    for s in tris_list:
        s = (int(s[0]), int(s[1]), int(s[2]))
        if any((i < 0 or i >= M) for i in s):
            continue
        pts = landmarks[list(s), :]
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
        pts = landmarks[[i, j], :]
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
        p = landmarks[i]
        col = color_for(0, (i,), base_colors.get(0, "lightblue"))

        if is_3d:
            ax.scatter(*p, color=col, s=float(node_size) ** 2, edgecolors="black", linewidths=1)
            if node_labels is not None:
                lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
                ax.text(*p, str(lab), fontsize=int(fontsize), color=font_color, ha="center", va="center")
        else:
            ax.scatter(
                p[0], p[1],
                color=col,
                s=float(node_size) ** 2,
                edgecolors="black",
                linewidths=1,
                zorder=10,
            )
            if node_labels is not None:
                lab = node_labels[i] if not isinstance(node_labels, dict) else node_labels.get(i, i)
                ax.text(
                    p[0], p[1],
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

                x3, y3, z3 = landmarks[i]
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
                    landmarks[i, :2],
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

    # --- save ---
    if save_path is not None:
        save_path = str(save_path)
        root, ext = os.path.splitext(save_path)
        if ext.lower() == ".pkl":
            save_path = root + ".png"
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    if show and ax is not None:
        plt.show()

    return fig, ax
