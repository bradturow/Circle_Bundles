# circle_bundles/viz/mesh_viz.py
from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Sequence, Tuple, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d


from circle_bundles.viz.image_utils import fig_to_rgba




__all__ = [
    # densities
    "make_density_visualizer",
    # mesh helpers
    "FaceGroup",
    "expand_face_groups",
    # mesh visualizers
    "make_tri_prism_visualizer",
    "make_star_pyramid_visualizer",
    # animation / export
    "fig_to_rgb_array",
    "make_rotating_mesh_clip",
]


# ============================
# Densities
# ============================

def make_density_visualizer(
    *,
    grid_size: int = 32,
    axis: str = "x",
    cmap: str = "inferno",
    normalize: bool = False,
    figsize: Tuple[float, float] = (3.0, 3.0),
    dpi: int = 150,
) -> Callable[[np.ndarray], Figure]:
    """
    Returns a visualization function for densities on a grid_size^3 voxel grid.

    Parameters
    ----------
    grid_size : int
    axis : {'x','y','z'}
        Axis to sum over.
    cmap : str
    normalize : bool
        If True, normalize projection to max=1 for display.
    figsize, dpi : Figure sizing

    Returns
    -------
    vis_func(density) -> matplotlib Figure
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("axis must be one of: 'x', 'y', 'z'")
    ax_idx = axis_map[axis]

    def vis_func(density: np.ndarray) -> Figure:
        arr = np.asarray(density, dtype=float)
        if arr.ndim == 1:
            if arr.size != grid_size**3:
                raise ValueError(f"Density size mismatch: got {arr.size}, expected {grid_size**3}.")
            vol = arr.reshape((grid_size, grid_size, grid_size))
        elif arr.shape == (grid_size, grid_size, grid_size):
            vol = arr
        else:
            raise ValueError(
                f"Density must be flat length {grid_size**3} or shape {(grid_size,)*3}. Got {arr.shape}."
            )

        proj = vol.sum(axis=ax_idx)
        if normalize:
            m = float(np.max(proj))
            if m > 0:
                proj = proj / m

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="none")
        ax.set_axis_off()
        # transpose so x/y look natural in imshow; origin lower for consistent orientation
        ax.imshow(proj.T, origin="lower", cmap=cmap)
        return fig

    return vis_func


# ============================
# Mesh utilities + visualizers
# ============================

FaceGroup = Union[
    Sequence[int],          # explicit indices, e.g. [0,5,9]
    Tuple[int, int],         # range (start, end_exclusive)
]


def expand_face_groups(face_groups: Sequence[FaceGroup]) -> List[List[int]]:
    out: List[List[int]] = []
    for g in face_groups:
        if (
            isinstance(g, tuple)
            and len(g) == 2
            and isinstance(g[0], (int, np.integer))
            and isinstance(g[1], (int, np.integer))
        ):
            start, end = int(g[0]), int(g[1])
            if end <= start:
                raise ValueError(f"Invalid face range (start,end_excl) = {(start, end)}")
            out.append(list(range(start, end)))
        else:
            out.append([int(x) for x in g])
    return out


def make_tri_prism_visualizer(
    mesh,
    face_groups: Sequence[FaceGroup],
    *,
    face_colors_list: Optional[Sequence[str]] = None,
    alpha: float = 1.0,
    show_edges: bool = True,
    edge_color: str = "black",
    edge_width: float = 2.0,
    elev: float = 0.0,
    azim: float = 0.0,
    figsize: Tuple[float, float] = (4.0, 4.0),
    dpi: int = 150,
    depth_sort: bool = True,   # <--- new: manual triangle ordering
) -> Callable[[np.ndarray], Figure]:
    if face_colors_list is None:
        face_colors_list = [
            "#FF6B6B",
            "#FFD93D",
            "#6BCB77",
            "#4D96FF",
            "#C780FA",
        ]

    faces = np.asarray(mesh.faces, dtype=int)
    n_vertices = int(mesh.vertices.shape[0])

    groups = expand_face_groups(face_groups)

    # color per triangle face index
    face_color_map: dict[int, Tuple[float, float, float, float]] = {}
    for gi, grp in enumerate(groups):
        base = tuple(to_rgba(face_colors_list[gi % len(face_colors_list)], alpha))
        for f_idx in grp:
            face_color_map[int(f_idx)] = base

    # Precompute boundary edges once (topology only)
    edge_count: dict[Tuple[int, int], int] = defaultdict(int)
    for f in faces:
        for t in range(3):
            e = tuple(sorted((int(f[t]), int(f[(t + 1) % 3]))))
            edge_count[e] += 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    def vis_func(flat_mesh: np.ndarray) -> Figure:
        verts = np.asarray(flat_mesh, dtype=float).reshape((n_vertices, 3))
        tris = verts[faces]  # (n_faces, 3, 3)

        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="none")
        ax = fig.add_subplot(111, projection="3d", facecolor="none")
        ax.set_axis_off()

        # View first (important: projection depends on this)
        ax.view_init(elev=float(elev), azim=float(azim))

        # Face colors aligned with faces
        facecolors = np.array(
            [face_color_map.get(i, (0.7, 0.7, 0.7, alpha)) for i in range(len(faces))],
            dtype=float,
        )

        # ---- Manual depth sort (back-to-front) ----
        if depth_sort:
            # Project triangle vertices to screen space and sort by projected depth
            M = ax.get_proj()

            # Flatten all triangle vertices for one projection call
            X = tris[:, :, 0].ravel()
            Y = tris[:, :, 1].ravel()
            Z = tris[:, :, 2].ravel()
            x2, y2, z2 = proj3d.proj_transform(X, Y, Z, M)

            z2 = np.asarray(z2).reshape(len(tris), 3)
            tri_depth = z2.mean(axis=1)   # average projected depth

            order = np.argsort(tri_depth)  # back-to-front
            tris = tris[order]
            facecolors = facecolors[order]

        poly = Poly3DCollection(
            tris,
            facecolors=facecolors,
            edgecolor="none",
        )
        # If you want, you can keep this, but don't use "min"
        poly.set_zsort("average")
        ax.add_collection3d(poly)

        if show_edges and boundary_edges:
            segments = [(verts[i], verts[j]) for i, j in boundary_edges]
            lc = Line3DCollection(segments, colors=edge_color, linewidths=edge_width)
            ax.add_collection3d(lc)

        # Equal-ish scaling
        max_range = float(np.ptp(verts, axis=0).max() + 1e-12)
        mid = verts.mean(axis=0)
        lims = [(float(m - max_range / 2), float(m + max_range / 2)) for m in mid]
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_box_aspect([1, 1, 1])

        return fig

    return vis_func



def make_star_pyramid_visualizer(
    mesh,
    *,
    base_color: str = "black",
    edge_color: str = "gray",
    alpha: float = 1.0,
    colormap: str = "twilight",
    figsize: Tuple[float, float] = (4.0, 4.0),
    dpi: int = 150,
    elev: float = 0.0,
    azim: float = 0.0,
) -> Callable[[np.ndarray], Figure]:
    """
    Visualizer for a star pyramid mesh with a smooth gradient on side faces.
    Uses a stable ordering of side faces based on the base-edge midpoint angle in yz-plane.
    """
    faces = np.asarray(mesh.faces, dtype=int)
    n_vertices = int(mesh.vertices.shape[0])
    apex_index = n_vertices - 1
    cmap = plt.get_cmap(colormap)

    def vis_func(flat_mesh: np.ndarray) -> Figure:
        verts = np.asarray(flat_mesh, dtype=float).reshape((n_vertices, 3))
        tris = verts[faces]

        # identify side faces (those containing the apex)
        side_idx = np.array([i for i, f in enumerate(faces) if apex_index in f], dtype=int)

        # stable ordering: angle of midpoint of the edge opposite apex, in yz-plane
        if side_idx.size > 0:
            mids = []
            for fi in side_idx:
                f = faces[int(fi)]
                base_verts = [v for v in f if int(v) != apex_index]
                p = 0.5 * (verts[int(base_verts[0])] + verts[int(base_verts[1])])
                mids.append(np.arctan2(p[2], p[1]))  # angle in yz-plane
            order = np.argsort(np.asarray(mids))
            side_idx_sorted = side_idx[order]
        else:
            side_idx_sorted = side_idx

        # assign colors
        face_colors: List[object] = [base_color] * len(faces)
        if side_idx_sorted.size > 0:
            vals = np.linspace(0.0, 1.0, side_idx_sorted.size, endpoint=True)
            for t, fi in enumerate(side_idx_sorted):
                face_colors[int(fi)] = cmap(float(vals[t]))

        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="none")
        ax = fig.add_subplot(111, projection="3d", facecolor="none")
        ax.set_axis_off()

        poly = Poly3DCollection(tris, facecolors=face_colors, edgecolor=edge_color, alpha=alpha)
        ax.add_collection3d(poly)

        max_range = float(np.ptp(verts, axis=0).max() + 1e-12)
        mid = verts.mean(axis=0)
        lims = [(float(m - max_range / 2), float(m + max_range / 2)) for m in mid]
        ax.set_xlim(*lims[0])
        ax.set_ylim(*lims[1])
        ax.set_zlim(*lims[2])
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=float(elev), azim=float(azim))

        return fig

    return vis_func


# ============================
# Figure -> array + rotating clip
# ============================

def fig_to_rgb_array(fig: Figure) -> np.ndarray:
    """
    Backend-safe conversion of a Matplotlib figure to an RGB uint8 image.
    Uses circle_bundles.viz.image_utils.fig_to_rgba under the hood.
    """
    rgba = fig_to_rgba(fig)            # (H,W,4) uint8
    return rgba[..., :3].copy()        # (H,W,3) uint8


def make_rotating_mesh_clip(
    flat_mesh: np.ndarray,
    vis_func: Callable[[np.ndarray], Figure],
    out_path: str = "rotation.gif",
    *,
    n_frames: int = 120,
    azim_start: float = 0.0,
    azim_end: float = 360.0,
    elev: float = 20.0,
    close_figs: bool = True,
    fps: int = 24,
) -> List[np.ndarray]:
    """
    Create a rotating 3D clip by changing the *camera view* each frame.

    Parameters
    ----------
    flat_mesh : (n_vertices*3,) array
        Flattened vertex array you pass into your vis_func.
    vis_func : callable
        Your mesh visualizer: vis_func(flat_mesh) -> matplotlib Figure.
        (We override the view each frame by finding a 3D axis and calling view_init.)
    out_path : str
        Output path ending with .gif or .mp4.
    n_frames : int
    azim_start, azim_end : float
        Azimuth sweep in degrees.
    elev : float
        Constant elevation angle in degrees.
    close_figs : bool
        Close each frame figure to avoid memory growth.
    fps : int
        Used for mp4; for gif we set frame duration from fps.

    Returns
    -------
    frames : list[np.ndarray]
        List of RGB frames (H,W,3) uint8.
    """
    flat_mesh = np.asarray(flat_mesh)
    frames: List[np.ndarray] = []
    azims = np.linspace(float(azim_start), float(azim_end), int(n_frames), endpoint=False)

    for a in azims:
        fig = vis_func(flat_mesh)

        # Find the first 3D axis and set view.
        ax3d = None
        for ax in fig.axes:
            if hasattr(ax, "view_init"):
                ax3d = ax
                break
        if ax3d is None:
            raise ValueError("vis_func figure did not contain a 3D axis (no view_init found).")

        ax3d.view_init(elev=float(elev), azim=float(a))

        frame = fig_to_rgb_array(fig)
        frames.append(frame)

        if close_figs:
            plt.close(fig)

    # Write output (optional)
    if out_path is not None:
        lower = out_path.lower()
        try:
            import imageio.v2 as imageio
        except Exception as e:
            raise ImportError(
                "Writing gifs/mp4 requires imageio. Install via `pip install imageio`."
            ) from e

        if lower.endswith(".gif"):
            imageio.mimsave(out_path, frames, duration=1.0 / float(fps))
        elif lower.endswith(".mp4"):
            imageio.mimsave(out_path, frames, fps=int(fps))
        else:
            raise ValueError("out_path must end with .gif or .mp4")

    return frames
