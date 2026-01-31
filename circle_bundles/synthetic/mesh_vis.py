# synthetic/mesh_vis.py
from __future__ import annotations

from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.colors import LinearSegmentedColormap


# NOTE: we keep using your existing canonical renderer from circle_bundles.
# If you ever want synthetic to be standalone, we can move fig_to_rgba into synthetic/viz_utils.py.
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
    grid_size = int(grid_size)
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
    Sequence[int],      # explicit indices, e.g. [0,5,9]
    Tuple[int, int],    # range (start, end_exclusive)
]


def expand_face_groups(face_groups: Sequence[FaceGroup]) -> List[List[int]]:
    """Expand face_groups into explicit lists of face indices."""
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
    face_groups: Optional[Sequence[FaceGroup]] = None,
    *,
    face_colors_list: Optional[Sequence[str]] = None,
    alpha: float = 1.0,
    show_edges: bool = True,
    edge_color: str = "black",
    edge_width: float = 2.5,
    elev: float = 0.0,
    azim: float = 0.0,
    figsize: Tuple[float, float] = (4.0, 4.0),
    dpi: int = 150,
    depth_sort: bool = True,
) -> Callable[[np.ndarray], Figure]:
    """
    Visualize a triangular prism-style mesh with custom face group coloring.

    Parameters
    ----------
    mesh : trimesh.Trimesh-like
        Must have .vertices and .faces.
    face_groups : groups of face indices (explicit lists or (start,end_excl) ranges)
    depth_sort : bool
        If True, manually sorts triangles back-to-front using projected depth
        (helps with alpha blending / occlusion in Matplotlib).

    Returns
    -------
    vis_func(flat_mesh) -> Figure
        flat_mesh is expected to be (n_vertices*3,) giving vertex positions.
    """
    if face_colors_list is None:
        face_colors_list = [
            '#FFB4A2',  # Rectangular Side 2 – coral blush
            '#FAE3B4',  # Soft buttery yellow
            '#A8DADC',  # Rectangular Side 1 – seafoam
            '#E3C8F2',  # Triangle Face 1 (top)
            '#9BB1FF'   # Rectangular Side 3 – pastel periwinkle
        ]

    n_vertices = int(np.asarray(mesh.vertices).shape[0])        
    faces = np.asarray(mesh.faces, dtype=int)
    n_faces = int(faces.shape[0])

    if face_groups is None:
        # default only when topology matches your canonical tri-prism triangulation
        if n_faces == 8:
            face_groups = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 8)]
        else:
            raise ValueError(
                "face_groups=None only supported for the canonical tri-prism mesh "
                "(expected 8 triangle faces). Please pass face_groups explicitly."
            )    
    
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

        # Set view early (projection depends on it)
        ax.view_init(elev=float(elev), azim=float(azim))

        facecolors = np.array(
            [face_color_map.get(i, (0.7, 0.7, 0.7, alpha)) for i in range(len(faces))],
            dtype=float,
        )

        if depth_sort:
            M = ax.get_proj()
            X = tris[:, :, 0].ravel()
            Y = tris[:, :, 1].ravel()
            Z = tris[:, :, 2].ravel()
            _x2, _y2, z2 = proj3d.proj_transform(X, Y, Z, M)

            z2 = np.asarray(z2).reshape(len(tris), 3)
            tri_depth = z2.mean(axis=1)
            order = np.argsort(tri_depth)  # back-to-front

            tris = tris[order]
            facecolors = facecolors[order]

        poly = Poly3DCollection(tris, facecolors=facecolors, edgecolor="none")
        poly.set_zsort("average")
        ax.add_collection3d(poly)

        if show_edges and boundary_edges:
            segments = [(verts[i], verts[j]) for i, j in boundary_edges]
            lc = Line3DCollection(segments, colors=edge_color, linewidths=float(edge_width))
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



def _default_pastel_gradient(n: int) -> LinearSegmentedColormap:
    """
    Pastel gradient that exactly matches the 5-color palette when n=5
    and smoothly interpolates for other n.
    """
    base_colors = ['#FFB4A2', '#FAE3B4', '#A8DADC', '#E3C8F2', '#9BB1FF']
    return LinearSegmentedColormap.from_list(
        f'pastel_sunset_{n}',
        base_colors,
        N=max(n, len(base_colors)),
    )



def make_star_pyramid_visualizer(
    mesh,
    *,
    base_color: str = "#94A3B8",
    edge_color: str = "gray",
    alpha: float = 1.0,
    colormap: str | LinearSegmentedColormap | None = None,
    figsize: Tuple[float, float] = (4.0, 4.0),
    dpi: int = 150,
    elev: float = 0.0,
    azim: float = 0.0,
) -> Callable[[np.ndarray], Figure]:
    """
    Visualizer for a star pyramid mesh with a smooth gradient on side faces.

    FIX: side-face ordering is computed ONCE from the template mesh vertices,
    so colors stay attached to the same faces under rotation.
    """
    faces = np.asarray(mesh.faces, dtype=int)
    verts0 = np.asarray(mesh.vertices, dtype=float)
    n_vertices = int(verts0.shape[0])
    apex_index = n_vertices - 1

    # ----------------------------
    # Precompute stable side-face ordering from template mesh
    # ----------------------------
    side_idx = np.array([i for i, f in enumerate(faces) if apex_index in f], dtype=int)

    if side_idx.size > 0:
        mids = []
        for fi in side_idx:
            f = faces[int(fi)]
            base_verts = [v for v in f if int(v) != apex_index]
            p = 0.5 * (verts0[int(base_verts[0])] + verts0[int(base_verts[1])])

            # Keep your original convention (yz-plane) BUT evaluated on verts0,
            # so it becomes a fixed ordering.
            mids.append(np.arctan2(p[2], p[1]))

        order = np.argsort(np.asarray(mids))
        side_idx_sorted_template = side_idx[order]
    else:
        side_idx_sorted_template = side_idx

    # Choose colormap once (length depends on number of side faces)
    if isinstance(colormap, LinearSegmentedColormap):
        cmap = colormap
    elif isinstance(colormap, str):
        cmap = plt.get_cmap(colormap)
    else:
        cmap = _default_pastel_gradient(len(side_idx_sorted_template))

    # Precompute the face color list (again: fixed per face index)
    face_colors_template: List[object] = [base_color] * len(faces)
    if side_idx_sorted_template.size > 0:
        vals = np.linspace(0.0, 1.0, side_idx_sorted_template.size, endpoint=True)
        for t, fi in enumerate(side_idx_sorted_template):
            face_colors_template[int(fi)] = cmap(float(vals[t]))

    def vis_func(flat_mesh: np.ndarray) -> Figure:
        verts = np.asarray(flat_mesh, dtype=float).reshape((n_vertices, 3))
        tris = verts[faces]

        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="none")
        ax = fig.add_subplot(111, projection="3d", facecolor="none")
        ax.set_axis_off()

        poly = Poly3DCollection(
            tris,
            facecolors=face_colors_template,  # <-- fixed mapping
            edgecolor=edge_color,
            alpha=float(alpha),
        )
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
    rgba = fig_to_rgba(fig)     # (H,W,4) uint8
    return rgba[..., :3].copy() # (H,W,3) uint8


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
    Create a rotating 3D clip by changing the camera view each frame.

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

        ax3d = None
        for ax in fig.axes:
            if hasattr(ax, "view_init"):
                ax3d = ax
                break
        if ax3d is None:
            raise ValueError("vis_func figure did not contain a 3D axis (no view_init found).")

        ax3d.view_init(elev=float(elev), azim=float(a))

        frames.append(fig_to_rgb_array(fig))
        if close_figs:
            plt.close(fig)

    if out_path is not None:
        lower = out_path.lower()
        try:
            import imageio.v2 as imageio
        except Exception as e:  # pragma: no cover
            raise ImportError("Writing gifs/mp4 requires imageio (`pip install imageio`).") from e

        if lower.endswith(".gif"):
            imageio.mimsave(out_path, frames, duration=1.0 / float(fps))
        elif lower.endswith(".mp4"):
            imageio.mimsave(out_path, frames, fps=int(fps))
        else:
            raise ValueError("out_path must end with .gif or .mp4")

    return frames
