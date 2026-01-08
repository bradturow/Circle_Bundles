# circle_bundles/synthetic/meshes.py
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import trimesh
from shapely.geometry import Polygon
from trimesh.creation import triangulate_polygon

__all__ = [
    "make_tri_prism",
    "make_star_pyramid",
]


def make_tri_prism(
    *,
    height: float = 1.0,
    radius: float = 1.0,
) -> Tuple[trimesh.Trimesh, List[Tuple[int, int]]]:
    """
    Create a triangular prism with base an equilateral triangle in the (y,z)-plane
    and extrusion along the x-axis.

    Returns
    -------
    mesh : trimesh.Trimesh
    face_groups : list[(start, end_exclusive)]
        Ranges of triangle faces belonging to the 5 prism faces, in order:
          0: bottom triangle
          1: top triangle
          2: side face between vertices 0-1
          3: side face between vertices 1-2
          4: side face between vertices 2-0
    """
    h = float(height)
    r = float(radius)

    # Equilateral triangle in yz-plane
    angles = np.array(
        [np.pi / 2, np.pi / 2 + 2 * np.pi / 3, np.pi / 2 + 4 * np.pi / 3],
        dtype=float,
    )
    y = r * np.cos(angles)
    z = r * np.sin(angles)

    # 6 vertices: 3 at x=-h/2, 3 at x=+h/2
    base = np.column_stack([-0.5 * h * np.ones(3), y, z])  # 0,1,2
    top = np.column_stack([0.5 * h * np.ones(3), y, z])    # 3,4,5
    vertices = np.vstack([base, top])

    A, B, C = 0, 1, 2
    A_, B_, C_ = 3, 4, 5

    faces: List[List[int]] = []
    face_groups: List[Tuple[int, int]] = []

    def add_face(tris: List[List[int]]) -> None:
        start = len(faces)
        faces.extend(tris)
        end = len(faces)
        face_groups.append((start, end))

    # Bottom face
    add_face([[A, B, C]])

    # Top face (flip winding for outward normals)
    add_face([[A_, C_, B_]])

    # Side faces (quads triangulated into 2 triangles)
    add_face([[A, B, B_], [A, B_, A_]])  # AB
    add_face([[B, C, C_], [B, C_, B_]])  # BC
    add_face([[C, A, A_], [C, A_, C_]])  # CA

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=int), process=False)
    return mesh, face_groups


def make_star_pyramid(
    *,
    n_points: int = 5,
    radius_outer: float = 1.0,
    radius_inner: float = 0.5,
    height: float = 1.0,
) -> trimesh.Trimesh:
    """
    Create a star-based pyramid mesh:
      - base is a 2D star polygon in the yz-plane at x=0
      - apex at (height, 0, 0)

    Notes
    -----
    - Triangulates the base polygon using trimesh.creation.triangulate_polygon.
    - Then connects the apex to the boundary cycle (2*n_points edges).

    Returns
    -------
    mesh : trimesh.Trimesh
    """
    if n_points < 3 or (n_points % 2 != 1):
        raise ValueError("n_points must be odd and >= 3.")

    n_outer = int(n_points)
    m = 2 * n_outer

    angles = np.linspace(0.0, 2 * np.pi, num=m, endpoint=False)
    radii = np.empty(m, dtype=float)
    radii[::2] = float(radius_outer)
    radii[1::2] = float(radius_inner)

    # star boundary vertices in (y,z)
    y = radii * np.cos(angles)
    z = radii * np.sin(angles)
    boundary_yz = np.column_stack([y, z])  # (m,2)

    polygon = Polygon(boundary_yz)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    tri_vertices_2d, tri_faces_2d = triangulate_polygon(polygon)

    # Lift triangulated base into 3D at x=0  -> (x=0, y, z)
    base_vertices = np.column_stack(
        [
            np.zeros(len(tri_vertices_2d), dtype=float),
            tri_vertices_2d[:, 0],
            tri_vertices_2d[:, 1],
        ]
    )

    # Add apex
    apex = np.array([[float(height), 0.0, 0.0]], dtype=float)
    vertices = np.vstack([base_vertices, apex])
    apex_idx = vertices.shape[0] - 1

    base_faces = np.asarray(tri_faces_2d, dtype=int)

    # Match boundary vertices to indices in tri_vertices_2d (robust via NN + tol)
    tol = 1e-8
    boundary_indices: List[int] = []
    for yz in boundary_yz:
        diffs = np.linalg.norm(tri_vertices_2d - yz[None, :], axis=1)
        j = int(np.argmin(diffs))
        if float(diffs[j]) > tol:
            raise RuntimeError(
                "Could not match boundary vertex in triangulated vertex list "
                "(tolerance too small or triangulation changed)."
            )
        boundary_indices.append(j)
    boundary_indices = np.asarray(boundary_indices, dtype=int)

    # Side faces: connect apex to each boundary edge (i -> i+1)
    side_faces: List[List[int]] = []
    for i in range(m):
        j = (i + 1) % m
        vi = int(boundary_indices[i])
        vj = int(boundary_indices[j])
        # (apex, vj, vi) gives consistent winding if boundary is CCW in yz-plane
        side_faces.append([apex_idx, vj, vi])

    faces = np.vstack([base_faces, np.asarray(side_faces, dtype=int)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
