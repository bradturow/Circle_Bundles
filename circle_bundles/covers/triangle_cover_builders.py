# circle_bundles/triangle_cover_builders_gudhi.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from ..base_covers import TriangulationStarCover  
from .triangle_covers import (              
    initialize_octahedron,
    get_sd,
    build_rp2_simplex_tree,
    sphere_to_octahedron,
    project_to_veronese,
    veronese_map,
)

__all__ = ["make_rp2_cover", "make_s2_cover"]


def make_s2_cover(base_points: np.ndarray, *, n_sd: int = 2, tol: float = 1e-8) -> TriangulationStarCover:
    """
    S^2 cover by vertex stars of an octahedron triangulation + barycentric subdivision,
    using your gudhi-based pipeline. Returns a CoverBase-style TriangulationStarCover
    (so it supports summarize(), show_nerve(), etc.).
    """
    base_points = np.asarray(base_points, dtype=float)
    if base_points.ndim != 2 or base_points.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {base_points.shape}.")

    # Triangulate S^2
    K, vc = initialize_octahedron()
    K, vc = get_sd(K, int(n_sd), vertex_coords=vc)

    # Preimages live on the octahedron surface in R^3
    K_preimages = sphere_to_octahedron(base_points)

    # Identity "embedding" for S^2 case: vertices already in R^3
    vertex_coords_dict = {int(k): np.asarray(v, float) for k, v in vc.items()}

    cover = TriangulationStarCover(
        base_points=base_points,
        K_preimages=K_preimages,
        K=K,
        vertex_coords_dict=vertex_coords_dict,
    )
    cover.build()
    return cover


def make_rp2_cover(base_points: np.ndarray, *, n_sd: int = 2, tol: float = 1e-8) -> TriangulationStarCover:
    """
    RP^2 cover (Veronese in R^6) rebuilt from your old notebook pipeline, but returns
    the new CoverBase-style TriangulationStarCover with summarize()/show_nerve().
    """
    base_points = np.asarray(base_points, dtype=float)
    if base_points.ndim != 2 or base_points.shape[1] != 3:
        raise ValueError(f"base_points must be (n,3). Got {base_points.shape}.")

    # 1) S^2 triangulation (octahedron + barycentric subdivision)
    K_s2, vc_s2 = initialize_octahedron()
    K_s2, vc_s2 = get_sd(K_s2, int(n_sd), vertex_coords=vc_s2)

    # 2) RP^2 triangulation as antipodal quotient of K_s2
    # build_rp2_simplex_tree returns (K_rp2, vertex_coords_rp2_r3, old_to_new)
    K_rp2, vc_rp2_r3, _old_to_new = build_rp2_simplex_tree(K_s2, vc_s2)

    # 3) Veronese-map RP^2 vertices into R^6
    vc_rp2_r6: Dict[int, np.ndarray] = {
        int(v): veronese_map(np.asarray(p3, float))
        for v, p3 in vc_rp2_r3.items()
    }

    # 4) Map each base point to a point in the *octahedron surface* (in R^3),
    #    then project into Veronese R^6 using barycentric interpolation on K_s2
    K_preimages = sphere_to_octahedron(base_points)  # (n,3) points on octahedron surface
    K_preimages_rp2_r6 = project_to_veronese(K_preimages, vc_s2, K_s2, tol=tol)  # (n,6)

    # 5) The cover’s “base points” live in R^6 (RP^2 embedded); that’s what downstream expects.
    base_points_rp2_r6 = veronese_map(base_points)

    cover = TriangulationStarCover(
        base_points=base_points_rp2_r6,
        K_preimages=K_preimages_rp2_r6,
        K=K_rp2,
        vertex_coords_dict=vc_rp2_r6,
    )
    
    cover.vc_rp2_r3 = vc_rp2_r3  # store the R^3 reps for viz 

    # Build flat vertex coordinates in a stable vertex-id order
    vids = sorted(vc_rp2_r3.keys())
    flat_vertex_coords = np.array([vc_rp2_r3[v][:2] for v in vids], dtype=float)

    # Apply a rotation for visualization purposes
    theta = np.pi / 4
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ], dtype=float)

    cover.flat_vertex_coords = flat_vertex_coords @ R.T

    cover.build()
    return cover
