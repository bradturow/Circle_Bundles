"""
Synthetic datasets and geometric models used for demonstrations and validation.

Typical usage
-------------
>>> from circle_bundles.synthetic import sample_s2_trivial, make_tri_prism, mesh_to_density
"""

from __future__ import annotations

# Keep submodules importable as namespaces (optional but nice)
from . import (
    densities,
    mesh_vis,
    meshes,
    nat_img_patches,
    opt_flow_patches,
    s2_bundles,
    so3_sampling,
    step_edges,
    tori_and_kb,
)

# ----------------------------
# densities
# ----------------------------
from .densities import (
    mesh_to_density,
    get_density_axes,
    make_rotated_density_dataset,    
    get_mesh_sample,
)

# ----------------------------
# meshes + mesh viz
# ----------------------------
from .meshes import (
    make_tri_prism,
    make_star_pyramid,
    mesh_vertex_normals
)

from .mesh_vis import (
    make_density_visualizer,
    make_tri_prism_visualizer,
    make_star_pyramid_visualizer,
    make_rotating_mesh_clip,
)

# ----------------------------
# natural images / optical flow patches
# ----------------------------
from .nat_img_patches import (
    sample_nat_img_kb,
    get_gradient_dirs,
)

from .opt_flow_patches import (
    sample_opt_flow_torus,
    make_flow_patches,
)

# ----------------------------
# S^2 / SO(3) bundle-ish synthetic datasets
# ----------------------------
from .s2_bundles import (
    sample_sphere,
    hopf_projection,
    spin3_adjoint_to_so3,
    so3_to_s2_projection,
    sample_s2_trivial,
    tangent_frame_on_s2,
    sample_s2_unit_tangent,
)

from .so3_sampling import (
    sample_so3,
    project_o3,
)

# ----------------------------
# step-edge patches / tori & KB
# ----------------------------
from .step_edges import (
    get_patch_types_list,
    make_step_edges,
    make_all_step_edges,
    sample_binary_step_edges,
    mean_center,
    sample_step_edge_torus,
)

from .tori_and_kb import (
    const,
    small_to_big,
    sample_C2_torus,
    sample_foldy_klein_bottle,
    sample_R3_torus,
)

__all__ = [
    # namespaces
    "densities",
    "mesh_vis",
    "meshes",
    "nat_img_patches",
    "opt_flow_patches",
    "s2_bundles",
    "so3_sampling",
    "step_edges",
    "tori_and_kb",

    # densities
    "mesh_to_density",
    "get_density_axes",
#    "rotate_density",
    "make_rotated_density_dataset",    
    "get_mesh_sample",

    # meshes + viz
    "make_tri_prism",
    "make_star_pyramid",
    "mesh_vertex_normals",
    "make_density_visualizer",
    "make_tri_prism_visualizer",
    "make_star_pyramid_visualizer",
    "make_rotating_mesh_clip",

    # nat img / flow patches
    "sample_nat_img_kb",
    "get_gradient_dirs",
    "sample_opt_flow_torus",
    "make_flow_patches",

    # S^2 / SO(3)
    "sample_sphere",
    "hopf_projection",
    "spin3_adjoint_to_so3",
    "so3_to_s2_projection",
    "sample_s2_trivial",
    "tangent_frame_on_s2",
    "sample_s2_unit_tangent",
    "sample_so3",
    "project_o3",

    # step edges / torus-KB
    "get_patch_types_list",
    "make_step_edges",
    "make_all_step_edges",
    "sample_binary_step_edges",
    "mean_center",
    "sample_step_edge_torus",
    "const",
    "small_to_big",
    "sample_C2_torus",
    "sample_foldy_klein_bottle",
    "sample_R3_torus",
]
