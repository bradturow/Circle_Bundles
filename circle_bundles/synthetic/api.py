# synthetic/api.py
from __future__ import annotations

"""
Public API re-exports for synthetic.

Import style:
    from synthetic.api import sample_S2_trivial, sample_SO3, sample_C2_torus, ...

Notes
-----
- This file is intentionally curated (not a dump of every internal helper).
- Prefer exporting stable, user-facing constructors + samplers + visualizers.
"""

# ----------------------------
# Densities
# ----------------------------
from .densities import (
    mesh_to_density,
    get_density_axes,
    rotate_density,
    get_mesh_sample,
)

# ----------------------------
# Mesh generators + visualization
# ----------------------------
from .meshes import (
    make_tri_prism,
    make_star_pyramid,
)

from .mesh_vis import (
    make_density_visualizer,
    expand_face_groups,
    make_tri_prism_visualizer,
    make_star_pyramid_visualizer,
    fig_to_rgb_array,
    make_rotating_mesh_clip,
)

# ----------------------------
# S^2 / Hopf / tangent-bundle style samplers
# ----------------------------
from .s2_bundles import (
    sample_sphere,
    hopf_projection,
    spin3_adjoint_to_so3,
    so3_to_s2_projection,
    sample_S2_trivial,
    tangent_frame_on_s2,
    sample_S2_unit_tangent,
)

# ----------------------------
# SO(3) / O(3) sampling utilities
# ----------------------------
from .so3_sampling import (
    sample_SO3,
    project_O3,
)

# ----------------------------
# Tori / Klein bottle embeddings (data + angle helpers only)
# ----------------------------
from .tori_and_kb import (
    AngleFunc,
    const,
    small_to_big,
    wrap_angle,
    sample_C2_torus,
    torus_base_projection_from_data,
    kb_pairwise_distances_from_data,
)

# ----------------------------
# "Natural image" style KB patches + gradient axis
# ----------------------------
from .nat_img_patches import (
    sample_nat_img_kb,
    get_gradient_dirs,
)

# ----------------------------
# Optical-flow synthetic patch models
# ----------------------------
from .opt_flow_patches import (
    sample_opt_flow_torus,
    make_flow_patches,
)

# ----------------------------
# Step-edge models
# ----------------------------
from .step_edges import (
    get_patch_types_list,
    make_step_edges,
    make_all_step_edges,
    sample_binary_step_edges,
    mean_center,
    sample_step_edge_torus,
)

# ----------------------------
# __all__ (explicit, curated)
# ----------------------------
__all__ = [
    # densities
    "mesh_to_density",
    "get_density_axes",
    "rotate_density",
    "get_mesh_sample",

    # meshes + viz
    "make_tri_prism",
    "make_star_pyramid",
    "make_density_visualizer",
    "expand_face_groups",
    "make_tri_prism_visualizer",
    "make_star_pyramid_visualizer",
    "fig_to_rgb_array",
    "make_rotating_mesh_clip",

    # s2 bundles / hopf / tangent
    "sample_sphere",
    "hopf_projection",
    "spin3_adjoint_to_so3",
    "so3_to_s2_projection",
    "sample_S2_trivial",
    "tangent_frame_on_s2",
    "sample_S2_unit_tangent",

    # so3 sampling
    "sample_SO3",
    "project_O3",

    # tori + klein bottle (data-level)
    "AngleFunc",
    "const",
    "small_to_big",
    "wrap_angle",
    "sample_C2_torus",
    "torus_base_projection_from_data",
    "kb_pairwise_distances_from_data",

    # nat img kb
    "sample_nat_img_kb",
    "get_gradient_dirs",

    # opt flow patches
    "sample_opt_flow_torus",
    "make_flow_patches",

    # step edges
    "get_patch_types_list",
    "make_step_edges",
    "make_all_step_edges",
    "sample_binary_step_edges",
    "mean_center",
    "sample_step_edge_torus",
]
