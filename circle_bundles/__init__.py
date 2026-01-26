# circle_bundles/__init__.py
from __future__ import annotations

"""
circle_bundles: tools for detecting, visualizing, and classifying circle-bundle structure in data.

Recommended usage:
    import circle_bundles as cb

Public API:
- This module re-exports the *user-facing* symbols so typical users can rely on `cb.*`.
- Internal module paths may change; prefer `cb.<name>` over deep imports.
"""

# ------------------------------------------------------------
# Optional version string
# ------------------------------------------------------------
try:
    from ._version import __version__  # type: ignore
except Exception:  # pragma: no cover
    __version__ = "0+unknown"

# ============================================================
# Analysis
# ============================================================

from .analysis.fiberwise_clustering import (
    safe_add_edges,
    get_weights,
    fiberwise_clustering,
    plot_fiberwise_pca_grid,
    plot_fiberwise_summary_bars,
    get_cluster_persistence,
    get_filtered_cluster_graph,
)

from .analysis.local_analysis import (
    get_local_pca,
    plot_local_pca,
    get_local_rips,
    plot_local_rips,
)

# ============================================================
# Covers (base-space specific helpers) + Cover objects
# ============================================================

from .covers.metric_ball_cover_builders import (
    S2GeodesicMetric,
    RP2GeodesicMetric,
)

from .covers.triangle_cover_builders_fibonacci import (
    make_s2_fibonacci_star_cover,
    make_rp2_fibonacci_star_cover,
)

from .base_covers import (
    MetricBallCover,
    TriangulationStarCover,
)

# ============================================================
# Geometry / algebra helpers
# ============================================================

from .geometry.geometric_unwrapping import (
    get_cocycle_dict,
    lift_base_points,
)

from .geometry.z2_linear import (
    solve_Z2_linear_system,
    solve_Z2_edge_coboundary,
)

# ============================================================
# Optical flow utilities
# ============================================================

from .optical_flow.contrast import (
    get_contrast_norms,
    get_dct_basis,
    get_predominant_dirs,
    get_lifted_predominant_dirs,
)

from .optical_flow.flow_frames import (
    change_path,
    get_labeled_frame,
    write_video_from_frames,
    get_labeled_video,
    annotate_optical_flow,
    get_sintel_scene_folders,
)

from .optical_flow.flow_processing import (
    read_flo,
    sample_from_frame,
    get_patch_sample,
    preprocess_flow_patches,
)

from .optical_flow.patch_viz import (
    make_patch_visualizer,
)

# ============================================================
# Synthetic data
# ============================================================

from .synthetic.densities import (
    mesh_to_density,
    get_density_axes,
    rotate_density,
    get_mesh_sample,
)

from .synthetic.mesh_vis import (
    make_density_visualizer,
    make_tri_prism_visualizer,
    make_star_pyramid_visualizer,
    make_rotating_mesh_clip,
)

from .synthetic.meshes import (
    make_tri_prism,
    make_star_pyramid,
)

from .synthetic.nat_img_patches import (
    sample_nat_img_kb,
    get_gradient_dirs,
)

from .synthetic.opt_flow_patches import (
    sample_opt_flow_torus,
    make_flow_patches,
)

from .synthetic.s2_bundles import (
    sample_sphere,
    hopf_projection,
    spin3_adjoint_to_so3,
    so3_to_s2_projection,
    sample_s2_trivial,
    tangent_frame_on_s2,
    sample_s2_unit_tangent,
)

from .synthetic.so3_sampling import (
    sample_so3,
    project_o3,
)

from .synthetic.step_edges import (
    get_patch_types_list,
    make_step_edges,
    make_all_step_edges,
    sample_binary_step_edges,
    mean_center,
    sample_step_edge_torus,
)

from .synthetic.tori_and_kb import (
    const,
    small_to_big,
    sample_C2_torus,
)

# ============================================================
# Trivializations
# ============================================================

from .trivializations.local_triv import (
    compute_circular_coords_pca2,
    compute_circular_coords_dreimac,
)

# ============================================================
# Core bundle
# ============================================================

from .bundle import (
    attach_bundle_viz_methods,
    build_bundle,
)

# ============================================================
# Metrics
# ============================================================

from .metrics import (
    EuclideanMetric,
    S1AngleMetric,
    RP1AngleMetric,
    S1UnitVectorMetric,
    RP1UnitVectorMetric,
    RP2UnitVectorMetric,
    T2FlatMetric,
    Torus_Z2QuotientMetric_R4,
    RP2_TrivialMetric,
    RP2_TwistMetric,
    RP2_FlipMetric,
    S3QuotientMetric,
)

# ============================================================
# Visualization
# ============================================================

from .viz.angles import (
    compare_angle_pairs,
    compare_trivs,
)

from .viz.base_vis import (
    base_vis,
)

from .viz.bundle_dash import (
    show_bundle_vis,
    save_bundle_snapshot,
)

from .viz.circle_vis import (
    circle_vis,
    circle_vis_grid,
)

from .viz.fiber_vis import (
    fiber_vis,
)

from .viz.fiberwise_clustering_vis import (
    make_patch_cluster_diagram,
    get_G_vertex_coords,
    plot_component_patch_diagram,
)

from .viz.gudhi_graph_utils import (
    graph_to_st,
    create_st_dicts,
)

from .viz.lattice_vis import (
    lattice_vis,
)

from .viz.nerve_circle import (
    show_circle_nerve,
)

from .viz.nerve_vis import (
    nerve_vis,
)

from .viz.pca_vis import (
    show_pca,
)

from .viz.thumb_grids import (
    show_data_vis,
)

# ============================================================
# Public export list (single source of truth)
# ============================================================

__all__ = [
    "__version__",

    # analysis.fiberwise_clustering
    "safe_add_edges",
    "get_weights",
    "fiberwise_clustering",
    "plot_fiberwise_pca_grid",
    "plot_fiberwise_summary_bars",
    "get_cluster_persistence",
    "get_filtered_cluster_graph",

    # analysis.local_analysis
    "get_local_pca",
    "plot_local_pca",
    "get_local_rips",
    "plot_local_rips",

    # covers + base_covers
    "S2GeodesicMetric",
    "RP2GeodesicMetric",
    "make_s2_fibonacci_star_cover",
    "make_rp2_fibonacci_star_cover",
    "MetricBallCover",
    "TriangulationStarCover",

    # geometry
    "get_cocycle_dict",
    "lift_base_points",
    "solve_Z2_linear_system",
    "solve_Z2_edge_coboundary",

    # optical_flow
    "get_contrast_norms",
    "get_dct_basis",
    "get_predominant_dirs",
    "get_lifted_predominant_dirs",
    "change_path",
    "get_labeled_frame",
    "write_video_from_frames",
    "get_labeled_video",
    "annotate_optical_flow",
    "get_sintel_scene_folders",
    "read_flo",
    "sample_from_frame",
    "get_patch_sample",
    "preprocess_flow_patches",
    "make_patch_visualizer",

    # synthetic
    "mesh_to_density",
    "get_density_axes",
    "rotate_density",
    "get_mesh_sample",
    "make_density_visualizer",
    "make_tri_prism_visualizer",
    "make_star_pyramid_visualizer",
    "make_rotating_mesh_clip",
    "make_tri_prism",
    "make_star_pyramid",
    "sample_nat_img_kb",
    "get_gradient_dirs",
    "sample_opt_flow_torus",
    "make_flow_patches",
    "sample_sphere",
    "hopf_projection",
    "spin3_adjoint_to_so3",
    "so3_to_s2_projection",
    "sample_s2_trivial",
    "tangent_frame_on_s2",
    "sample_s2_unit_tangent",
    "sample_so3",
    "project_o3",
    "get_patch_types_list",
    "make_step_edges",
    "make_all_step_edges",
    "sample_binary_step_edges",
    "mean_center",
    "sample_step_edge_torus",
    "const",
    "small_to_big",
    "sample_C2_torus",

    # trivializations
    "compute_circular_coords_pca2",
    "compute_circular_coords_dreimac",

    # core bundle
    "attach_bundle_viz_methods",
    "build_bundle",

    # metrics
    "EuclideanMetric",
    "S1AngleMetric",
    "RP1AngleMetric",
    "S1UnitVectorMetric",
    "RP1UnitVectorMetric",
    "RP2UnitVectorMetric",
    "T2FlatMetric",
    "Torus_Z2QuotientMetric_R4",
    "RP2_TrivialMetric",
    "RP2_TwistMetric",
    "RP2_FlipMetric",
    "S3QuotientMetric",

    # viz
    "compare_angle_pairs",
    "compare_trivs",
    "base_vis",
    "show_bundle_vis",
    "save_bundle_snapshot",
    "circle_vis",
    "circle_vis_grid",
    "fiber_vis",
    "make_patch_cluster_diagram",
    "get_G_vertex_coords",
    "plot_component_patch_diagram",
    "graph_to_st",
    "create_st_dicts",
    "lattice_vis",
    "show_circle_nerve",
    "nerve_vis",
    "show_pca",
    "show_data_vis",
]
