from __future__ import annotations

"""
Public API re-exports for circle_bundles.

Import style:
    from circle_bundles.api import build_bundle, show_bundle, compute_classes, compute_bundle_persistence, ...

Notes
-----
- This file is intentionally curated (not a dump of every internal helper).
- We avoid re-exporting implementation artifacts (dataclass, PCA, defaultdict, etc.).
- Back-compat: compute_persistence is provided as an alias of compute_bundle_persistence.
"""

# ----------------------------
# Core bundle construction / results
# ----------------------------
from .bundle import (
    BundleResult,
    BundleMapResult,
    GlobalTrivializationResult,
    MaxTrivialSubcomplex,
    build_bundle,
    show_bundle,
    attach_bundle_viz_methods,
    # bundle_app,                  # optional (Dash app wrapper)
    bundle_compare_trivs,
    bundle_show_nerve,
    bundle_show_circle_nerve,
    bundle_show_max_trivial,
)

# ----------------------------
# Bundle map pipeline
# ----------------------------
from .trivializations.bundle_map import (
    BundleMapReport,
    ChartDisagreementStats,
    FrameDataset,
    FrameReducerConfig,
    FrameReductionReport,
    TrueFramesResult,
    angles_to_unit,
    apply_frame_reduction,
    build_bundle_map,
    build_true_frames,
    chart_disagreement_stats,
    cocycle_from_frames,
    cocycle_projection_distance,
    get_bundle_map,
    get_classifying_map,
    get_frame_dataset,
    get_local_frames,
    infer_edges_from_U,
    polar_stiefel_projection,
    project_frames_to_stiefel,
    project_to_rank2_projection,
    reduce_frames_psc,
    reduce_frames_subspace_pca,
    show_bundle_map_summary,
    stack_active_frames,
    weighted_angle_mean_anchored,
    witness_error_stats,
)

from .reduction.frame_reduction import (
    reduction_curve_psc,
    reduction_curve_subspace_pca,
)

from .gauge_canon import GaugeCanonConfig


# ----------------------------
# Characteristic classes
# ----------------------------
from .characteristic_class import (
    ClassResult,
    H2_dimensions,
    compute_classes,
    compute_twisted_euler_class,
    show_summary,
    # Advanced / linear algebra helpers
    build_boundary_mod2_C2_to_C1,
    build_boundary_mod2_C3_to_C2,
    build_delta_C0_to_C1_Z2,
    build_delta_C1_to_C2_Z_twisted,
    build_twisted_boundary_C2_to_C1,
    build_twisted_boundary_C3_to_C2,
    canonicalize_o1_cochain,
    euler_pairing_Z,
    euler_pairing_Z2,
    fundamental_class_Z_rank1,
    fundamental_class_Z2_rank1,
    in_colspace_over_Q,
    in_image_Z_fast_pipeline,
    in_image_mod2,
    normalize_cycle_Z,
    nullspace_basis_mod2,
    nullspace_over_Q,
    principal_lift_RZ,
    rank_over_Q,
    rref_mod2,
    twisted_delta_euler_on_tet,
    twisted_delta_theta_real,
)

# ----------------------------
# Characteristic class persistence
# ----------------------------
from .analysis.class_persistence import (
    CobirthResult,
    CodeathResult,
    PersistenceResult,
    compute_bundle_persistence,
    summarize_edge_driven_persistence,
    build_edge_weights_from_transition_report,
)

# Back-compat alias
compute_persistence = compute_bundle_persistence

# ----------------------------
# Global coordinatization / max-trivial subcomplex
# ----------------------------
from .coordinatization import (
    build_global_trivialization,
    compute_max_trivial_subcomplex,
    # useful helpers
    frechet_mean_circle,
    global_from_mu,
    mu_from_partition_unity_radians,
    mu_vertices_from_singer_radians,
    mu_vertices_from_spanning_tree_radians,
    reflect_angles_about_axis,
    theta_dict_to_edge_vector_radians,
    theta_dir_from_canonical,
    wrap_angle_rad,
    wrap_to_pi,
)

# ----------------------------
# Covers / nerve / geometry
# ----------------------------
from .covers import (
    CoverBase,
    MetricBallCover,
    TriangulationStarCover,
    NerveSummary,
    plot_cover_summary_boxplot,
)


from .metric_ball_cover_builders import make_s2_metric_ball_cover, make_rp2_metric_ball_cover


from .triangle_cover_builders import (
    make_s2_cover,
    make_rp2_cover,
    veronese_map,
    project_to_veronese,
    sphere_to_octahedron,
    initialize_octahedron,
    get_sd,
    build_rp2_simplex_tree,
)

from .triangle_covers import (
    get_star_cover,
    get_U_from_subcomplexes,
    get_simplices,
    points_to_simplices,
    bary_extend,
    barycentric_coords_triangle,
    barycentric_refinement,
    insert_triangle_complex,
)

from .nerve.nerve_utils import (
    get_simplices as get_nerve_simplices,
    max_trivial_to_simplex_tree,
)

from .geometry.geometry import (
    get_bary_coords,
    points_in_triangle_mask,
)


# ----------------------------
# Cover builders (S^2 / RP^2 etc.)
# ----------------------------
from .triangle_cover_builders_fibonacci import (
    make_s2_fibonacci_star_cover,
    make_rp2_fibonacci_star_cover,
    fibonacci_sphere,
)


# ----------------------------
# O(2) cocycles / quality
# ----------------------------
from .o2_cocycle import (
    O2Cocycle,
    TransitionReport,
    complete_edge_orientations,
    decompose_O2_as_R_times_r,
    det_sign,
    estimate_transitions,
    project_to_O2,
    r_from_det,
    reflection_axis_matrix,
    rotation_matrix,
)

from .analysis.quality import (
    BundleQualityReport,
    compute_bundle_quality,
    compute_O2_cocycle_defect,
    compute_delta_from_triples,
    compute_eps_alignment_stats,
    h_dist_S1,
)

# IMPORTANT:
# alpha is now computed from *Euclidean/chordal* eps (per your update).
# We keep the old public name `compute_alpha_from_eps_delta` but point it at
# the Euclidean version so existing notebooks don’t break.
from .analysis.quality import compute_alpha_from_eps_delta_euc as compute_alpha_from_eps_delta

# ----------------------------
# Metrics
# ----------------------------
from .metrics import (
    Metric,
    EuclideanMetric,
    SciPyCdistMetric,
    get_dist_mat,
    as_metric,
    # S^1 / T^2 / KB / RP^1 / RP^2 metrics
    S1AngleMetric,
    S1UnitVectorMetric,
    S1_dist,
    S1_dist2,
    T2FlatMetric,
    T2_dist,
    KleinBottleFlatMetric,
    KB_flat_dist,
    # diagonal torus Z2 quotient in angle coords + factory
    TorusDiagFlatMetric,
    T2_Z2QuotientFlatMetric,
    T2_diag_flat_dist,
    RP1AngleMetric,
    RP1UnitVectorMetric,
    RP1_dist,
    RP1_dist2,
    RP2UnitVectorMetric,
    RP2_FlipMetric,
    RP2_TrivialMetric,
    RP2_TwistMetric,
    RP2_dist,
    # Euclidean Z2 quotient + torus->KB/diag quotient in R4
    Z2QuotientMetricEuclidean,
    Torus_KleinQuotientMetric_R4,
    Torus_DiagQuotientMetric_R4,
    Torus_Z2QuotientMetric_R4,
    act_klein_C2_torus,
    act_diag_C2_torus,
    # S^3 quotient metrics
    S3QuotientMetric,
    ZpHopfQuotientMetricS3,
    Z2LensAntipodalQuotientMetricS3,
    Z2QuotientMetricR5,
    act_base_only,
    act_pi_twist,
    act_reflection_twist,
)

from .t2_bundle_metrics import (
    T2_circle_bundle_metric_oriented,
    T2_circle_bundle_metric_nonorientable,
    T2xS1ProductAngleMetric,
)


# ----------------------------
# Local analysis / triv
# ----------------------------
from .trivializations.local_triv import (
    LocalTrivResult,
    compute_local_triv,
    compute_circular_coords_dreimac,
    compute_circular_coords_pca2,
)

from .analysis.local_analysis import (
    get_dense_fiber_indices,
    get_local_pca,
    get_local_rips,
    plot_local_pca,
    plot_local_rips,
)

from .geometry.geometric_unwrapping import (
    get_cocycle_dict,
    lift_base_points,
)

# ----------------------------
# Fiberwise clustering
# ----------------------------
from .fiberwise_clustering import (
    fiberwise_clustering,
    get_cluster_persistence,
    get_filtered_cluster_graph,
    get_weights,
    plot_fiberwise_pca_grid,
    plot_fiberwise_summary_bars,
    safe_add_edges,
)

# ----------------------------
# Z2 linear utilities
# ----------------------------
from .geometry.z2_linear import (
    solve_Z2_edge_coboundary,
    solve_Z2_linear_system,
    phi_Z2_to_pm1,
)

# ----------------------------
# Viz re-exports (optional but convenient)
# ----------------------------
from .viz.pca_vis import show_pca
from .viz.thumb_grids import show_data_vis
from .viz.nerve_vis import nerve_vis
from .viz.nerve_plotly import make_nerve_figure, nerve_with_slider
from .viz.circle_vis import circle_vis, circle_vis_grid
from .viz.fiber_vis import fiber_vis
from .viz.base_vis import base_vis

from .viz.lattice_vis import lattice_vis
from .viz.bundle_dash import show_bundle_vis, save_bundle_snapshot

from .viz.angles import (
    align_angles_to,
    compare_angle_pairs,
    compare_trivs,
    fit_o2_on_circle,
    set_pi_ticks,
)

from .viz.gudhi_graph_utils import create_st_dicts
from .viz.nerve_circle import show_circle_nerve

from .viz.fiberwise_clustering_vis import (
    make_patch_cluster_diagram,
    get_G_vertex_coords,
    GraphComponentData,
    extract_component_subgraph,
    representative_indices_for_clusters,
    component_patch_reps,
    plot_component_patch_diagram,
)

# ----------------------------
# __all__ (explicit, curated)
# ----------------------------
__all__ = [
    # bundle
    "BundleResult", "BundleMapResult", "GlobalTrivializationResult", "MaxTrivialSubcomplex",
    "build_bundle", "show_bundle", "attach_bundle_viz_methods",
    "bundle_compare_trivs", "bundle_show_nerve", "bundle_show_circle_nerve", "bundle_show_max_trivial",

    # bundle_map
    "BundleMapReport", "ChartDisagreementStats", "FrameDataset", "FrameReducerConfig",
    "FrameReductionReport", "TrueFramesResult",
    "angles_to_unit", "apply_frame_reduction", "build_bundle_map", "build_true_frames",
    "chart_disagreement_stats", "cocycle_from_frames", "cocycle_projection_distance",
    "get_bundle_map", "get_classifying_map", "get_frame_dataset", "get_local_frames",
    "infer_edges_from_U", "polar_stiefel_projection", "project_frames_to_stiefel",
    "project_to_rank2_projection", "reduce_frames_psc", "reduce_frames_subspace_pca",
    "reduction_curve_psc", "reduction_curve_subspace_pca",
    "show_bundle_map_summary", "stack_active_frames", "weighted_angle_mean_anchored",
    "witness_error_stats", "GaugeCanonConfig",

    # characteristic classes
    "ClassResult", "H2_dimensions", "compute_classes", "compute_twisted_euler_class", "show_summary",
    "build_boundary_mod2_C2_to_C1", "build_boundary_mod2_C3_to_C2",
    "build_delta_C0_to_C1_Z2", "build_delta_C1_to_C2_Z_twisted",
    "build_twisted_boundary_C2_to_C1", "build_twisted_boundary_C3_to_C2",
    "canonicalize_o1_cochain", "euler_pairing_Z", "euler_pairing_Z2",
    "fundamental_class_Z_rank1", "fundamental_class_Z2_rank1",
    "in_colspace_over_Q", "in_image_Z_fast_pipeline", "in_image_mod2",
    "normalize_cycle_Z", "nullspace_basis_mod2", "nullspace_over_Q",
    "principal_lift_RZ", "rank_over_Q", "rref_mod2",
    "twisted_delta_euler_on_tet", "twisted_delta_theta_real",

    # persistence
    "CobirthResult", "CodeathResult", "PersistenceResult",
    "compute_bundle_persistence", "summarize_edge_driven_persistence",
    "build_edge_weights_from_transition_report",
    "compute_persistence",

    # coordinatization
    "build_global_trivialization", "compute_max_trivial_subcomplex",
    "frechet_mean_circle", "global_from_mu",
    "mu_from_partition_unity_radians", "mu_vertices_from_singer_radians",
    "mu_vertices_from_spanning_tree_radians",
    "reflect_angles_about_axis", "theta_dict_to_edge_vector_radians",
    "theta_dir_from_canonical", "wrap_angle_rad", "wrap_to_pi",

    # covers / geometry / nerve
    "CoverBase", "MetricBallCover", "TriangulationStarCover", "NerveSummary",
    "plot_cover_summary_boxplot",
    "make_s2_cover", "make_rp2_cover", "veronese_map", "project_to_veronese",
    "sphere_to_octahedron", "initialize_octahedron", "get_sd", "build_rp2_simplex_tree",
    "get_star_cover", "get_U_from_subcomplexes", "get_simplices", "points_to_simplices",
    "bary_extend", "barycentric_coords_triangle", "barycentric_refinement", "insert_triangle_complex",
    "get_nerve_simplices", "max_trivial_to_simplex_tree",
    "get_bary_coords", "points_in_triangle_mask", "make_s2_metric_ball_cover", "make_rp2_metric_ball_cover",
    "make_s2_fibonacci_star_cover", "fibonacci_sphere", "make_rp2_fibonacci_star_cover",

    # o2 cocycle / quality
    "O2Cocycle", "TransitionReport",
    "complete_edge_orientations", "decompose_O2_as_R_times_r", "det_sign",
    "estimate_transitions", "project_to_O2", "r_from_det",
    "reflection_axis_matrix", "rotation_matrix",
    "BundleQualityReport", "compute_bundle_quality", "compute_O2_cocycle_defect",
    "compute_alpha_from_eps_delta", "compute_delta_from_triples", "compute_eps_alignment_stats",
    "h_dist_S1",

    # metrics
    "Metric", "EuclideanMetric", "SciPyCdistMetric", "get_dist_mat", "as_metric",
    "S1AngleMetric", "S1UnitVectorMetric", "S1_dist", "S1_dist2",
    "T2FlatMetric", "T2_dist",
    "KleinBottleFlatMetric", "KB_flat_dist",
    "TorusDiagFlatMetric", "T2_Z2QuotientFlatMetric", "T2_diag_flat_dist",
    "RP1AngleMetric", "RP1UnitVectorMetric", "RP1_dist", "RP1_dist2",
    "RP2UnitVectorMetric", "RP2_FlipMetric", "RP2_TrivialMetric", "RP2_TwistMetric", "RP2_dist",
    "Z2QuotientMetricEuclidean",
    "Torus_KleinQuotientMetric_R4", "Torus_DiagQuotientMetric_R4", "Torus_Z2QuotientMetric_R4",
    "act_klein_C2_torus", "act_diag_C2_torus",
    "S3QuotientMetric", "ZpHopfQuotientMetricS3", "Z2LensAntipodalQuotientMetricS3",
    "Z2QuotientMetricR5",
    "act_base_only", "act_pi_twist", "act_reflection_twist", "T2_circle_bundle_metric_oriented",
    "T2_circle_bundle_metric_nonorientable", "T2xS1ProductAngleMetric",

    # local
    "LocalTrivResult", "compute_local_triv", "compute_circular_coords_dreimac", "compute_circular_coords_pca2",
    "get_dense_fiber_indices", "get_local_pca", "get_local_rips", "plot_local_pca", "plot_local_rips",
    "get_cocycle_dict", "lift_base_points",

    # clustering
    "fiberwise_clustering", "get_cluster_persistence", "get_filtered_cluster_graph",
    "get_weights", "plot_fiberwise_pca_grid", "plot_fiberwise_summary_bars", "safe_add_edges",

    # fiberwise clustering viz helpers
    "make_patch_cluster_diagram",
    "get_G_vertex_coords",
    "GraphComponentData",
    "extract_component_subgraph",
    "representative_indices_for_clusters",
    "component_patch_reps",
    "plot_component_patch_diagram",
    "create_st_dicts",

    # z2 linear
    "solve_Z2_edge_coboundary", "solve_Z2_linear_system", "phi_Z2_to_pm1",

    # viz
    "show_pca", "show_data_vis", "nerve_vis", "make_nerve_figure", "nerve_with_slider",
    "circle_vis", "circle_vis_grid", "fiber_vis", "lattice_vis",
    "show_bundle_vis", "save_bundle_snapshot",
    "align_angles_to", "compare_angle_pairs", "compare_trivs", "fit_o2_on_circle", "set_pi_ticks",
    "show_circle_nerve", "base_vis",
]
