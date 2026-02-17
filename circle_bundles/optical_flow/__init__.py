"""
Optical flow utilities: preprocessing, contrast/feature extraction, patch sampling,
frame I/O helpers, and visualization helpers.
"""

from __future__ import annotations

# Keep submodules importable as namespaces
from . import (
    contrast,
    flow_frames,
    flow_processing,
    patch_viz,
)

# ----------------------------
# contrast
# ----------------------------
from .contrast import (
    get_contrast_norms,
    get_dct_basis,
    get_predominant_dirs,
    get_lifted_predominant_dirs,
)

# ----------------------------
# flow_processing
# ----------------------------
from .flow_processing import (
    read_flo,
    sample_from_frame,
    get_patch_sample,
    preprocess_flow_patches,
)

# ----------------------------
# patch_viz
# ----------------------------
from .patch_viz import (
    make_patch_visualizer,
)

__all__ = [
    # namespaces
    "contrast",
    "flow_frames",
    "flow_processing",
    "patch_viz",

    # contrast
    "get_contrast_norms",
    "get_dct_basis",
    "get_predominant_dirs",
    "get_lifted_predominant_dirs",

    # flow_processing
    "read_flo",
    "sample_from_frame",
    "get_patch_sample",
    "preprocess_flow_patches",

    # patch_viz
    "make_patch_visualizer",
]
