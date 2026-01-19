# optical_flow/api.py
from __future__ import annotations

"""
Public API for optical_flow.

Import style:
    from optical_flow.api import (
        read_flo,
        get_patch_sample,
        preprocess_flow_patches,
        get_contrast_norms,
        get_predominant_dirs,
        make_patch_visualizer,
        annotate_optical_flow,
    )

Notes
-----
- This file is intentionally curated (not a dump of every internal helper).
- Visualization utilities may require optional dependencies (Pillow, cv2, imageio, matplotlib).
  Those are imported lazily inside the viz functions to keep base imports lightweight.
"""

# ----------------------------
# Contrast + feature extraction
# ----------------------------
from .contrast import (
    get_contrast_norms,
    get_dct_basis,
    get_predominant_dirs,
    get_lifted_predom_dirs,
)

# ----------------------------
# IO + patch sampling + preprocessing
# ----------------------------
from .flow_processing import (
    read_flo,
    sample_from_frame,
    get_patch_sample,
    preprocess_flow_patches,
)

# ----------------------------
# Patch visualization (matplotlib)
# ----------------------------
from .patch_viz import (
    PatchKind,
    make_patch_visualizer,
)

# ----------------------------
# Frame / video visualization (optional deps, lazily imported inside)
# ----------------------------
from .flow_frames import (
    change_path,
    get_labeled_frame,
    write_video_from_frames,
    get_labeled_video,
    annotate_optical_flow,
    get_sintel_scene_folders,
)

__all__ = [
    # contrast + features
    "get_contrast_norms",
    "get_dct_basis",
    "get_predominant_dirs",
    "get_lifted_predom_dirs",
    # IO + sampling + preprocessing
    "read_flo",
    "sample_from_frame",
    "get_patch_sample",
    "preprocess_flow_patches",
    # patch viz
    "PatchKind",
    "make_patch_visualizer",
    # frame/video viz helpers
    "change_path",
    "get_labeled_frame",
    "write_video_from_frames",
    "get_labeled_video",
    "annotate_optical_flow",
    "get_sintel_scene_folders",
]
