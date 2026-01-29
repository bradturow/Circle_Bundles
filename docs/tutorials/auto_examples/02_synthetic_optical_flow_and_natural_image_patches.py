"""
Application demo: optical flow and natural image patches
========================================================

This tutorial demonstrates applications of ``circle_bundles`` to synthetic optical flow data
and synthetic natural image patches.

We consider two synthetic datasets sampled from known models for spaces of high-contrast
:math:`n\\times n` patches:

- Optical flow patch model: a torus embedded in :math:`\\mathbb{R}^{2n^2}`.
- Natural image patch model: a Klein bottle embedded in :math:`\\mathbb{R}^{n^2}`.

For each dataset, we compute a feature map
:math:`\\pi(x) \\in \\mathbb{RP}^1 \\cong \\mathbb{S}^1` (predominant direction),
build a metric-ball cover of the base, and run the local-to-global bundle pipeline.

Notes
-----
- This is an application-oriented demo. For a minimal tutorial of the core pipeline,
  see the "Hopf and SO(3)" tutorial.
- Some steps (Ripser) can be expensive; set ``RUN_HEAVY = True`` if you want full runs.
"""

# %%
# Imports
# -------
import numpy as np
import matplotlib.pyplot as plt

import circle_bundles as cb
from circle_bundles import RP1AngleMetric as rp1_metric
import circle_bundles.synthetic as sy
import circle_bundles.viz as vz
import circle_bundles.optical_flow as of

# Persistent homology (optional / potentially heavy)
from ripser import ripser
from persim import plot_diagrams

# Optional: local circular coordinates via Dreimac
from dreimac import CircularCoords

RUN_HEAVY = False  # set True to run Ripser with larger n_perm, etc.

# %%
# Torus model: high-contrast optical flow patches
# -----------------------------------------------
n_flow_patches = 5000
n_flow = 3  # patch size

rng = np.random.default_rng(0)
flow_data = sy.sample_opt_flow_torus(n_flow_patches, dim=n_flow, rng=rng)[0]
print(f"{n_flow_patches} {n_flow}-by-{n_flow} optical flow patches generated.")

patch_vis = of.make_patch_visualizer()

# %%
# Predominant flow direction :math:`\\pi(x) \\in \\mathbb{RP}^1`
predom_dirs = of.get_predominant_dirs(flow_data)[0]
print("Predominant directions computed.")

# %%
# Visualize a random sample, arranged by direction
n_samples = 30
label_func = [fr"$\theta = {np.round(t/np.pi, 2)}$" + r"$\pi$" for t in predom_dirs]

fig = vz.show_data_vis(
    flow_data,
    patch_vis,
    label_func=label_func,
    angles=predom_dirs,
    sampling_method="angle",
    max_samples=n_samples,
)
plt.show()

# %%
# Persistent homology evidence (optional)
if RUN_HEAVY:
    dgms_2 = ripser(flow_data, coeff=2, maxdim=2, n_perm=500)["dgms"]
    dgms_3 = ripser(flow_data, coeff=3, maxdim=2, n_perm=500)["dgms"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    plot_diagrams(dgms_2, ax=axes[0], title="coeff = 2")
    plot_diagrams(dgms_3, ax=axes[1], title="coeff = 3")
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Ripser demo (set RUN_HEAVY = True to run).")

# %%
# Cover the base space RP^1 by metric balls
n_flow_landmarks = 12
flow_landmarks = np.linspace(0, np.pi, n_flow_landmarks, endpoint=False)
flow_overlap = 1.4
flow_radius = flow_overlap * np.pi / (2 * n_flow_landmarks)

flow_cover = cb.MetricBallCover(predom_dirs, flow_landmarks, flow_radius, metric=rp1_metric())
flow_cover.build()

flow_cover.summarize(plot=True)
plt.show()

# %%
# Local circular features (persistence on each fiber preimage)
fiber_ids, dense_idx_list, rips_list = cb.get_local_rips(
    flow_data,
    flow_cover.U,
    to_view=[0, 3, 8],
    maxdim=1,
    n_perm=500 if RUN_HEAVY else 200,
    random_state=0,
)

cb.plot_local_rips(
    fiber_ids,
    rips_list,
    n_cols=3,
    titles="default",
    font_size=16,
)
plt.show()

# %%
# Main bundle construction: local circular coords + Procrustes transitions
flow_bundle = cb.build_bundle(
    flow_data,
    flow_cover,
    # CircularCoords_cls=CircularCoords,  # optional
    show=True,
)

# %%
# Global trivialization (torus case should be orientable/trivial in this sense)
flow_triv_result = flow_bundle.get_global_trivialization()
print("Global coordinates computed.")

# %%
# Visualize patches arranged by (base angle, fiber coordinate)
per_row = 5
per_col = 9
coords = np.column_stack([predom_dirs.reshape(-1, 1), flow_triv_result.F.reshape(-1, 1)])

fig = vz.lattice_vis(
    flow_data,
    coords,
    patch_vis,
    per_row=per_row,
    per_col=per_col,
    figsize=19,
    thumb_px=350,
    dpi=350,
)
plt.show()

# %%
# Klein bottle model: high-contrast natural image patches
# -------------------------------------------------------
n_img_patches = 5000
n_img = 3

rng = np.random.default_rng(0)
img_data = sy.sample_nat_img_kb(n_img_patches, n=n_img, rng=rng)[0]
print(f"{n_img_patches} {n_img}-by-{n_img} natural image patches generated.")

# %%
# Predominant gradient direction in RP^1
grad_dirs = sy.get_gradient_dirs(img_data)[0]
print("Predominant gradient directions computed.")

# %%
# Visualize a sample arranged by direction
n_samples = 30
label_func = [fr"$\theta = {np.round(t/np.pi, 2)}$" + r"$\pi$" for t in grad_dirs]

fig = vz.show_data_vis(
    img_data,
    patch_vis,
    label_func=label_func,
    angles=grad_dirs,
    sampling_method="angle",
    max_samples=n_samples,
)
plt.show()

# %%
# Persistent homology evidence (optional)
if RUN_HEAVY:
    dgms_2 = ripser(img_data, coeff=2, maxdim=2, n_perm=500)["dgms"]
    dgms_3 = ripser(img_data, coeff=3, maxdim=2, n_perm=500)["dgms"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    plot_diagrams(dgms_2, ax=axes[0], title="coeff = 2")
    plot_diagrams(dgms_3, ax=axes[1], title="coeff = 3")
    plt.tight_layout()
    plt.show()
else:
    print("Skipping Ripser demo (set RUN_HEAVY = True to run).")

# %%
# Cover RP^1
n_img_landmarks = 12
img_landmarks = np.linspace(0, np.pi, n_img_landmarks, endpoint=False)
img_overlap = 1.4
img_radius = img_overlap * np.pi / (2 * n_img_landmarks)

img_cover = cb.MetricBallCover(grad_dirs, img_landmarks, img_radius, metric=rp1_metric())
img_cover.build()

img_cover.summarize(plot=True)
plt.show()

# %%
# Local circular features
fiber_ids, dense_idx_list, rips_list = cb.get_local_rips(
    img_data,
    img_cover.U,
    to_view=[0, 3, 8],
    maxdim=1,
    n_perm=500 if RUN_HEAVY else 200,
    random_state=0,
)

cb.plot_local_rips(
    fiber_ids,
    rips_list,
    n_cols=3,
    titles="default",
    font_size=16,
)
plt.show()

# %%
# Main bundle construction (nontrivial global structure; expect obstruction)
img_bundle = cb.build_bundle(
    img_data,
    img_cover,
    # CircularCoords_cls=CircularCoords,  # optional
    show=True,
)

# %%
# Global trivialization after "cut" (as implemented in your pipeline)
img_triv_result = img_bundle.get_global_trivialization()
print("Global coordinates computed.")

# %%
# Show coordinatized patches
per_row = 5
per_col = 9
coords = np.column_stack([grad_dirs.reshape(-1, 1), img_triv_result.F.reshape(-1, 1)])

fig = vz.lattice_vis(
    img_data,
    coords,
    patch_vis,
    per_row=per_row,
    per_col=per_col,
    figsize=19,
    thumb_px=350,
    dpi=350,
)
plt.show()

# %%
# Orientation cocycle comparisons on the circle nerve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=150)

flow_bundle.show_circle_nerve(title="Optical Flow Patches", ax=ax1, show=False)
img_bundle.show_circle_nerve(title="Natural Image Patches", ax=ax2, show=False)

plt.tight_layout()
plt.show()
