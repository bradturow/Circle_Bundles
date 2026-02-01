#!/usr/bin/env python
# coding: utf-8

# # Star Density Unwrapping
# This notebook demonstrates usage of `circle_bundles` on a dataset endowed with a (non-Euclidean) metric.
# 
# Given a vector $v\in \mathbb{S}^{2}$ and an integer $p\in \mathbb{Z}$, one has a continuous, free right $\mathbb{S}^{1}$-action on $\mathbb{S}^{3}$ defined by $q\cdot e^{i\theta} = qe^{i\frac{v}{2}}$ (here we are identifying $\mathbb{S}^{3}$ with the unit quaternions and $\mathbb{S}^{2}$ with the pure imaginary unit quaternions). The hopf projection map $\pi:\mathbb{S}^{3}\to \mathbb{S}^{2}$ defined by $\pi(q) = qvq^{-1}$ is equivariant with respect to this action, so we obtain a new principal $\mathbb{S}^{1}$-bundle whose total space is the lens space $\mathbb{S}^{3}/\mathbb{Z}_{p}$ (and whose projection map is defined by $\widetilde{\pi}([q]) = [\pi(q)]$).  The Euler number of this bundle is $\pm p$.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


import circle_bundles as cb

# Local circular coordinates
from dreimac import CircularCoords


# First, generate the dataset:

# In[ ]:


#Create the template triangle mesh
mesh = cb.make_star_pyramid(n_points = 5, height = 1)

n_samples = 5000
rng = np.random.default_rng(0)
R = cb.sample_so3(n_samples, rng=rng)[0]  #generate a random sample of SO(3)

mesh_data = cb.get_mesh_sample(mesh, R)  #generate the mesh dataset for visualization

grid_size = 32  #density resolution
sigma = 0.05  

data = cb.make_rotated_density_dataset(
    mesh,
    R,
    grid_size = grid_size,
    sigma = sigma,
)


#Create visualization functions for triangle meshes and 3D densities
vis_mesh = cb.make_star_pyramid_visualizer(mesh)
vis_density = cb.make_density_visualizer(grid_size=grid_size)


# Densities are stored as vectors of length $32^{3} = 32768$.  For visualization, the original meshes are stored as vectors of length $11\times 3 = 33$.

# 
# 
# View a small sample of the data, represented by 2D projections of the 3D densities and also by the triangle meshes used to produce the densities.

# In[ ]:


n_to_show = 8
fig = cb.show_data_vis(data, 
                       vis_density, 
                       max_samples = n_to_show, 
                       n_cols = n_to_show, 
                       sampling_method = 'first')
plt.show()


fig = cb.show_data_vis(mesh_data, 
                       vis_mesh, 
                       max_samples = n_to_show, 
                       n_cols = n_to_show, 
                       pad_frac = 0.3, 
                       sampling_method = 'first')
plt.show()


# Note: 2D projections of the 3D densities are computed by summing intensities along the z-axis, shown here as perpendicular to the screen.  Reflecting a pyramid mesh through the z-axis produces a very different 3D density, but the 2D projections appear the same.  On the other hand, two pyramid meshes which differ by a rotation in the symmetry group yield indistinguishable 3D densities -- this is a genuine symmetry of the dataset.  
# 
# Now, compute the base projections of the densities to $\mathbb{RP}^{2}$ (represented as unit vectors in the upper hemisphere of $\mathbb{S}^{2}$):

# In[ ]:


base_points = cb.get_density_axes(data)


# Construct an open cover of $\mathbb{RP}^{2}$ using a collection of nearly equidistant landmark points (see reference section):

# In[ ]:


n_landmarks = 60
rp2_cover = cb.make_rp2_fibonacci_star_cover(base_points, n_pairs = n_landmarks)

summ = rp2_cover.summarize(plot = True)


# Optionally run the cell below to view a Plotly visualization of the nerve of the open cover:

# In[ ]:


fig = rp2_cover.show_nerve()


# Compute a persistence diagram for the data in each set $\pi^{-1}(U_{j})$:

# In[ ]:


fiber_ids, dense_idx_list, rips_list = cb.get_local_rips(
    data,
    rp2_cover.U,
    p_values=None,
    to_view = [5,32,49],
    maxdim=1,
    n_perm=500,
    random_state=None,
)

fig, axes = cb.plot_local_rips(
    fiber_ids,
    rips_list,
    n_cols=3,
    titles='default',
    font_size=20,
)


# Observe that each set shows a pair of persistent classes in dimensions 0 and 1, suggesting the presence of a pair of local circular clusters in each set. This reflects the fact that the underlying pyramid mesh does not have additional 2-fold symmetry with respect to its rotational axis of symmetry. 

# Optionally show a visualization of a 'fat fiber' of the projection map, with densities represented by both 2D projections and the underlying meshes

# In[ ]:


center_ind = 148
r = 0.2
dist_mat = cb.RP2UnitVectorMetric().pairwise(X=base_points)
nearby_indices = np.where(dist_mat[center_ind] < r)[0]

fiber_data = data[nearby_indices]
vis_data = mesh_data[nearby_indices]


fig = plt.figure(figsize=(18, 6), dpi=120)
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax3 = fig.add_subplot(1, 3, 3, projection="3d")

# PCA labeled with meshes
cb.fiber_vis(
    fiber_data,
    vis_mesh,
    vis_data=vis_data,
    max_images=50,
    zoom=0.08,
    ax=ax1,
    show=False,
)
ax1.set_title("Fiber PCA")

# PCA labeled with density projections
cb.fiber_vis(
    fiber_data,
    vis_func=vis_density,
    max_images=50,
    zoom=0.05,
    ax=ax2,
    show=False,
)
ax2.set_title("Fiber PCA")

# Base visualization
cb.base_vis(
    base_points,
    center_ind,
    r,
    dist_mat,
    use_pca=False,
    ax=ax3,
    show=False,
)
ax3.set_title("Base neighborhood")

plt.tight_layout()
plt.show()


# Use fiberwise clustering to separate the data in each open set into its two components and track the global connected components

# In[ ]:


eps_values = 0.0125*np.ones(len(rp2_cover.U))   #Choose an epsilon value based on 0D persistence diagrams
min_sample_values = 5*np.ones(len(rp2_cover.U))

components, G, graph_dict, cl, summary = cb.fiberwise_clustering(data, 
                                                              rp2_cover.U, 
                                                              eps_values, 
                                                              min_sample_values)

n_clusters = len(np.unique(components))-1
print(f'Total number of global clusters: {n_clusters}')


# Observe that the data in each fiber was separated into two distinct clusters, but globally, the clusters organize into a single connected component.  This is expected, because the underlying model for the dataset has a single connected component.
# 
# Optionally show the 1-skeleton of the nerve of the cover labeled by the $\mathbb{Z}_{2}$ permutation cocycle -- red edges indicate that the local +/- cluster labels are reversed on an overlap, where as blue edges indicate that the local +/- cluster labels agree.  

# In[ ]:


signs = cb.get_cocycle_dict(G)
signs_O1 = {edge:(-1) ** signs[edge] for edge in signs.keys()}

dist_mat = rp2_cover.metric.pairwise(X = rp2_cover.landmarks, Y = rp2_cover.base_points)

node_labels = [f"{i+1}" for i in range(rp2_cover.landmarks.shape[0])]

fig, axes = cb.nerve_vis(
    rp2_cover,
    cochains={1:signs_O1},
    base_colors={0:'black', 1:'black', 2:'pink'},
    cochain_cmaps={1:{1: 'blue', -1:'darkred'}},
    opacity=0,
    node_size=18,
    line_width=1.5,
    node_labels=node_labels,
    fontsize=9,
    font_color='lavender',
    title='1-Skeleton Of The Nerve Of The Cover'
)

plt.show()


# Optionally show a visualization of the local clusters. Each cluster is labeled by the pyramid mesh underlying a  representative 3D density.

# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

for ax, g in zip(axes, [0, 1]):

    sample_inds = []

    # Choose a representative for each cluster with label g
    for node in G.nodes():
        (j, k) = node
        if k == g:
            node_inds = cl[j] == k
            min_idx_local = np.argmin(dist_mat[j, node_inds])
            min_index = np.where(node_inds)[0][min_idx_local]
            sample_inds.append(min_index)

    sample_inds = np.array(sample_inds, dtype=int)

    sample_mesh_data = mesh_data[sample_inds]
    sign = "+" if g == 0 else "-"
    cb.nerve_vis(
        rp2_cover,
        cochains={1: signs_O1},
        base_colors={0: "black", 1: "black", 2: "pink"},
        cochain_cmaps={1: {1: "blue", -1: "lightgray"}},
        node_size=25,
        line_width=1,
        node_labels=None,
        fontsize=16,
        font_color="white",
        vis_func=vis_mesh,
        data=sample_mesh_data,
        image_zoom=0.09,
        title = rf"$X^{{{sign}}}$ Clusters",
        ax=ax,                
        show=False,           
    )

plt.show()


# Use the local cluster labels and interchange data on overlaps to lift the base map from $\mathbb{RP}^{2}$ to $\mathbb{S}^{2}$:

# In[ ]:


lifted_base_points = cb.lift_base_points(G, cl, base_points)

n_landmarks = 80       #Use more base points this time to compute the local circular coordinates

s2_cover = cb.make_s2_fibonacci_star_cover(lifted_base_points, n_vertices = n_landmarks)
summ = s2_cover.summarize(plot = True)


# Compute local trivializations, approximate transition matrices and characteristic classes

# In[ ]:


cc_alg = cb.DreimacCCConfig(
    CircularCoords_cls = CircularCoords,
    landmarks_per_patch = 250,
    prime = 41,
)


s2_bundle = cb.build_bundle(
    data,
    s2_cover,
    cc = cc_alg,
    show=True
)


# In[ ]:


fig = s2_bundle.compare_trivs()
plt.show()


# Now, restrict the original bundle to the equator $\mathbb{S}^{1}\subset \mathbb{S}^{2}$:

# In[ ]:


eps = 0.15  # thickness of equatorial band

# Points near the equator: last coordinate close to 0
eq_mask = np.abs(s2_cover.base_points[:, -1]) < eps

eq_data = s2_bundle.data[eq_mask]
eq_mesh_data = mesh_data[eq_mask]


eq_base_angles = np.arctan2(s2_cover.base_points[eq_mask, 1], s2_cover.base_points[eq_mask, 0]) % (2*np.pi)

print(f"Equator band: {eq_data.shape[0]} / {s2_bundle.data.shape[0]} samples (eps={eps}).")


# Constuct an open cover of $\mathbb{S}^{1}$ by metric balls around equally-spaced landmark points:

# In[ ]:


n_landmarks = 12
landmarks = np.linspace(0, 2*np.pi, n_landmarks, endpoint= False)
overlap = 1.6
radius = overlap* np.pi/n_landmarks

eq_cover = cb.MetricBallCover(eq_base_angles, landmarks, radius, metric = cb.S1AngleMetric())
eq_cover_data = eq_cover.build()

eq_summ = eq_cover.summarize(plot = True)


# Construct local circular coordinates using Dreimac, compute approximate transition matrices and characteristic classes:

# In[ ]:


cc_alg = cb.DreimacCCConfig(
    CircularCoords_cls = CircularCoords,
    landmarks_per_patch = 250,
    prime = 41,
)

eq_bundle = cb.build_bundle(
    eq_data,
    eq_cover,
    cc = cc_alg,
    show=True,
)


# Observe that the restricted bundle is orientable, hence trivial, as expected. Construct a global toroidal coordinate system by synchronizing local circular coordinates:

# In[ ]:


eq_triv_result = eq_bundle.get_global_trivialization()


# Finally, show a visualization of the coordinatized densities, represented by both the corresponding meshes and 2D projections:

# In[ ]:


#Get a visualization of the data arranged by assigned coordinates using meshes
coords = np.array([eq_base_angles, eq_triv_result.F]).T
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), dpi=200)

fig = cb.lattice_vis(
    eq_mesh_data,
    coords,
    vis_mesh, 
    per_row=7,          
    per_col=7,          
    figsize=10, 
    thumb_px=100,
    dpi=200,
    ax = ax1,
)

#Get a visualization of the data arranged by assigned coordinates using meshes
fig = cb.lattice_vis(
    eq_data,
    coords,
    vis_density, 
    per_row=7,          
    per_col=7,          
    figsize=11, 
    thumb_px=120,
    dpi=200,
    ax = ax2,
)
plt.show()

