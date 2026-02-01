#!/usr/bin/env python
# coding: utf-8

# # Quotients $(\mathbb{S}^{2}\times\mathbb{S}^{1})/\mathbb{Z}_{2}$ As Circle Bundles Over $\mathbb{RP}^{2}$ 

# Consider the trivial circle bundle over $\mathbb{S}^{2}$ defined by the projection of the product $\mathbb{S}^{2}\times\mathbb{S}^{1}$ onto the first factor.  We define three quotient metrics on $\mathbb{S}^{2}\times\mathbb{S}^{1}$ corresponding to three different $\mathbb{Z}_{2}$-actions covering the antipodal map on $\mathbb{S}^{2}$:
# 
# $\textbf{1.}$ $(v,z)\cdot (-1) := (-v,z)$
# 
# $\textbf{2.}$ $(v,z)\cdot (-1) := (-v,-z)$
# 
# $\textbf{3.}$ $(v,z)\cdot (-1) := (-v,\bar{z})$
# 
# In each case, we obtain an induced circle bundle over $\mathbb{RP}^{2}$.  The quotient induced by $\textbf{1}$ above yields the trivial bundle; $\textbf{2}$ yields the (unique) non-trivial orientable circle bundle over $\mathbb{RP}^{2}$, and $\textbf{3}$ yields the non-orientable circle bundle over $\mathbb{RP}^{2}$ with (twisted) Euler number 0.   

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import circle_bundles as cb


# First, generate a sample of the product $\mathbb{S}^{2}\times\mathbb{S}^{1}$ as vectors in $\mathbb{R}^{5}$:

# In[ ]:


n_samples = 10000
sigma = 0.05  #noise level

data, base_points, angles = cb.sample_s2_trivial(
    n_points = n_samples,
    sigma = sigma,
    radius_clip = (0.0, 5.0),
)


# Next, construct a cover of $\mathbb{RP}^{2}$ using nearly-equidistant landmark points (see reference section for details):

# In[ ]:


n_landmarks = 80
rp2_cover = cb.make_rp2_fibonacci_star_cover(base_points, n_pairs = n_landmarks)
rp2_summ = rp2_cover.summarize(plot = True)


# Observe that the Euler characteristic of the nerve of our (good) open cover is $\chi = 80 - 237 + 158 = 1$, as expected for $\mathbb{RP}^{2}$.
# 
# Now, choose a quotient metric for the dataset and construct a bundle

# In[ ]:


total_metrics = [cb.RP2_TrivialMetric(), cb.RP2_TwistMetric(), cb.RP2_FlipMetric()]  


j = 0   #CHOOSE: a metric on the total space
total_metric = total_metrics[j]


bundle = cb.build_bundle(
    data,
    rp2_cover,
    total_metric = total_metric,
    show=True,
)


# Below we compute the persistence of each characteristic class representative with respect to the weights filtration of the nerve of our open cover of $\mathbb{RP}^{2}$ (see theory section for details).  We expect an orientable cocycle to become a coboundary after a single edge is removed from the nerve, but a non-orientable cocycle over $\mathbb{RP}^{2}$ may have a much longer 'lifetime.' The restriction of the total space of a non-orientable bundle over $\mathbb{RP}^{2}$ to any 'equator' $\mathbb{RP}^{1}\subset\mathbb{RP}^{2}$ is still non-trivial, so the maximal subcomplex of the nerve on which a representative cocycle restricts to a coboundary cannot have any non-trivial cycles.  

# In[ ]:


pers = bundle.get_persistence(show = True)


# Optionally run the cell below to see an interactive visualization of the weights filtration on the nerve of the cover.  Use the slider to see different stages of weights filtration.  Edges shown in red are removed to reach a subcomplex on which both characterstic class representatives become coboundaries.  

# In[ ]:


fig = bundle.show_max_trivial(show_labels = False)  #change to True to see set labels
plt.show()

