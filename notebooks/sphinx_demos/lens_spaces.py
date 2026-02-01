#!/usr/bin/env python
# coding: utf-8

# # Lens Spaces $L(p,1) = \mathbb{S}^{3}/\mathbb{Z}_{p}$ As Circle Bundles Over $\mathbb{S}^{2}$
# This notebook demonstrates usage of `circle_bundles` on a dataset endowed with a (non-Euclidean) metric.
# 
# Given a vector $v\in \mathbb{S}^{2}$ and an integer $p\in \mathbb{Z}$, one has a continuous, free right $\mathbb{S}^{1}$-action on $\mathbb{S}^{3}$ defined by $q\cdot e^{i\theta} = qe^{i\frac{v}{2}}$ (here we are identifying $\mathbb{S}^{3}$ with the unit quaternions and $\mathbb{S}^{2}$ with the pure imaginary unit quaternions). The hopf projection map $\pi:\mathbb{S}^{3}\to \mathbb{S}^{2}$ defined by $\pi(q) = qvq^{-1}$ is equivariant with respect to this action, so we obtain a new principal $\mathbb{S}^{1}$-bundle whose total space is the lens space $\mathbb{S}^{3}/\mathbb{Z}_{p}$ (and whose projection map is defined by $\widetilde{\pi}([q]) = [\pi(q)]$).  The Euler number of this bundle is $\pm p$.

# In[3]:


import numpy as np
import circle_bundles as cb


# First, generate a sample of the 3-sphere:

# In[5]:


n_samples = 10000
s3_data = cb.sample_sphere(n = n_samples, dim = 3)


# Compute base projections to $\mathbb{S}^{2}$ using the Hopf projection map

# In[7]:


v = np.array([1.0, 0.0, 0.0]) 
base_points = cb.hopf_projection(s3_data, v = v)


# Construct an open cover of $\mathbb{S}^{2}$:

# In[8]:


n_landmarks = 80
s2_cover = cb.make_s2_fibonacci_star_cover(base_points, n_vertices = n_landmarks)
s2_summ = s2_cover.summarize(plot = True)


# $\textbf{Main bundle computation}$: Compute local trivializations and characteristic classes using a quotient metric on $\mathbb{S}^{3}$.

# In[10]:


p = 4  #CHOOSE: a p-value for the quotient metric

total_metric = cb.S3QuotientMetric(p=p, v_axis = v, antipodal = False)    

lens_bundle = cb.build_bundle(
    s3_data,
    s2_cover,
    total_metric = total_metric,
    show=True,
)

