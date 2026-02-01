#!/usr/bin/env python
# coding: utf-8

# # Quotients $\mathbb{S}^{3}/(\mathbb{Z}_{2p}\rtimes\mathbb{Z}_{2})$ As Circle Bundles Over $\mathbb{RP}^{2}$

# We combine the ideas from the two previous examples to construct the non-orientable bundles over $\mathbb{RP}^{2}$ which also have non-trivial twisted Euler class. One can show that such bundles are classified up to isomorphism by the absolute values of their twisted Euler numbers. In particular, for any even integer $2p\in \mathbb{Z}$, we define a $\mathbb{Z}_{2p}$-action and a $\mathbb{Z}_{2}$-action on $\mathbb{S}^{3}\subset\mathbb{C}^{2}$ by $(w,z)\cdot m := (w,z)\cdot  e^{2\pi i\left(\frac{m}{2p}\right)}$ and $(w,z)\cdot (-1) = (-w,\bar{z})$. The resulting quotient space has the structure of a non-orientable circle bundle over $\mathbb{RP}^{2}$ with twisted Euler number $\pm p$.  

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import circle_bundles as cb


# First, generate a sample of $\mathbb{S}^{3}$ and compute base projections using the Hopf map (see previous examples):

# In[ ]:


n_samples = 10000
s3_data = cb.sample_sphere(n = n_samples, dim = 3)

v = np.array([1.0, 0.0, 0.0]) 
base_points = cb.hopf_projection(s3_data, v= v)


# Construct an open cover of $\mathbb{RP}^{2}$ using nearly-equidistant landmark points (see reference section):

# In[ ]:


# --- Construct an open cover of rp2 ---

n_landmarks = 80
rp2_cover = cb.make_rp2_fibonacci_star_cover(base_points, n_pairs = n_landmarks)
rp2_summ = rp2_cover.summarize(plot = True)


# Now, choose a quotient metric for the total space and construct a bundle:

# In[ ]:


p = 3  #CHOOSE: a p-value for the quotient metric

total_metric = cb.S3QuotientMetric(p = 2*p, v_axis = v, antipodal = True)

bundle = cb.build_bundle(
    s3_data,
    rp2_cover,
    total_metric = total_metric,
    show=True,
)

