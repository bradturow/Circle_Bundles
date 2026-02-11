Theory
======

``circle_bundles`` is a toolkit for analyzing and coordinatizing high-dimensional
datasets which approximately arise as noisy samples from the total spaces of
circle bundles.

This page provides a brief overview of the underlying theory and the algorithms
implemented in the package. For a complete theoretical development and proofs,
see the accompanying paper [1].

Introduction
------------

We use the term *circle bundle* to refer to any fiber bundle whose fiber is the
circle :math:`\mathbb{S}^1`, not necessarily a principal bundle.

Up to isomorphism, circle bundles over a metric space :math:`B` are completely
classified by a pair of characteristic classes:

- the **orientation class** :math:`\mathrm{sw}_1 \in H^1(B; \mathbb{Z}_2)`, and
- the **(twisted) Euler class** :math:`\tilde e \in H^2(B; \widetilde{\mathbb{Z}})`,

which encode global obstructions to orientability and triviality, respectively.
Crucially, these invariants admit formulations that can be recovered from purely
local data, making them well suited for data-analytic settings.

A nonzero characteristic class indicates that the bundle is *non-trivial*, in the
sense that its total space :math:`E` is not homeomorphic to the product
:math:`B \times \mathbb{S}^1`. Only trivial bundles admit global circular
coordinate systems.

In general, any circle bundle over a compact metric space can be realized
(non-uniquely) as the pullback of the tautological circle bundle

.. math::

   V(2,d) \times_{O(2)} \mathbb{S}^1 \;\longrightarrow\; \mathrm{Gr}(2,d),

for sufficiently large :math:`d`. When the total space :math:`E` is a
low-dimensional manifold embedded in :math:`\mathbb{R}^D` with :math:`D` large,
it is often possible to choose :math:`d` such that :math:`2d \ll D`. This
perspective underlies the topology-respecting dimensionality reduction methods
implemented in ``circle_bundles``.

In [1], the authors introduce the notion of a *discrete approximate circle
bundle*, together with algorithms for estimating characteristic classes and for
constructing global coordinate systems that respect the inferred topology. The
``circle_bundles`` package implements several of these algorithms.

Algorithms
----------

Overview
^^^^^^^^

Given a dataset :math:`X` equipped with a projection :math:`\pi : X \to B` and a
finite open cover :math:`\mathcal{U} = \{U_j\}_{j=1}^n` of the base space, the
pipeline proceeds in three main stages:

1. **Cocycle fitting** (local trivializations and transition matrices),
2. **Classification** (estimation of characteristic classes),
3. **Coordinatization** (construction of global bundle coordinates).

Each stage is described in detail below.

Cocycle Fitting
^^^^^^^^^^^^^^^

Given a family of local circular coordinate functions

.. math::

   f_j : \pi^{-1}(U_j) \to \mathbb{S}^1,

the goal is to estimate transition matrices
:math:`\Omega_{jk} \in O(2)` such that

.. math::

   f_j(x) \;\approx\; \Omega_{jk} f_k(x),
   \quad x \in \pi^{-1}(U_j \cap U_k).

This is formulated as a least-squares Procrustes problem over :math:`O(2)` on each
overlap.

A corresponding **weight function**

.. math::

   w : N(\mathcal{U})^{(1)} \to [0,\infty)

is defined by measuring the mean squared alignment error on overlaps. These
weights quantify the reliability of each transition and induce a filtration on
the nerve of the cover.

Classification
^^^^^^^^^^^^^^

The weight function :math:`w` induces a **weights filtration**
:math:`\{W^r\}_{r \ge 0}` on the nerve :math:`N(\mathcal{U})`, where :math:`W^r`
denotes the maximal subcomplex whose 1-skeleton consists of edges with weight at
most :math:`r`.

Given the simplicial :math:`O(2)`-valued cocycle :math:`\Omega` and the weights
:math:`w`, the algorithm computes cochains

.. math::

   \mathrm{sw}_1 \in C^1(N(\mathcal{U}); \mathbb{Z}_2),
   \qquad
   \tilde e \in C^2(N(\mathcal{U}); \mathbb{Z}),

representing estimates of the orientation class and twisted Euler class,
respectively.

The **cobirth** and **codeath** of these classes with respect to the filtration
:math:`\{W^r\}` are computed. When well-defined, the twisted Euler number (up to
sign) is also extracted at the cobirth scale.

Coordinatization
^^^^^^^^^^^^^^^^

Let :math:`(b_{\mathrm{sw}_1}, d_{\mathrm{sw}_1})` and
:math:`(b_{\tilde e}, d_{\tilde e})` denote the lifetimes of the estimated
characteristic classes.

For a given weight threshold :math:`r`, the base space is cut along unreliable
overlaps to produce a modified space :math:`B^r` with an induced cover
:math:`\mathcal{U}^r` satisfying

.. math::

   N(\mathcal{U}^r) = W^r.

- For :math:`r \le \min\{b_{\mathrm{sw}_1}, b_{\tilde e}\}`, a **classifying map**

  .. math::

     \mathrm{cl}^r : B^r \to \mathrm{Gr}(2,d)

  is constructed, together with an associated bundle map

  .. math::

     F^r : X \to V(2,d) \times_{O(2)} \mathbb{S}^1 \subset \mathbb{R}^{2d}.

- For :math:`r \le \min\{d_{\mathrm{sw}_1}, d_{\tilde e}\}`, the bundle over
  :math:`B^r` is trivial, and a **global trivialization**

  .. math::

     F^r : X \to B^r \times \mathbb{S}^1

  is produced.

Algorithm 1: Local Trivializations and Transition Matrices
----------------------------------------------------------

**Inputs**

- Dataset :math:`X` (point cloud or distance matrix)
- Projection :math:`\pi : X \to B` and good open cover :math:`\mathcal{U}`
- Binary membership array encoding :math:`\pi(x_m) \in U_j`

Optional inputs:

- Local circular coordinates :math:`\{f_j\}`
- Choice of local coordinate algorithm (default: PCA2 or UMAP2)

**Outputs**

- Local circular coordinates :math:`\{f_j\}`
- Transition matrices :math:`\{\Omega_{jk} \in O(2)\}`
- Weight function :math:`w : N(\mathcal{U})^{(1)} \to [0,\infty)`

Algorithm 2: Characteristic Classes
-----------------------------------

**Inputs**

- Injective weight function :math:`w`
- Simplicial :math:`O(2)`-valued 1-cochain :math:`\Omega`

**Outputs**

- Cochains :math:`\mathrm{sw}_1` and :math:`\tilde e`
- Cobirth and codeath scales
- Twisted Euler number (when defined)

Algorithm 3: Pullback Bundle Coordinates
----------------------------------------

**Inputs**

- Dataset :math:`X`
- Cover membership array
- Local circular coordinates :math:`\{f_j\}`
- Transition matrices :math:`\{\Omega_{jk}\}`
- Partition of unity subordinate to :math:`\mathcal{U}`
- Weight function :math:`w`
- Threshold :math:`r`

**Outputs**

- Classifying map :math:`\mathrm{cl}^r` and bundle map :math:`F^r`
- Global trivialization when it exists

References
----------

.. [1]  
   *Discrete Approximate Circle Bundles*.  
   arXiv:2508.12914 [math.AT], 2025.  
   https://doi.org/10.48550/arXiv.2508.12914
