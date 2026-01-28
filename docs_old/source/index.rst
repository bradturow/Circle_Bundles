circle_bundles
==============

``circle_bundles`` is a Python library for detecting, visualizing, and classifying
**circle-bundle structure in data**. It provides tools for constructing covers,
building discrete circle bundles from point clouds, computing characteristic
classes (such as the Euler class), and visualizing both local trivializations
and global topology.

The library is designed for researchers working in topological data analysis,
geometry, and applications such as optical flow and synthetic geometric models.

Recommended usage::

    import circle_bundles as cb


Installation
------------

From source (recommended during development):

.. code-block:: bash

   git clone https://github.com/<your-username>/circle_bundles.git
   cd circle_bundles
   pip install -e .

(Once released, this section can be updated to ``pip install circle-bundles``.)


Quickstart: Hopf fibration :math:`\mathbb{S}^3 \to \mathbb{S}^2`
---------------------------------------------------------------

This example constructs the classical **Hopf fibration**, viewed as a circle
bundle over the 2-sphere, from sampled data.

.. code-block:: python
   :caption: Quickstart: Hopf fibration S³ → S²

   import numpy as np
   import circle_bundles as cb

   # ------------------------------------------------------------
   # Quickstart: Hopf fibration  S^3 -> S^2
   # ------------------------------------------------------------

   rng = np.random.default_rng(0)

   # Sample points on S^3 ⊂ R^4  (i.e. unit vectors in R^4)
   n_samples = 5000
   s3 = cb.sample_sphere(n_samples, dim=3, rng=rng)

   # Hopf projection to S^2 ⊂ R^3
   v = np.array([1.0, 0.0, 0.0])
   base_points = cb.hopf_projection(s3, v=v)

   # Cover of S^2 using Fibonacci landmarks
   n_landmarks = 60
   cover = cb.make_s2_fibonacci_star_cover(base_points, n_vertices=n_landmarks)

   # Build the bundle and visualize local trivializations
   bundle = cb.build_bundle(s3, cover, show=True)

Running this code will:
- construct a cover of the base space :math:`\mathbb{S}^2`,
- build a discrete circle bundle from the sampled data,
- compute characteristic class information internally, and
- launch an interactive visualization of the bundle structure.

This example serves as a minimal end-to-end demonstration of the core workflow.


Documentation overview
----------------------

The documentation is organized by **user intent**, rather than internal modules.

- **Core workflow**: how to build and visualize bundles
- **Covers**: base-space covers and landmark constructions
- **Metrics**: base, fiber, and quotient metrics
- **Analysis**: clustering and local topology diagnostics
- **Synthetic data**: geometric and image-based test models
- **Optical flow**: tools for real vision datasets
- **Visualization**: plotting and interactive inspection utilities

For detailed function and class documentation, see the API reference below.


Contents
--------

.. toctree::
   :maxdepth: 2

   bundle/index
   reference/index
