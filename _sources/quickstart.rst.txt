Quickstart
==========

This page demonstrates a minimal end-to-end example using a classical circle bundle:
the Hopf fibration :math:`S^3 \to S^2`. The Hopf fibration is a non-trivial orientable circle bundle with Euler number :math:`\pm 1`.

Minimal working example
-----------------------

.. code-block:: python

   import numpy as np
   import circle_bundles as cb

   # Sample points on S^3
   n_samples = 5000
   rng = np.random.default_rng(0)
   s3_data = cb.sample_sphere(n_samples, dim=3, rng=rng)

   # Hopf projection S^3 -> S^2
   base_points = cb.hopf_projection(s3_data)

   # Build an open cover of S^2
   n_landmarks = 60
   s2_cover = cb.get_s2_fibonacci_cover(
       base_points,
       n_vertices=n_landmarks,
   )

   bundle = cb.Bundle(X = s3_data, U = s2_cover.U)
   triv_result = bundle.get_local_trivs()
   class_result = bundle.get_classes(show_classes = True)