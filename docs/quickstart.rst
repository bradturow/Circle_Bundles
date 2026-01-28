Quickstart
==========

This page shows the canonical workflow using the curated public API:

.. code-block:: python

   import circle_bundles as cb

   # 1) Build a cover over the base points (example: metric ball cover)
   cover = cb.MetricBallCover(base_points, metric=cb.EuclideanMetric(), r=0.25)
   cover.build()

   # 2) Build the bundle
   bundle = cb.build_bundle(total_space_data, cover, show=True)

   # 3) Downstream cached computations
   pers = bundle.get_persistence(show=True)
   mt = bundle.get_max_trivial_subcomplex()
   gt = bundle.get_global_trivialization()

   # 4) Bundle map + pullback total space (base | fiber) with a product metric
   bm = bundle.get_bundle_map()
   pb = bundle.get_pullback_data()

   # Distance matrix using the pullback product metric
   D = pb.metric.pairwise(pb.total_data)

Notes
-----

- The primary entry point is :func:`circle_bundles.build_bundle`.
- Downstream computations live as cached methods on :class:`circle_bundles.BundleResult`.
- Visualization utilities live under :mod:`circle_bundles.viz` (optional dependencies).
