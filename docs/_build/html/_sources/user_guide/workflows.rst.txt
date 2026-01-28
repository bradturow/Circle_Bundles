Workflows
=========

Typical workflows:

1. Build a bundle from data and a cover:

   .. code-block:: python

      import circle_bundles as cb
      bundle = cb.build_bundle(data, cover)

2. Inspect quality and class representatives:

   .. code-block:: python

      bundle.quality
      bundle.classes

3. Compute persistence and max-trivial subcomplex:

   .. code-block:: python

      pers = bundle.get_persistence()
      mt = bundle.get_max_trivial_subcomplex()

4. Construct global trivializations / bundle maps:

   .. code-block:: python

      gt = bundle.get_global_trivialization()
      bm = bundle.get_bundle_map()

5. Work in the pullback total space with a product metric:

   .. code-block:: python

      pb = bundle.get_pullback_data(base_weight=1.0, fiber_weight=1.0)
      D = pb.metric.pairwise(pb.total_data)
