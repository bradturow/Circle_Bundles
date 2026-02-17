Core pipeline
=============

The main entry point is :class:`circle_bundles.Bundle`.

Typical workflow
----------------

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.Bundle(X, U=U)
   lt = bundle.get_local_trivs()
   cp = bundle.get_classes()
   bm = bundle.get_bundle_map(pou = pou)

   bundle.summary()

Reference
---------

.. autoclass:: circle_bundles.Bundle
   :members: get_local_trivs, get_classes, get_global_trivialization, get_bundle_map, get_frame_dataset, show_nerve, show_circle_nerve, compare_trivs, summary
   :show-inheritance:
