Bundles
=======

Bundles are constructed using :func:`circle_bundles.build_bundle`.

The result is a :class:`BundleResult` object, which stores all relevant
information about the reconstructed bundle and provides computational
and visualization methods.

Building a bundle
-----------------

.. autofunction:: circle_bundles.build_bundle

BundleResult
------------

.. autoclass:: circle_bundles.bundle.BundleResult
   :members:
   :member-order: bysource
   :undoc-members:

User-facing methods include:

- ``get_persistence``
- ``get_global_trivialization``
- ``get_frame_dataset``
- ``get_pullback_data``
- ``show_bundle``
- ``show_nerve``
- ``show_max_trivial``
- ``show_circle_nerve``
- ``compare_trivs``

All other attributes and methods should be considered internal unless
explicitly documented.
