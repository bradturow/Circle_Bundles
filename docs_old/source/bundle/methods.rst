User-facing methods
===================

After constructing a bundle, users typically interact with it via a small number of methods.
These methods are intended to be stable and documented; other methods and attributes are
internal or advanced.

Analysis methods
----------------

Persistence
^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.get_persistence

Max-trivial subcomplex
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.get_max_trivial_subcomplex

Global trivialization
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.get_global_trivialization

Derived datasets
----------------

Frame dataset
^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.get_frame_dataset

Bundle map (fiber coordinates)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.get_bundle_map

Pullback total space
^^^^^^^^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.get_pullback_data
