Bundle objects
==============

The central object in ``circle_bundles`` is the :class:`~circle_bundles.bundle.BundleResult`.
A ``BundleResult`` represents the output of a discrete circle-bundle reconstruction:
it stores the core artifacts (cover, local trivializations, cocycle, characteristic classes)
and provides cached convenience methods for downstream analysis and visualization.

Quickstart
----------

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.build_bundle(data, cover, show=True)
   bundle.classes
   bundle.quality

After constructing a bundle, typical next steps include:

- computing persistence-style filtrations of characteristic class representatives,
- extracting a large subcomplex on which the representatives become coboundaries,
- computing global trivializations,
- building derived datasets (frame coordinates, pullback total spaces),
- visualizing the cover nerve and related structures.

.. toctree::
   :maxdepth: 1
   :caption: Bundle docs

   building
   overview
   methods
   visualization

