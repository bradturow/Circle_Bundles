Building a bundle
=================

A bundle reconstruction begins with:

- a dataset ``X`` (an array of samples),
- a projection ``π : X → B`` to a base space (represented implicitly by the cover),
- and a cover of the base space (charts + nerve + partition of unity).

The function :func:`~circle_bundles.bundle.build_bundle` runs the core pipeline:

1. compute local circle coordinates on each cover chart,
2. estimate O(2) transition functions on overlaps,
3. assemble an O(2) cocycle on the nerve,
4. compute characteristic class representatives,
5. compute diagnostic/quality summaries.

Minimal example
---------------

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.build_bundle(data, cover)

Circular coordinates options
----------------------------

The ``cc`` parameter selects how local circle coordinates are computed.
The default is a lightweight PCA-based fallback (``"pca2"``), but users can supply
alternative algorithms (e.g. a Dreimac config) or a custom callable.

Total-space metrics
-------------------

If ``total_metric`` is provided, local coordinate computations and overlap estimates
can use metric-aware distance matrices.

Reference angle / gauge convention
----------------------------------

The ``ref_angle`` parameter fixes a gauge convention used when reflections are present.

API reference
-------------

.. autofunction:: circle_bundles.bundle.build_bundle
