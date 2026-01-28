BundleResult overview
=====================

A :class:`~circle_bundles.bundle.BundleResult` packages the core outputs of the reconstruction.
It is returned by :func:`~circle_bundles.bundle.build_bundle`.

Mental model
------------

A bundle object stores:

- a **cover** of the base space (charts, nerve, partition of unity),
- a **local trivialization** (local circle coordinates on charts),
- an estimated **O(2) cocycle** (transition functions on overlaps),
- **characteristic class representatives**,
- **quality diagnostics**,
- lightweight **provenance** in ``meta``.

Key attributes
--------------

``cover``
  The cover object (membership matrix ``U``, partition of unity, nerve accessors, landmarks, etc.).

``data``
  The original dataset array.

``local_triv``
  Local circle coordinates on each chart.

``cocycle``
  Transition data and cocycle representation on the nerve.

``classes``
  Characteristic class representatives.

``quality``
  Diagnostics and consistency summaries.

``meta``
  Minimal provenance describing semantic construction choices (kept intentionally small).

Caching
-------

Methods named ``get_*`` cache results in ``bundle._cache`` and may also populate a corresponding
attribute (for example ``bundle.persistence``) when called with default settings.

API reference
-------------

.. autoclass:: circle_bundles.bundle.BundleResult
   :members:
   :member-order: bysource
