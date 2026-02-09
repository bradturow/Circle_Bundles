Result objects and dataclasses
==============================

Most computations return small, stable dataclass containers. These are designed
to be user-facing, pickle-friendly, and resilient to refactors of internal
implementations.

Core result containers
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   circle_bundles.LocalTrivAndCocycle
   circle_bundles.ClassesAndPersistence
   circle_bundles.BundleMapResult

Lower-level results (stable)
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   circle_bundles.LocalTrivResult
   circle_bundles.O2Cocycle
   circle_bundles.TransitionReport
   circle_bundles.BundleQualityReport
   circle_bundles.PersistenceResult
   circle_bundles.CobirthResult
   circle_bundles.CodeathResult
