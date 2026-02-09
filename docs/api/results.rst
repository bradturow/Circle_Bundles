Result objects and dataclasses
==============================

Most computations return small, stable dataclass containers. These are designed
to be user-facing, pickle-friendly, and resilient to refactors of internal
implementations.

Core result containers
----------------------

These are the primary outputs returned by the high-level ``Bundle`` methods.

.. autosummary::
   :toctree: generated
   :nosignatures:

   circle_bundles.NerveSummary
   circle_bundles.LocalTrivsResult
   circle_bundles.ClassesAndPersistence
   circle_bundles.BundleMapResult


Lower-level result containers
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:
    
   circle_bundles.BundleQualityReport
   circle_bundles.O2Cocycle
   circle_bundles.PersistenceResult
   circle_bundles.CobirthResult
   circle_bundles.CodeathResult


Configuration objects
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   circle_bundles.DreimacCCConfig
   circle_bundles.FrameReducerConfig
