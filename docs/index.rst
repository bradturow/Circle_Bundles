circle_bundles
==============

A toolkit for detecting, classifying, coordinatizing and visualizing circle bundle structures in data.

When to use this
----------------

High-dimensional datasets in computer vision, computational chemistry, and
motion tracking often concentrate near low-dimensional manifolds whose global
topology is too complex to capture with a single coordinate chart or a direct
persistent homology computation. ``circle_bundles`` is designed for the common
special case where the data is **locally circular**: nearby points are organized
along circles (or, more precisely, the data admits the structure of a *circle
bundle* over some base space).

The package provides a complete **local-to-global inference pipeline**:

1. **Detect** local circular structure via approximate local trivializations.
2. **Classify** the global topology by computing characteristic classes —
   discrete invariants that distinguish, for example, a torus from a Klein
   bottle, or a trivial bundle over S² from SO(3).
3. **Coordinatize** the dataset by constructing a bundle map that respects the
   discovered topology, enabling principled dimensionality reduction even when
   the bundle is non-trivial.

Because characteristic classes can be computed from purely local measurements
and are stable under perturbation, this pipeline is well-suited to noisy,
high-dimensional data where direct global methods are intractable.

Datasets that have been successfully analyzed with ``circle_bundles`` include
natural image patches (Klein bottle), optical flow fields (torus and extended
models), triangle meshes under rotation (SO(3)), and synthetic density functions
over RP². See the :doc:`tutorials <auto_examples/index>` for worked examples.

The recommended usage is:

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.Bundle(X = data, U = U)
   triv_result = bundle.get_local_trivs()
   class_result = bundle.get_classes()
   bundle_map_result = bundle.get_bundle_map(pou = pou)

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :caption: References

   citing
   references
