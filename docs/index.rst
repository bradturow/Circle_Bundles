circle_bundles
==============

Tools for detecting, visualizing, and classifying circle-bundle structure in data.

The recommended usage is:

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.build_bundle(data, cover, show=True)
   pers = bundle.get_persistence()
   gt = bundle.get_global_trivialization()
   bm = bundle.get_bundle_map()
   pb = bundle.get_pullback_data()

This documentation is organized around:

- **Quickstart**: the shortest path to running the full pipeline.
- **User guide**: conceptual and workflow documentation.
- **API reference**: the curated public surface (stable).

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/concepts
   user_guide/workflows

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/public_api
   api/covers
   api/metrics
   api/analysis
