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
   :maxdepth: 2
   :caption: Extras

   extras/index

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/index
