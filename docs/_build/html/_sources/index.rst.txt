circle_bundles
==============

Tools for detecting, visualizing, and classifying circle-bundle structure in data.

The recommended usage is:

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.Bundle(X = data, U = U)
   triv_result = bundle.get_local_trivs()
   class_result = bundle.get_classes()
   bundle_map_result = bundle.get_bundle_map()

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

   tutorials/auto_examples/index
