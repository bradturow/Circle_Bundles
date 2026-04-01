circle_bundles
==============

A toolkit for detecting, classifying, coordinatizing and visualizing circle bundle structures in data.

When to use this
----------------

Many real-world datasets have hidden **circular or rotational structure** that
standard methods miss. ``circle_bundles`` detects and characterizes this structure
using tools from fiber bundle theory. Example applications include:

- **Image analysis** — natural image patches and optical flow fields organize
  around circles (gradient orientation) and tori (joint orientation/frequency),
  forming Klein bottles and non-trivial circle bundles.
- **Sensor and signal data** — periodic signals with spatially varying phase
  naturally form circle bundles over the parameter space.
- **Molecular and shape data** — conformational flexibility in molecules or
  articulated shapes often introduces circular fiber structure.
- **Dimensionality reduction validation** — when circular coordinates from tools
  like `DREiMaC <https://github.com/scikit-tda/DREiMaC>`_ are available,
  ``circle_bundles`` determines whether those coordinates can be made globally
  consistent, or whether topological obstructions (non-trivial characteristic
  classes) prevent it.

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

   references
