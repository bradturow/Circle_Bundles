Visualization
=============

Bundle objects include convenience visualization methods. Some of these depend on optional
packages (e.g. Plotly, Dash, scikit-learn). If optional dependencies are missing, these
methods may raise an ImportError with installation instructions.

Interactive bundle viewer
-------------------------

.. automethod:: circle_bundles.bundle.BundleResult.show_bundle

Nerve visualizations
--------------------

Cover nerve
^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.show_nerve

Max-trivial subcomplex
^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.show_max_trivial

Circle-layout nerve (cycle graphs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: circle_bundles.bundle.BundleResult.show_circle_nerve

Trivialization comparisons
--------------------------

.. automethod:: circle_bundles.bundle.BundleResult.compare_trivs
