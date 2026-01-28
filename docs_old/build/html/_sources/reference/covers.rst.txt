Covers
======

Covers describe how the **base space** is decomposed into overlapping sets.
They are a required input to :func:`circle_bundles.build_bundle`.

Cover objects store:
- base points
- membership information
- the nerve of the cover
- visualization and summary utilities

Base cover classes
------------------

.. autoclass:: circle_bundles.base_covers.MetricBallCover
   :members:
   :undoc-members:

.. autoclass:: circle_bundles.base_covers.TriangulationStarCover
   :members:
   :undoc-members:

Common cover builders
---------------------

.. autofunction:: circle_bundles.make_s2_fibonacci_star_cover
.. autofunction:: circle_bundles.make_rp2_fibonacci_star_cover

Cover inspection and visualization
----------------------------------

Cover objects provide methods such as:

- ``summarize`` — display basic statistics and diagnostics
- ``show_nerve`` — visualize the nerve complex

See individual class documentation for details.
