Covers
======

Cover classes
-------------

.. currentmodule:: circle_bundles.base_covers

.. autoclass:: MetricBallCover
   :members: build, ensure_built, nerve_edges, nerve_triangles, nerve_tetrahedra, show_nerve, summarize
   :undoc-members: False

.. autoclass:: TriangulationStarCover
   :members: build, ensure_built, nerve_edges, nerve_triangles, nerve_tetrahedra, show_nerve, summarize
   :undoc-members: False


Cover builder helpers
---------------------

.. currentmodule:: circle_bundles.covers.metric_ball_cover_builders

.. autoclass:: S2GeodesicMetric
.. autoclass:: RP2GeodesicMetric

.. currentmodule:: circle_bundles.covers.triangle_cover_builders_fibonacci

.. autofunction:: make_s2_fibonacci_star_cover
.. autofunction:: make_rp2_fibonacci_star_cover
