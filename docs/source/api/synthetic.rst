Synthetic data
==============

Densities
---------

.. autofunction:: circle_bundles.mesh_to_density
.. autofunction:: circle_bundles.get_density_axes
.. autofunction:: circle_bundles.rotate_density
.. autofunction:: circle_bundles.get_mesh_sample

Meshes + mesh visualization
---------------------------

.. autofunction:: circle_bundles.make_tri_prism
.. autofunction:: circle_bundles.make_star_pyramid
.. autofunction:: circle_bundles.make_density_visualizer
.. autofunction:: circle_bundles.make_tri_prism_visualizer
.. autofunction:: circle_bundles.make_star_pyramid_visualizer
.. autofunction:: circle_bundles.make_rotating_mesh_clip

Natural image patches
---------------------

.. autofunction:: circle_bundles.sample_nat_img_kb
.. autofunction:: circle_bundles.get_gradient_dirs

Optical-flow patch models
-------------------------

.. autofunction:: circle_bundles.sample_opt_flow_torus
.. autofunction:: circle_bundles.make_flow_patches

S2 / Hopf / unit tangent models
-------------------------------

.. autofunction:: circle_bundles.sample_sphere
.. autofunction:: circle_bundles.hopf_projection
.. autofunction:: circle_bundles.spin3_adjoint_to_so3
.. autofunction:: circle_bundles.so3_to_s2_projection
.. autofunction:: circle_bundles.sample_s2_trivial
.. autofunction:: circle_bundles.tangent_frame_on_s2
.. autofunction:: circle_bundles.sample_s2_unit_tangent

SO(3) sampling
--------------

.. autofunction:: circle_bundles.sample_so3
.. autofunction:: circle_bundles.project_o3

Step-edge patches
-----------------

.. autofunction:: circle_bundles.get_patch_types_list
.. autofunction:: circle_bundles.make_step_edges
.. autofunction:: circle_bundles.make_all_step_edges
.. autofunction:: circle_bundles.sample_binary_step_edges
.. autofunction:: circle_bundles.mean_center
.. autofunction:: circle_bundles.sample_step_edge_torus

Tori / Klein bottle helpers
---------------------------

.. autofunction:: circle_bundles.const
.. autofunction:: circle_bundles.small_to_big
.. autofunction:: circle_bundles.sample_C2_torus
