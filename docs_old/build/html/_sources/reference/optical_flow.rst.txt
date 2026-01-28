Optical flow utilities
======================

These utilities support loading, preprocessing, and analyzing
optical flow datasets such as Sintel.

Sampling and preprocessing
--------------------------

.. autofunction:: circle_bundles.get_sintel_scene_folders
.. autofunction:: circle_bundles.read_flo
.. autofunction:: circle_bundles.sample_from_frame
.. autofunction:: circle_bundles.preprocess_flow_patches

Analysis helpers
----------------

.. autofunction:: circle_bundles.get_contrast_norms
.. autofunction:: circle_bundles.get_predominant_dirs
.. autofunction:: circle_bundles.get_lifted_predominant_dirs
