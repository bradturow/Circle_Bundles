Core pipeline
=============

The main entry point is :class:`circle_bundles.Bundle`.

Typical workflow
----------------

.. code-block:: python

   import circle_bundles as cb

   bundle = cb.Bundle(X, U=U, pou=pou, landmarks=landmarks)
   lt = bundle.get_local_trivs()
   cp = bundle.get_classes()
   bm = bundle.get_bundle_map()

   bundle.summary()

Reference
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   circle_bundles.Bundle
