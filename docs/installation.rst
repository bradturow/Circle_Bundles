Installation
============

``circle_bundles`` requires Python 3.9 or newer.

Basic installation
------------------

Install the package directly from GitHub using ``pip``:

.. code-block:: bash

   pip install git+https://github.com/bradturow/Circle_Bundles.git

Then import the package in Python as:

.. code-block:: python

   import circle_bundles as cb

Optional dependencies
---------------------

The core package installs everything needed for the main pipeline.
For interactive visualization support, install the ``viz`` extra:

.. code-block:: bash

   pip install "circle_bundles[viz] @ git+https://github.com/bradturow/Circle_Bundles.git"

This adds `Plotly <https://plotly.com/python/>`_ and Dash for interactive figures.

Install the complete environment for the tutorial notebooks with:

.. code-block:: bash

   pip install "circle-bundles[notebooks] @ git+https://github.com/bradturow/Circle_Bundles.git"

This includes the persistent-homology packages, a Trimesh triangulation backend,
and the Jupyter execution tools used by the repository's reproducibility checks.

The MPI-Sintel tutorial additionally requires optical-flow table support:

.. code-block:: bash

   pip install "circle-bundles[optical-flow] @ git+https://github.com/bradturow/Circle_Bundles.git"

Download the Sintel optical-flow frames separately under their original license and
set ``MPI_SINTEL_FLOW_DIR`` to the dataset's ``training/flow`` directory.

Developer installation
----------------------

To contribute or run the test suite, clone the repository and install in editable mode
with development dependencies:

.. code-block:: bash

   git clone https://github.com/bradturow/Circle_Bundles.git
   cd Circle_Bundles
   pip install -e ".[dev]"

This installs the test, lint, and distribution-build tools.
Run the test suite with:

.. code-block:: bash

   pytest

Verifying the installation
--------------------------

You can verify that ``circle_bundles`` is installed correctly by running:

.. code-block:: python

   import circle_bundles as cb
   print(cb.__version__)
