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

This adds `Plotly <https://plotly.com/python/>`_ for interactive 3D plots.

Some tutorials also use the following packages, which can be installed separately:

.. code-block:: bash

   pip install ripser persim dreimac

Developer installation
----------------------

To contribute or run the test suite, clone the repository and install in editable mode
with development dependencies:

.. code-block:: bash

   git clone https://github.com/bradturow/Circle_Bundles.git
   cd Circle_Bundles
   pip install -e ".[dev]"

This installs ``pytest`` and ``ruff`` for testing and linting.
Run the test suite with:

.. code-block:: bash

   pytest

Verifying the installation
--------------------------

You can verify that ``circle_bundles`` is installed correctly by running:

.. code-block:: python

   import circle_bundles as cb
   print(cb.__version__)
