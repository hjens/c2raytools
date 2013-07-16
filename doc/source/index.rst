.. c2raytools documentation master file, created by
   sphinx-quickstart on Tue Jul 16 11:08:35 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to c2raytools's documentation!
======================================

About c2raytools
----------------
:mod:`c2raytools` is a Python module for reading and analyzing data files produced by C2Ray and CubeP3M. In addition to providing a Python interface for reading data and converting to physically meaningful units, it provides functions for calculating derived quantities such as brightness temperature and statistics such as power spectra. There are also functions for converting data to redshift space and to produce lightcone boxes.

Installation
------------
First, download the necessary files from `GitHub <https://github.com/hjens/c2raytools>`_. You can do this easily by running the command:

>>> git clone https://github.com/hjens/c2raytools

Then ``cd`` into the newly created directory run the :mod:`setup.py` script to install it:

>>> python setup.py install

This will install the package to your default Python directory. You may have to add a ``sudo`` before the command. If you do not have write permission to this directory, you can install :mod:`c2raytools` to some custom directory as such:

>>> python setup.py install --home=~/mydir

In this case, you have to make sure the directory you are installing to is in your PYTHONPATH.

Updating to the latest version
------------------------------
All updates to :mod:`c2raytools` are published on GitHub. To update your installation to the latest version, first ``cd`` into the directory where you first downloaded :mod:`c2raytools`, and run:

>>> git pull origin

Then run the :mod:`setup.py` scipt again, in the same way as before. If you deleted the downloaded files, you can update by simply repeating the installation procedure as described above.

Dependencies
------------
:mod:`c2raytools` requires :mod:`numpy` to work. To visualize data, :mod:`matplotlib` is recommended.

.. _documentation:

Documentation
--------------

Contents:

.. toctree::
    :maxdepth: 2

    tutorial
    reading_files
    analysis
    constants
    utilities



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

