.. ARTS Localisation documentation master file, created by
   sphinx-quickstart on Thu Aug 13 14:11:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ARTS Localisation's documentation!
=============================================



Workflow
========

In short, the localistion consists of the following steps:

#. Note in which CBs the FRB was detected.
#. Save a snippet for the detection CBs, as well as for all surrounding CBs.
#. Get S/N in all SBs of these CBs.
#. Decide which CBs to use for localisation
#. Create a configuration file for the localisation
#. Run the localisation at coarse resolution
#. Repeat the previous step with different settings until the entire uncertainty region is covered
#. Run the localisation at high resolution

Contents
========

.. toctree::
   :maxdepth: 2

   _modules/install
   _modules/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
