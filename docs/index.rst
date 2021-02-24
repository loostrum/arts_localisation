.. ARTS Localisation documentation master file, created by
   sphinx-quickstart on Thu Aug 13 14:11:37 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ARTS Localisation
=============================================
This project contains all necessary tools to localise transient bursts discovered with the Apertif Radio Transient
System (ARTS). The localisation uses a chi squared method, comparing the S/N of a detected burst across beams to a model
of the telescope response. The beam model includes a simulation of the ARTS tied-array beamformer and the
generation of synthesised beams (SBs). The beam model and localisation method are described in more detail in Leon Oostrum's
PhD thesis: `Fast Radio Bursts with Apertif <http://hdl.handle.net/11245.1/abe5c8fa-1fdf-490b-ac0d-61e946f5791f>`_.

Contents
========

.. toctree::
   :maxdepth: 1

   _modules/install
   _modules/tutorial
   _modules/api
