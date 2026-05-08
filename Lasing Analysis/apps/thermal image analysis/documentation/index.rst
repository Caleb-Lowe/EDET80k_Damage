.. Thermal Image Analysis documentation master file, created by
   sphinx-quickstart on Thu Apr 30 11:15:09 2026.

Thermal Image Analysis
======================

Welcome to the documentation for the Thermal Image Analysis package! This document provides an overview of the code written during the 25/26 work term for analyzing thermal images taken by the Optris camera.

* For documentation of the code used to simulate chip heating, see ``Simulation and Beamlib/Guide Part 2 Simulation Overview.ipynb``
* For documentation of the code used to control and fire the laser, see ``Simulation and Beamlib/Guide Part 3 LasingLib.ipynb``



.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Python Files:

   camera
   optrisROI
   temperatureimage
   oldstuff



.. admonition:: Notes on Documentation
   :collapsible: closed
   
   The documentation of this codebase was done using Sphinx. If any information is updated or changed, this document can be updated by running
   ``cd ./Desktop/EDET80k_Damage/Lasing Analysis/apps/thermal image analysis``
   followed by
   ``sphinx-build -M html ./documentation ./documentation/_build``
   For more information on how to use sphinx, see sphinx-doc.org