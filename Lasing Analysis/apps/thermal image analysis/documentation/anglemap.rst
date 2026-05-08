anglemap.py
===========
.. deprecated:: 2025
   The functions in this file were originally used for analysis of Optris images, in particular for lambertian and occlusion corrections. 
   However, due to the high number of necessary and redundant parameters, these functions have been refactored to use OOP principals.
   Thus, for any new analysis, the ``TemperatureSet``, ``CameraPosition``, and ``ROI`` classes should be used instead of the functions in this file. 

.. seealso::
   :class:`temperatureimage.TemperatureSet <temperatureimage.TemperatureSet>`

   :class:`camera.CameraPosition <camera.CameraPosition>`

   :class:`optrisROI.ROI <optrisROI.ROI>`

Temperature Helper Functions
----------------------------
.. autofunction:: anglemap::cameraChipVector
.. autofunction:: anglemap::apertureRadii

Temperature Calculation Functions
---------------------------------
.. autofunction:: anglemap::pointTemp
.. autofunction:: anglemap::approxTemp
.. autofunction:: anglemap::correctedTemp

Image Functions
---------------
.. autofunction:: anglemap::colorTemp
.. autofunction:: anglemap::getRoi
.. autofunction:: anglemap::onBorder
.. autofunction:: anglemap::inRoi
.. autofunction:: anglemap::getRoiPoly
.. autofunction:: anglemap::onBorderPoly
.. autofunction:: anglemap::inRoiPoly

Holistic Correction Helper Functions
------------------------------------
.. autofunction:: anglemap::imageCorrectRuntimeEstimate

Holistic Correction Function
----------------------------
.. autofunction:: anglemap::imageCorrect
.. autofunction:: anglemap::renderImage
