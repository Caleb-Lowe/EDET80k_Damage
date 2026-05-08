Temperature Fitting and Modelling
=================================
.. autoclass:: temperatureimage::TemperatureModel
   :members:

.. important::
   The following classes and associated methods have not been thorughly tested and may not work as expected. Further work is necessary before extensive use.

   It may be easier to use these classes as a guideline for implementing your own fitting and modelling functions rather than using them directly.

.. note::
   The fit classes exist to support a proposed method of calibration of the camera. 
   The idea is to take multiple thermal images at various known temperatures, fit a function to each image (to allow for intrapolation between coordinates), 
   and then fit a superfunction to each image function (to allow for interpolation between temperatures).

   As of April 2026, a different method of calibration is currently being explored (filming the heating of an aluminum brick). Thus, these classes may not be used, but they are still included in the documentation for reference and potential future use.

.. autoclass:: temperatureimage::TemperatureFit
   :members:

.. autoclass:: temperatureimage::TemperaturePolyFit
   :members:

.. autoclass:: temperatureimage::SuperPolyFit
   :members: