perspectivemap.py
=================
Code from the 24/25(?) work term used to correct for distortions in Optris images.
Currently only corrects for perspective distortion, but ``camera_to_roi`` contains framework to correct for (lens) distortions.
It has yet to be determined whether consideration of barrel, pincushion, and mustache distortion would be valuable / necessary for the project.

.. autofunction:: perspectivemap::get_target_corners
.. autofunction:: perspectivemap::get_chip_corners
.. autofunction:: perspectivemap::get_fbrick_corners
.. autofunction:: perspectivemap::get_calibration_dimensions
.. autofunction:: perspectivemap::new_image_transform
.. autofunction:: perspectivemap::refresh_reference
.. autofunction:: perspectivemap::cv_xy_package
.. autofunction:: perspectivemap::perspective_map_points
.. autofunction:: perspectivemap::image_coords_to_cartesian
.. autofunction:: perspectivemap::to_arbitrary_coords
.. autofunction:: perspectivemap::camera_to_roi
