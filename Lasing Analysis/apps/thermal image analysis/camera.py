import os
import perspectivemap as pm
import cv2
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from pylab import *
from rich.progress import Progress
from rich.console import Console
from IPython.display import HTML
from math import radians as rad
from math import degrees as deg



# We want to be able to provide xy coordinates, the angle of the camera, and the angle of the lid? and maximum temperature to provide the expected measured temperature.

# Start with a helper function to account for lambertian emission and distance

# Constants for camera and chip setup. Origin is defuined to be at the center of the lid on which the camrea is mounted. Chip offset constants are taken from old CAD; may be inaccurate.

# Neu Vorrichtung
CHIP_OFFSET_X = 3.675327 # millimeters, x offset of the top right corner of the chip from the origin
CHIP_OFFSET_Y = 14.096069 # millimeters, y offset of the top right corner of the chip from the origin
CHIP_OFFSET_Z = 155.700000 #- 2.1 # millimeters, z offset of the top right corner of the chip from the origin

LID_ANGLE =  rad(-43.8400825)
MOUNT_ANGLE = rad(deg(LID_ANGLE) - 27.8106366)
MOUNT_OFFSET_Y = 19.10335 # millimeters, intersection of the y=axis with the plane in which the camera rotates (CAMERA_ANGLE rotation)
CAMERA_RADIUS = 38.86401 # millimeters, distance from the camera mount to the origin + MOUNT_OFFSET
CAMERA_HEIGHT = 84.5 # millimeters [?,91.11], height of the camera mount (center of rotation)
CAMERA_ANGLE = rad(10) # radians, angle of the camera from the vertical
CAMERA_LENGTH = 73.97719 # millimeters, length of the camera from the point of rotation to the lens

LENS_DIAMETER = 18 # millimeters, diameter of the camera lens
# APERTURE_DIAMETER = 47 # millimeters, diameter of the hole in the lid through which the camera views the sample
UPPER_APERTURE_DIAMETER = 38 # millimeters, diameter of the hole holding the lid through which the camera views the sample
LOWER_APERTURE_DIAMETER = 28 # millimeters, diameter of the hole in the lid through which the camera views the sample
APERTURE_DISTANCE = 29 # millimeters, distance from the origin to the center of the aperture
APERTURE_ANGLE = rad(deg(LID_ANGLE) + 17.4675436) # degrees, angle of the aperture from the y-axis
APERTURE_HEIGHT = 5 # millimeters, height of the aperture above the top of the lid
LID_THICKNESS = 12 # millimeters, thickness of the lid

LOWER_APERTURE_HEIGHT = CHIP_OFFSET_Z - LID_THICKNESS # millimeters, height of the bottom of the aperture above the sample
CENTER_APERTURE_HEIGHT = CHIP_OFFSET_Z
UPPER_APERTURE_HEIGHT = CHIP_OFFSET_Z + APERTURE_HEIGHT # millimeters, height of the top of the aperture above the sample

# EMISSIVITY_PAPER = 0.68
# STEFAN_BOLTZMANN = 5.67e-8 # W/m^2/K^4, Stefan-Boltzmann constant
AMBIENT_TEMPERATURE = 299.15 # Kelvin, ambient temperature

KELVIN_OFFSET = 273.15
WIDTH, HEIGHT, SCALING = pm.get_calibration_dimensions()
CHIP_CORNER_ORIGIN = pm.get_chip_corners()[2]  # Bottom-left corner of the chip in image coordinates

# Create "iron" colormap
IRON_RAW = np.flipud(np.asarray(Image.open("C:/Users/ssuub/Desktop/EDET80k_Damage/Lasing Analysis/apps/thermal image analysis/Iron Color Palette.png")))
IRON = LinearSegmentedColormap.from_list('iron', IRON_RAW / 255)

#=======================================================================================


CO = [CAMERA_RADIUS * np.sin(LID_ANGLE), CAMERA_RADIUS * np.cos(LID_ANGLE), CAMERA_HEIGHT] # Coordinates of the camera origin in the lab coordinate system

LFM = [[np.sin(MOUNT_ANGLE) * np.cos(CAMERA_ANGLE), np.cos(MOUNT_ANGLE) * np.cos(CAMERA_ANGLE), - np.sin(CAMERA_ANGLE)],
       [np.cos(MOUNT_ANGLE), - np.sin(MOUNT_ANGLE), 0],
       [-np.sin(MOUNT_ANGLE) * np.sin(CAMERA_ANGLE), -np.cos(MOUNT_ANGLE) * np.sin(CAMERA_ANGLE), - np.cos(CAMERA_ANGLE)]]

CFM = np.linalg.inv(LFM)


class CameraPosition():
    """Creates a camera position object. Includes information about the position of the camera as well as its line of sight towards the chip.

    Parameters:
        lid_angle (float):
            The angle in radians of the lid relative to the vertical axis.
        camera_angle (float):
            The angle in radians of the camera relative to the vertical axis.
        camera_height (float):
            The height in millimeters of the camera above the sample.
        corners (list):
            The coordinates of the corners of the chip as seen in the image. Coordinates should be in the form [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], where the corners are ordered as follows: top-left, top-right, bottom-right, bottom-left.

    .. note::
        ``lid_angle``, ``camera_angle``, and ``camera_height`` are physical measurements that need to be taken by hand each time the camera is moved.
        ``corners`` should be determined by taking an image of the chip after moving the camera and recording the coordinates of the corners of the chip in the image.\
        

        For an interactive visualization of what the physical measurements should look like, see

        https://www.desmos.com/3d/4agexlbubf,

        
        where the ``lid_angle`` is :math:`\\phi_L`, the ``camera_angle`` is :math:`\\rho_C`, and the ``camera_height`` is :math:`H`.
    """
    
    # Maximum acceptance angle of the camera lens; this value was guessed based on physical measurements, may need to be refined
    _maxRho = np.arcsin(LENS_DIAMETER / (2 * CAMERA_LENGTH))

    def __init__(self, lid_angle:float, camera_angle:float, camera_height:float, corners:list):
        self.lid_angle = lid_angle
        self.camera_angle = camera_angle
        self.camera_height = camera_height
        self.corners = np.float32(corners)

        self.CO = [CAMERA_RADIUS * np.sin(lid_angle), CAMERA_RADIUS * np.cos(lid_angle), camera_height] # Coordinates of the camera origin in the lab coordinate system

        self.LFM = [[np.sin(MOUNT_ANGLE) * np.cos(camera_angle), np.cos(MOUNT_ANGLE) * np.cos(camera_angle), - np.sin(camera_angle)],
               [np.cos(MOUNT_ANGLE), - np.sin(MOUNT_ANGLE), 0],
               [-np.sin(MOUNT_ANGLE) * np.sin(camera_angle), -np.cos(MOUNT_ANGLE) * np.sin(camera_angle), - np.cos(camera_angle)]]
        
        self.CFM = np.linalg.inv(self.LFM)

    def printParameters(self):
        """Prints the parameters of the camera position."""
        print("Lid Angle (deg):", deg(self.lid_angle))
        print("Camera Angle (deg):", deg(self.camera_angle))
        print("Camera Height (mm):", self.camera_height)
    
    def imageTransform(self):
        """Returns a transformation matrix that maps points in the camera image to points in the chip coordinate system.

            The transformation is determined by the position of the corners of the annealing brick in the image, which are provided when the camera position object is created as ``corners``.
            The goal of this transformation is to convert an image taken from an angle to one take from a "top-down" view of the chip.

            Returns:
                np.ndarray: A 3x3 transformation matrix that can be applied to points in the camera image to map them to points in the chip coordinate system.
        """
        return pm.new_image_transform(self.corners)

    def cameraChipVector(self, x, y, rho = 0, phi = 0):
        """Returns the vector from a point on the chip to a point on the camera lens.
        
        Parameters:
            x (float): 
                The x coordinate of the point on the chip in pixels, with the origin at the top-left corner of the chip.
            y (float): 
                The y coordinate of the point on the chip in pixels, with the origin at the top-left corner of the chip.
            rho (float): 
                The angle between the point on the camera lens and the center of the camera lens.
            phi (float): 
                The angle of rotation about the center of the camera lens.

        Returns:
            tuple (float, float, float): A tuple containing the x, y, and z components of the vector from the point on the chip to the point on the camera lens in millimeters.

        .. note::
            For an interactive visualization of what the parameters represent, see

            https://www.desmos.com/3d/4agexlbubf,

            
            where ``rho`` is :math:`\\rho_l` and ``phi`` is :math:`\\phi_l`.
        """
        coordinates = [CAMERA_LENGTH * np.sin(rho) * np.cos(phi),
                       CAMERA_LENGTH * np.sin(rho) * np.sin(phi),
                       CAMERA_LENGTH * np.cos(rho)]
        
        x, y = pm.pixels_to_mm(x, y)
        
        # Coordinates of the camera lens
        cx, cy, cz = np.matmul(coordinates, self.LFM)
        
        cx += self.CO[0]
        cy += self.CO[1]
        cz += self.CO[2]
        
        # Cartesian vector components from the chip to the camera lens
        rx = cx - CHIP_OFFSET_X - x
        ry = cy + CHIP_OFFSET_Y - y
        rz = cz + CHIP_OFFSET_Z

        return rx, ry, rz

    def apertureRadii(self, x, y, rho = 0, phi = 0):
        """Returns the radius at which the camera vector intersects the bottom, center, and top of the aperture.
        This can be compared to the radius of the aperture at the bottom, center, and top to determine if the point on the chip is visible from the camera given the camera position and the position of the aperture.

        Parameters:
            x (float): 
                The x coordinate of the point on the chip in pixels, with the origin at the top-left corner of the chip.
            y (float): 
                The y coordinate of the point on the chip in pixels, with the origin at the top-left corner of the chip.
            rho (float): 
                The angle between the point on the camera lens and the center of the camera lens.
            phi (float): 
                The angle of rotation about the center of the camera lens.

        Returns:
            tuple (float, float, float): A tuple containing the radii at which the camera vector intersects the bottom, center, and top of the aperture in millimeters.

        .. note::
            For an interactive visualization of what the parameters represent, see

            https://www.desmos.com/3d/4agexlbubf,

            
            where ``rho`` is :math:`\\rho_l` and ``phi`` is :math:`\\phi_l`.
        """

        rx, ry, rz = self.cameraChipVector(x, y, rho, phi)

        x, y = pm.pixels_to_mm(x, y)
        
        # Deterimes the angle of the point on the camera to the chip from the vertical
        theta = np.arctan2(np.sqrt(rx**2 + ry**2), rz)

        # Find the x and y coordinates of the center of the aperture from the sample origin
        aperture_x = APERTURE_DISTANCE * np.sin(APERTURE_ANGLE) - CHIP_OFFSET_X - x
        aperture_y = APERTURE_DISTANCE * np.cos(APERTURE_ANGLE) + CHIP_OFFSET_Y - y

        # Calculate the radius (with center at the aperture center) at which the camera vector intersects the bottom and top of the aperture
        radius_u = np.sqrt(((rx * UPPER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * UPPER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
        radius_c = np.sqrt(((rx * CENTER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * CENTER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
        radius_l = np.sqrt(((rx * LOWER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * LOWER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
        
        return radius_l, radius_c, radius_u
    


ORIGINAL_CAMERA_POSITION = CameraPosition(LID_ANGLE, CAMERA_ANGLE, CAMERA_HEIGHT, [[455,327],[1038,311],[474,798],[1034,744]])
"""Original camera position left from the 24/25 work term. Used until August 26, 2025 when investigations into camera position began."""
AUGUST25_CAMERA_POSITION = CameraPosition(LID_ANGLE + np.arcsin(14 / 90), CAMERA_ANGLE, CAMERA_HEIGHT, [[102,94],[326,85],[329,249],[115,272]])
"""Camera position used from August 2025 to January 2026."""
JANUARY26_CAMERA_POSITION = CameraPosition(LID_ANGLE + np.arcsin(14 / 90), CAMERA_ANGLE, CAMERA_HEIGHT, [[48,65],[287,49],[295,227],[65,249]])
"""Camera position used from January 2026 to March 2026."""
MARCH26_CAMERA_POSITION = CameraPosition(LID_ANGLE + np.arcsin(14 / 90), CAMERA_ANGLE, CAMERA_HEIGHT, [[65,65],[304,50],[307,225],[79,248]])
"""Camera position used from March 2026 to April 2026."""
APRIL26_CAMERA_POSITION = CameraPosition(LID_ANGLE + np.arcsin(14 / 90), CAMERA_ANGLE, CAMERA_HEIGHT, [[66,68],[298,46],[316,222],[87,252]])
"""Camera position used from April 2026 to May 2026."""





