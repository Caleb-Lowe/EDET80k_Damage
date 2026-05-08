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

#==================================================================================================
# DEPRECATED CODE; helper functions have been refactored into
# - camera.py for camera/chip geometry (constants and vector calculations)
# - temperatureimage.py for temperature image processing (image correction and rendering)
# - optrisROI.py for ROI definitions
#==================================================================================================


# Constants for camera and chip setup. Origin is defuined to be at the center of the lid on which the camrea is mounted. Chip offset constants are taken from old CAD; may be inaccurate.

# Neu Vorrichtung
CHIP_OFFSET_X = 3.675327 # millimeters, x offset of the top right corner of the chip from the origin
CHIP_OFFSET_Y = 14.096069 # millimeters, y offset of the top right corner of the chip from the origin
CHIP_OFFSET_Z = 155.700000 #- 2.1 # millimeters, z offset of the top right corner of the chip from the origin

LID_ANGLE =  rad(-43.8400825) + np.arcsin(14 / 90)
MOUNT_ANGLE = rad(deg(LID_ANGLE) - 27.8106366)
MOUNT_OFFSET_Y = 19.10335 # millimeters, intersection of the y=axis with the plane in which the camera rotates (CAMERA_ANGLE rotation)
CAMERA_RADIUS = 38.86401 # millimeters, distance from the camera mount to the origin + MOUNT_OFFSET
CAMERA_HEIGHT = 84.5 # millimeters [?,91.11], height of the camera mount (center of rotation)
CAMERA_ANGLE = rad(10) # radians, angle of the camera from the vertical
CAMERA_LENGTH = 73.97719 # millimeters, length of the camera from the point of rotation to the lens

LENS_DIAMETER = 18 # millimeters, diameter of the camera lens
APERTURE_DIAMETER = 47 # millimeters, diameter of the hole in the lid through which the camera views the sample
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

sub_limit = 50

CO = [CAMERA_RADIUS * np.sin(LID_ANGLE), CAMERA_RADIUS * np.cos(LID_ANGLE), CAMERA_HEIGHT] # Coordinates of the camera origin in the lab coordinate system

LFM = [[np.sin(MOUNT_ANGLE) * np.cos(CAMERA_ANGLE), np.cos(MOUNT_ANGLE) * np.cos(CAMERA_ANGLE), - np.sin(CAMERA_ANGLE)],
       [np.cos(MOUNT_ANGLE), - np.sin(MOUNT_ANGLE), 0],
       [-np.sin(MOUNT_ANGLE) * np.sin(CAMERA_ANGLE), -np.cos(MOUNT_ANGLE) * np.sin(CAMERA_ANGLE), - np.cos(CAMERA_ANGLE)]]

CFM = np.linalg.inv(LFM)

#=============================== Temperature Helper Functions ==================================

def cameraChipVector(x, y, rho = 0, phi = 0, view_coords = False):
    """Finds the vector from a point on the chip (defined by x and y coordinates) to a point on the camera lens (defined by rho and phi)"""
    coordinates = [CAMERA_LENGTH * np.sin(rho) * np.cos(phi),
                   CAMERA_LENGTH * np.sin(rho) * np.sin(phi),
                   CAMERA_LENGTH * np.cos(rho)]
    
    # Coordinates of the camera lens
    cx, cy, cz = np.matmul(coordinates, LFM)
    
    cx += CO[0]
    cy += CO[1]
    cz += CO[2]
    
    # Cartesian vector components from the chip to the camera lens
    rx = cx - CHIP_OFFSET_X - x
    ry = cy + CHIP_OFFSET_Y - y
    rz = cz + CHIP_OFFSET_Z

    if view_coords:
        print(rx, ry, rz) 

    return rx, ry, rz

def apertureRadii(x, y, rho = 0, phi = 0, view_coords = False):
    """Determines the distance from the center(s) of the lid aperture(s) to the intersection of the camera chip vector and the aperture(s)'s plane"""
    rx, ry, rz = cameraChipVector(x, y, rho, phi)
    #print(rx, ry, rz)
    
    # Deterimes the angle of the point on the camera to the chip from the vertical
    theta = np.arctan2(np.sqrt(rx**2 + ry**2), rz)

    # Find the x and y coordinates of the center of the aperture from the sample origin
    aperture_x = APERTURE_DISTANCE * np.sin(APERTURE_ANGLE) - CHIP_OFFSET_X - x
    aperture_y = APERTURE_DISTANCE * np.cos(APERTURE_ANGLE) + CHIP_OFFSET_Y - y
    
    if view_coords:
        print("Aperture Center:", aperture_x, aperture_y)

    # Calculate the radius (with center at the aperture center) at which the camera vector intersects the bottom and top of the aperture
    radius_u = np.sqrt(((rx * UPPER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * UPPER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
    radius_c = np.sqrt(((rx * CENTER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * CENTER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
    radius_l = np.sqrt(((rx * LOWER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * LOWER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
    
    return radius_l, radius_c, radius_u

#=========================== Temperature Calculation Functions ===============================

def pointTemp(x, y, maxTemp, rho = 0, phi = 0):
    """Simulates lambertian and occlusion distortion of a temperature measurement for a point on the chip and a single point on the camera lens."""
    rx, ry, rz = cameraChipVector(x, y, rho, phi)
    #print(rx, ry, rz)
    
    # Deterimes the angle of the point on the camera to the chip from the vertical
    theta = np.atan2(np.sqrt(rx**2 + ry**2), rz)

    # Calculate the radius (with center at the aperture center) at which the camera vector intersects the bottom and top of the aperture
    radius_l, radius_c, radius_u = apertureRadii(x, y, rho, phi)
    #print("Radius Lower:", radius_l, "Radius Upper:", radius_u)
    
    # Calculate the temperature based on the angle
    if (radius_l > 28 / 2) or (radius_c > 28 / 2) or (radius_u > 38 / 2):
        rectifiedTemp = AMBIENT_TEMPERATURE  # If the camera vector is blocked by the lid, return ambient temperature
    else:
        rectifiedTemp = ((maxTemp ** 4) * np.cos(theta)) ** 0.25
    return rectifiedTemp

def approxTemp(x, y, maxTemp, resolution = 20):
    """Simulates lambertian and occlusion distortion of a temperature measurement for a point on the chip over the entire camera lens.
    Resolution determines how many points on the camera lens will be used for the approximation.
    
    .. Note::
        Point selection on the camera lens uses an algorithm that scales with resolution cubed (rather than squared), and biases towards outer points."""
    maxRho = np.arcsin(LENS_DIAMETER / (2 * CAMERA_LENGTH))
    for i in range(resolution):
        rho = maxRho * i / resolution
        for j in range((i + 1)**2):
            phi = 2 * np.pi * (j + 0.5) / (i + 1)**2
            if i == 0 and j == 0:
                netTemp = pointTemp(x, y, maxTemp, rho, phi)
            else:
                netTemp += pointTemp(x, y, maxTemp, rho, phi)
    avgTemp = netTemp * 6 / ((2 * resolution + 1) * (resolution + 1) * (resolution))
    return avgTemp

def correctedTemp(x, y, temperature, resolution = 2, lambertian = True, occlusion = True, quadratic = True):
    """Gives an approximate correction (lambertian and/or occlusion) for a temperature measurement for a point on the chip over the entire camera lens."""
    # Return early if temperature is at ambient; assume comes from extraneous points in generation of warped image
    if temperature <= KELVIN_OFFSET:
        return temperature

    maxRho = np.arcsin(LENS_DIAMETER / (2 * CAMERA_LENGTH))
    if quadratic:
        numPoints = resolution ** 2
        deltaPoints = lambda x : 2 * x - 1
    else:
        numPoints = (2 * resolution + 1) * (resolution + 1) * (resolution) / 6
        deltaPoints = lambda x : (x + 1)**2

    interimTemp = temperature

    if occlusion:
        # count how many rays are occluded given the resolution
        occludedCounter = 0
        for i in range(resolution):
            rho = maxRho * i / resolution
            for j in range(deltaPoints(i)):
                phi = 2 * np.pi * (j + 0.5) / deltaPoints(i)
                radius_l, radius_c, radius_u = apertureRadii(x, y, rho, phi)
                if (radius_l > 28 / 2) or (radius_c > 28 / 2) or (radius_u > 38 / 2):
                    occludedCounter += 1
        
        # Intermediate calculation of temperature (accounting for occlusion but not Lambertian emission)
        if occludedCounter >= numPoints:
            return temperature  # All rays occluded; return measured temperature
        elif occludedCounter == 0:
            interimTemp = temperature
        else:
            interimTemp = (temperature * numPoints - AMBIENT_TEMPERATURE * occludedCounter) / (numPoints - occludedCounter)

    if lambertian:
        # Deterimes the angle of the point on the camera to the chip from the vertical
        rx, ry, rz = cameraChipVector(x, y)
        theta = np.atan2(np.sqrt(rx**2 + ry**2), rz)
        
        trueTemp = (interimTemp ** 4 / np.cos(theta)) ** 0.25
        return trueTemp
    else:
        return interimTemp

#==================================== Image Functions ========================================

def colorTemp(temperature, min, max, cmap = 'magma'):
    """Returns an RGB tuple corresponding to the temperature value"""
    progress = (temperature - min) / (max - min)
    if cmap != IRON:
        cmap = matplotlib.colormaps.get_cmap(cmap)
    color_val = cmap(int(progress * 255))
    return (int(color_val[0] * 255), int(color_val[1] * 255), int(color_val[2] * 255))

def getRoi(roi):
    """Returns the corners of a rectangular ROI. ``roi`` must be either a tuple of at least four xy-coordinates or one of 'chip', 'brick', and 'fullbrick'"""
    try:
        if roi == "chip":
            chip = pm.get_chip_corners()
            return chip[0], chip[2]
        elif roi == "brick":
            target = pm.get_target_corners()
            return target[0], target[2]
        elif roi == "fullbrick":
            fbrick = pm.get_fbrick_corners()
            return fbrick[0], fbrick[2]
        else:
            return roi[0], roi[2]
    except:
        raise ValueError("Invalid ROI specified. Use 'chip', 'brick', 'fullbrick', or provide custom coordinates.")

def onBorder(x, y, top_left, bottom_right):
    """Returns true if a point (defined with x and y coordinates) lies on the perimeter of a rectangle (defined by its top_left and bottom_right corners)"""
    if (y == top_left[0] or y == bottom_right[0]) and x >= top_left[1] and x <= bottom_right[1]:
        return True
    elif (x == top_left[1] or x == bottom_right[1]) and y >= top_left[0] and y <= bottom_right[0]:
        return True
    return False

def inRoi(x, y, top_left, bottom_right):
    """Returns true if a point (defined with x and y coordinates) lies within a rectangle (defined by its top_left and bottom_right corners)"""
    if x >= top_left[1] and x <= bottom_right[1] and y >= top_left[0] and y <= bottom_right[0]:
        return True
    return False

def getRoiPoly(roi):
    """Returns all vertices of a polygonal ROI. ``roi`` must be either a tuple of xy-coordinates or one of 'chip', 'brick', and 'fullbrick"""
    try:
        if roi == "chip":
            chip = pm.get_chip_corners()
            return chip
        elif roi == "brick":
            target = pm.get_target_corners()
            return target
        elif roi == "fullbrick":
            fbrick = pm.get_fbrick_corners()
            return fbrick
        else:
            return roi
    except:
        raise ValueError("Invalid ROI specified. Use 'chip', 'brick', 'fullbrick', or provide custom coordinates.")

def onBorderPoly(x, y, roi):
    """Returns true if a point (defined with x and y coordinates) lies on the perimeter of a polygon"""
    for i in range(len(roi)):
        next_i = (i + 1) % len(roi)
        # Vertical line case
        if (roi[i][1] == roi[next_i][1]):
            if (x == roi[i][1]) and (y >= min(roi[i][0], roi[next_i][0])) and (y <= max(roi[i][0], roi[next_i][0])):
                return True
        # Horizontal line case
        elif (roi[i][0] == roi[next_i][0]):
            if (y == roi[i][0]) and (x >= min(roi[i][1], roi[next_i][1])) and (x <= max(roi[i][1], roi[next_i][1])):
                return True
        # Diagonal line case
        else:
            # Define a line between two points and check if (x,y) is on that line segment
            linex = lambda x: (roi[next_i][0] - roi[i][0]) / (roi[next_i][1] - roi[i][1]) * (x - roi[i][1]) + roi[i][0]
            liney = lambda y: (roi[next_i][1] - roi[i][1]) / (roi[next_i][0] - roi[i][0]) * (y - roi[i][0]) + roi[i][1]
            if (x >= min(roi[i][1], roi[next_i][1])) and (x <= max(roi[i][1], roi[next_i][1])):
                if np.isclose(y, linex(x), atol=0.5):
                    return True
            if (y >= min(roi[i][0], roi[next_i][0])) and (y <= max(roi[i][0], roi[next_i][0])):
                if np.isclose(x, liney(y), atol=0.5):
                    return True
    return False

def inRoiPoly(x, y, roi):
    """Returns true if a point (defined with x and y coordinates) lies within a polygon"""
    # Ray-casting algorithm to determine if point is in polygon
    inside = False
    n = len(roi)
    p1y, p1x = roi[0]
    for i in range(n + 1):
        p2y, p2x = roi[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
#========================== Holistic Correction Helper Functions ==============================

def imageCorrectRuntimeEstimate(roi, correction_resolution, lambertian, occlusion, quadratic, cmap):
    """Gives an approximate runtime for temperature correction. 
    Runtime scales linearly with number of pixels (temperatures) and quadratically (or sometimes cubically) with the correction_resolution."""
    # Determine ROI coordinates
    if roi != None:
        top_left, bottom_right = getRoi(roi)

    # Estimate runtime
    if cmap == IRON:
        BASE_TIME = 4.0 # runtime for lambertian, occlusion = False
    else:
        BASE_TIME = 22.9 # runtime for lambertian, occlusion = False, magma colormap

    est_runtime = BASE_TIME # runtime in seconds if lambertian calculations are not performed
    if lambertian:
        est_runtime += 2.2 # additional time for lambertian correction

    if occlusion:
        runtime_factor = 1.0
        # Runtime factor seems to be linear when limiting to a ROI
        if roi != None:
            runtime_factor = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1]) / 44826
            
        if quadratic:
            est_runtime += runtime_factor * (0.81944085 * (correction_resolution ** 2) - 1.63533937 * correction_resolution)
        else:
            runtime_factor *= (3618.1 - BASE_TIME) / (20 * 21 * 41 / 6) # empirical factor based on runtime tests
            est_runtime += runtime_factor * correction_resolution * (correction_resolution + 1) * (2 * correction_resolution + 1) / 6

        
    print(f"Estimated Runtime: {int(est_runtime // 3600)}:{int(est_runtime % 3600 // 60):02d}:{int(est_runtime % 60 // 1):02d}")
    return est_runtime

#============================== Holistic Correction Function ==================================

def imageCorrect(filename, title = None, mintemp = None, maxtemp = None, roi = None, lambertian = True, occlusion = True, correction_resolution = 2, quadratic = True, render_image = True, cmap = 'iron'):
    """Corrects (lambertian and occlusion) an Optris image and returns the result as an array."""
    # Determine ROI coordinates
    if roi != None:
        top_left, bottom_right = getRoi(roi)
    if cmap == 'iron':
        cmap = IRON
    
    imageCorrectRuntimeEstimate(roi, correction_resolution, lambertian, occlusion, quadratic, cmap)

    # Load raw data from CSV
    with open(filename, newline='') as csvfile:raw = np.array(list(csv.reader(csvfile, delimiter=';')))

    # Format data into floats, then replace with intensity
    intensity_data = np.zeros((len(raw),len(raw[0])), dtype = np.float64)

    for y in range(len(intensity_data) - 1):
        for x in range(len(intensity_data[0]) - 1):
            intensity = (float(raw[y,x].replace(',', '.')) + KELVIN_OFFSET) ** 4
            intensity_data[y,x] = intensity

    

    # Correct intensity values for perspective
    transform = pm.image_transform
    with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
        task = progress.add_task("Correcting Image Distortion...", total=1)
        spacial_corrected = cv2.warpPerspective(
            intensity_data, transform, (len(raw[0]), len(raw)), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) ** 0.25
        progress.update(task, advance=1)

    # Account for Lambertian Emission and Occlusion
    with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
        if roi != None:
            problem_space = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
        else:
            problem_space = len(spacial_corrected) * (len(spacial_corrected[0])) 
        task = progress.add_task("Correcting for Angle Dependence...", total = problem_space)
        for y in range(len(spacial_corrected) - 1):
            for x in range(len(spacial_corrected[0]) - 1):
                if roi == None or inRoi(y, x, top_left, bottom_right):
                    xn = (x - CHIP_CORNER_ORIGIN[0]) / SCALING
                    yn = (CHIP_CORNER_ORIGIN[1] - y) / SCALING
                    measured_temp = (spacial_corrected[y,x])
                    true_temp = correctedTemp(xn, yn, measured_temp, correction_resolution, lambertian, occlusion, quadratic)
                    spacial_corrected[y,x] = true_temp
                    progress.update(task, advance=1)


    # Determine min and max temperatures for color mapping
    if mintemp == None:
        mintemp = (np.min(spacial_corrected[spacial_corrected > KELVIN_OFFSET]) - KELVIN_OFFSET)
    if maxtemp == None:
        maxtemp = (np.max(spacial_corrected)  - KELVIN_OFFSET)
    print(f"Temperature Range: {mintemp:.2f} °C to {maxtemp:.2f} °C")

    if render_image:
        # Map to color scale
        with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
            task = progress.add_task("Mapping to Color Scale...", total = len(spacial_corrected) * (len(spacial_corrected[0]) - 1))
            imageData = np.zeros((len(spacial_corrected), len(spacial_corrected[0]), 3), dtype = np.uint8)

            for y in range(len(spacial_corrected) - 1):
                for x in range(len(spacial_corrected[0]) - 1):
                    if roi != None and onBorder(y, x, top_left, bottom_right):
                        imageData[y,x] = (255,255,255)  # Highlight ROI in white
                    else:
                        imageData[y,x] = colorTemp((spacial_corrected[y,x]) - KELVIN_OFFSET, mintemp, maxtemp, cmap)
                    progress.update(task, advance=1)
        
        plt.close("Corrected Thermal Image")

        img = Image.fromarray(imageData)

        fig, ax = plt.subplots(1, 1, num = "Corrected Thermal Image")

        if title != None:
            plt.title(title)
        
        imgplot = ax.imshow(img, vmin = mintemp, vmax = maxtemp, cmap = cmap)
        ax.set_axis_off()

        plt.axis('off')
        plt.colorbar(imgplot, ax = ax, label = 'Temperature (°C)')
        plt.show()

    return spacial_corrected - KELVIN_OFFSET


def renderImage(filename, title = None, mintemp = None, maxtemp = None, roi = None, cmap = 'iron'):
    """Given an array of temperatures, renders an image. Also renders any roi on top of the image using white lines to represent the perimeter."""
    roi = getRoiPoly(roi)
    if cmap == 'iron':
        cmap = IRON
    
    if isinstance(filename, str):
        # Load raw data from CSV
        with open(filename, newline='') as csvfile:
            raw = np.array(list(csv.reader(csvfile, delimiter=';')))

        temperature_data = np.zeros((len(raw), len(raw[0]) - 1), dtype = np.float64)

        for x in range(len(raw)):
            for y in range(len(raw[0]) - 1):
                temperature_data[x,y] = float(raw[x,y].replace(',', '.'))
    else:
        temperature_data = filename

    if mintemp == None:
        mintemp = np.min(temperature_data)
    if maxtemp == None:
        maxtemp = np.max(temperature_data)

    imageData = np.zeros((len(temperature_data), len(temperature_data[0]), 3), dtype = np.uint8)
    for x in range(len(temperature_data)):
            for y in range(len(temperature_data[0])):
                if roi is not None and onBorderPoly(x, y, roi):
                    imageData[x,y] = (255,255,255)  # Highlight ROI in white
                # elif roi is not None and inRoiPoly(x, y, roi):
                #     imageData[x,y] = (0,200,0)
                else:
                    imageData[x,y] = colorTemp((temperature_data[x,y]), mintemp, maxtemp, cmap)

    plt.close("Thermal Image")

    img = Image.fromarray(imageData)

    fig, ax = plt.subplots(1, 1, num = "Thermal Image")

    if title != None:
        plt.title(title)
    
    imgplot = ax.imshow(img, vmin = mintemp, vmax = maxtemp, cmap = cmap)
    ax.set_axis_off()

    plt.axis('off')
    plt.colorbar(imgplot, ax = ax, label = 'Temperature (°C)')
    plt.show()