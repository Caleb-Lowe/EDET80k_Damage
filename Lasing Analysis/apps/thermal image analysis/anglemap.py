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


# We want to be able to provide xy coordinates, the angle of the camera, and the angle of the lid? and maximum temeprature to provide the expected measured temperature.

# Start with a helper function to account for lambertian emission and distance

# Constants for camera and chip setup. Origin is defuined to be at the center of the lid on which the camrea is mounted. Chip offset constants are taken from old CAD; may be inaccurate.

# Neu Vorrichtung
CHIP_OFFSET_X = 3.675327 # millimeters, x offset of the top right corner of the chip from the origin
CHIP_OFFSET_Y = 14.096069 # millimeters, y offset of the top right corner of the chip from the origin
CHIP_OFFSET_Z = 155.700000 # millimeters, z offset of the top right corner of the chip from the origin

LID_ANGLE =  rad(-43.8400825)
MOUNT_ANGLE = rad(deg(LID_ANGLE) - 27.8106366)
MOUNT_OFFSET_Y = 19.10335 # millimeters, intersection of the y=axis with the plane in which the camera rotates (CAMERA_ANGLE rotation)
CAMERA_RADIUS = 38.86401 # millimeters, distance from the camera mount to the origin + MOUNT_OFFSET
CAMERA_HEIGHT = 91.11 # millimeters, height of the camera mount (center of rotation)
CAMERA_ANGLE = rad(22.5) # radians, angle of the camera from the vertical
CAMERA_LENGTH = 73.97719 # millimeters, length of the camera from the point of rotation to the lens

LENS_DIAMETER = 18 # millimeters, diameter of the camera lens
APERTURE_DIAMETER = 47 # millimeters, diameter of the hole in the lid through which the camera views the sample
APERTURE_DISTANCE = 29 # millimeters, distance from the origin to the center of the aperture
APERTURE_ANGLE = rad(deg(LID_ANGLE) + 17.4675436) # degrees, angle of the aperture from the y-axis
APERTURE_HEIGHT = 5 # millimeters, height of the aperture above the top of the lid
LID_THICKNESS = 12 # millimeters, thickness of the lid

LOWER_APERTURE_HEIGHT = CHIP_OFFSET_Z - LID_THICKNESS # millimeters, height of the bottom of the aperture above the sample
UPPER_APERTURE_HEIGHT = CHIP_OFFSET_Z + APERTURE_HEIGHT # millimeters, height of the top of the aperture above the sample

# EMISSIVITY_PAPER = 0.68
# STEFAN_BOLTZMANN = 5.67e-8 # W/m^2/K^4, Stefan-Boltzmann constant
AMBIENT_TEMPERATURE = 299.15 # Kelvin, ambient temperature

IMDIMS = (400,300)
KELVIN_OFFSET = 273.15
WIDTH, HEIGHT, SCALING = pm.get_calibration_dimensions()

# Create "iron" colormap
IRON_RAW = np.flipud(np.asarray(Image.open("C:/Users/ssuub/Desktop/EDET80k_Damage/Lasing Analysis/apps/thermal image analysis/Iron Color Palette.png")))
IRON = LinearSegmentedColormap.from_list('iron', IRON_RAW / 255)

#=======================================================================================

sub_limit = 50

CO = [CAMERA_RADIUS * np.sin(LID_ANGLE), CAMERA_RADIUS * np.cos(LID_ANGLE), CAMERA_HEIGHT]

LFM= [[-np.sin(MOUNT_ANGLE) * np.cos(CAMERA_ANGLE), -np.cos(MOUNT_ANGLE) * np.cos(CAMERA_ANGLE), np.sin(CAMERA_ANGLE)],
                       [np.cos(MOUNT_ANGLE), - np.sin(MOUNT_ANGLE), 0],
                       [-np.sin(MOUNT_ANGLE) * np.sin(CAMERA_ANGLE), -np.cos(MOUNT_ANGLE) * np.sin(CAMERA_ANGLE), - np.cos(CAMERA_ANGLE)]]

CFM = np.linalg.inv(LFM)

#=============================== Temperature Helper Functions ==================================

def cameraChipVector(x, y, rho = 0, phi = 0, view_coords = False):
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

def apertureRadii(x, y, rho = 0, phi = 0):
    rx, ry, rz = cameraChipVector(x, y, rho, phi)
    #print(rx, ry, rz)
    
    # Deterimes the angle of the point on the camera to the chip from the vertical
    theta = np.arctan2(np.sqrt(rx**2 + ry**2), rz)

    # Find the x and y coordinates of the center of the aperture from the sample origin
    aperture_x = APERTURE_DISTANCE * np.sin(APERTURE_ANGLE) - CHIP_OFFSET_X - x
    aperture_y = APERTURE_DISTANCE * np.cos(APERTURE_ANGLE) + CHIP_OFFSET_Y - y
    # print("Aperture Center:", aperture_x, aperture_y)

    # Calculate the radius (with center at the aperture center) at which the camera vector intersects the bottom and top of the aperture
    radius_u = np.sqrt(((rx * UPPER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * UPPER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
    radius_l = np.sqrt(((rx * LOWER_APERTURE_HEIGHT / rz) - aperture_x) ** 2 + ((ry * LOWER_APERTURE_HEIGHT / rz) - aperture_y) ** 2)
    
    return radius_l, radius_u

#=========================== Temperature Calculation Functions ===============================

def pointTemp(x, y, maxTemp, rho = 0, phi = 0):
    rx, ry, rz = cameraChipVector(x, y, rho, phi)
    #print(rx, ry, rz)
    
    # Deterimes the angle of the point on the camera to the chip from the vertical
    theta = np.atan2(np.sqrt(rx**2 + ry**2), rz)

    # Calculate the radius (with center at the aperture center) at which the camera vector intersects the bottom and top of the aperture
    radius_l, radius_u = apertureRadii(x, y, rho, phi)
    #print("Radius Lower:", radius_l, "Radius Upper:", radius_u)
    
    # Calculate the temperature based on the angle
    if (radius_l > APERTURE_DIAMETER / 2) or (radius_u > APERTURE_DIAMETER / 2):
        rectifiedTemp = AMBIENT_TEMPERATURE  # If the camera vector is blocked by the lid, return ambient temperature
    else:
        rectifiedTemp = ((maxTemp ** 4) * np.cos(theta)) ** 0.25
    return rectifiedTemp

def approxTemp(x, y, maxTemp, resolution = 20):
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

def correctedTemp(x, y, temperature, resolution = 2, lambertian = True, occlusion = True, quadratic = False):

    # Return early if temperature is at ambient; assume comes from extraneous points in generation of warped image
    if temperature <= KELVIN_OFFSET:
        return temperature

    maxRho = np.arcsin(LENS_DIAMETER / (2 * CAMERA_LENGTH))
    if quadratic:
        numPoints = resolution * 2
    else:
        numPoints = (2 * resolution + 1) * (resolution + 1) * (resolution) / 6

    interimTemp = temperature

    if occlusion:
        # count how many rays are occluded given the resolution
        occludedCounter = 0
        for i in range(resolution):
            rho = maxRho * i / resolution
            if quadratic:
                jrange = range(2 * i - 1)
            else:
                jrange = range((i + 1)**2)
            for j in jrange:
                if quadratic:
                    phi = 2 * np.pi * (j + 0.5) / (2 * i - 1)
                else:
                    phi = 2 * np.pi * (j + 0.5) / (i + 1)**2
                radius_l, radius_u = apertureRadii(x, y, rho, phi)
                if (radius_l > APERTURE_DIAMETER / 2) or (radius_u > APERTURE_DIAMETER / 2):
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
    # Returns an RGB tuple corresponding to the temperature value
    progress = (temperature - min) / (max - min)
    if cmap != IRON:
        cmap = matplotlib.colormaps.get_cmap(cmap)
    color_val = cmap(int(progress * 255))
    return (int(color_val[0] * 255), int(color_val[1] * 255), int(color_val[2] * 255))

def getRoi(roi):
    if roi == "chip":
        chip = pm.get_chip_corners()
        return chip[0], chip[3]
    elif roi == "brick":
        target = pm.get_target_corners()
        return target[0], target[3]
    elif roi == "fullbrick":
        fbrick = pm.get_fbrick_corners()
        return fbrick[0], fbrick[3]
    else:
        return roi[0], roi[1]

def onBorder(x, y, top_left, bottom_right):
    if (y == top_left[0] or y == bottom_right[0]) and x >= top_left[1] and x <= bottom_right[1]:
        return True
    elif (x == top_left[1] or x == bottom_right[1]) and y >= top_left[0] and y <= bottom_right[0]:
        return True
    return False

def inRoi(x, y, top_left, bottom_right):
    if x >= top_left[1] and x <= bottom_right[1] and y >= top_left[0] and y <= bottom_right[0]:
        return True
    return False


#========================== Holistic Correction Helper Functions ==============================

def imageCorrectRuntimeEstimate(roi, correction_resolution, lambertian, occlusion):
    # Determine ROI coordinates
    if roi != None:
        top_left, bottom_right = getRoi(roi)

    # Estimate runtime
    BASE_TIME = 22.9 # runtime for lambertian, occlusion = False
    est_runtime = BASE_TIME # runtime in seconds if lambertian calculations are not performed
    if lambertian:
        est_runtime += 2.2 # additional time for lambertian correction
    runtime_factor = (42128.1 - BASE_TIME) / (40 * 41 * 81 / 6) # empirical factor based on runtime tests

    # Runtime factor seems to be linear when limiting to a ROI
    if roi != None:
        runtime_factor = (3618.1 - BASE_TIME) / (20 * 21 * 41 / 6) # empirical factor based on runtime tests
        runtime_factor *= (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1]) / 69750
    if occlusion:
        est_runtime += runtime_factor * correction_resolution * (correction_resolution + 1) * (2 * correction_resolution + 1) / 6
    print(f"Estimated Runtime: {int(est_runtime // 3600)}:{int(est_runtime % 3600 // 60):02d}:{int(est_runtime % 60 // 1):02d}")
    return est_runtime

#============================== Holistic Correction Function ==================================

def imageCorrect(filename, title = None, mintemp = None, maxtemp = None, roi = None, lambertian = True, occlusion = True, correction_resolution = 2, cmap = 'iron'):
    # Determine ROI coordinates
    if roi != None:
        top_left, bottom_right = getRoi(roi)
    if cmap == 'iron':
        cmap = IRON
    
    imageCorrectRuntimeEstimate(roi, correction_resolution, lambertian, occlusion)

    # Load raw data from CSV
    with open(filename, newline='') as csvfile:raw = np.array(list(csv.reader(csvfile, delimiter=';')))

    # Format data into floats, then replace with intensity
    intensity_data = np.zeros((len(raw), len(raw[0])), dtype = np.float64)

    for x in range(len(raw)):
        for y in range(len(raw[0]) - 1):
            intensity_data[x,y] = (float(raw[x,y].replace(',', '.')) + KELVIN_OFFSET) ** 4

    

    # Correct intensity values for perspective
    transform = pm.image_transform
    with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
        task = progress.add_task("Correcting Image Distortion...", total=1)
        spacial_corrected = cv2.warpPerspective(
            intensity_data, transform, IMDIMS, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) ** 0.25
        progress.update(task, advance=1)

    # Account for Lambertian Emission and Occlusion
    with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
        if roi != None:
            problem_space = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
        else:
            problem_space = len(spacial_corrected) * (len(spacial_corrected[0]) - 1) 
        task = progress.add_task("Correcting for Angle Dependence...", total = problem_space)
        for x in range(len(spacial_corrected)):
            for y in range(len(spacial_corrected[0]) - 1):
                if roi == None or inRoi(x, y, top_left, bottom_right):
                    measured_temp = (spacial_corrected[x,y])
                    true_temp = correctedTemp(x / SCALING, y / SCALING, measured_temp, correction_resolution, lambertian, occlusion)
                    spacial_corrected[x,y] = true_temp
                    progress.update(task, advance=1)


    # Determine min and max temperatures for color mapping
    if mintemp == None:
        mintemp = (np.min(spacial_corrected[spacial_corrected > KELVIN_OFFSET]) - KELVIN_OFFSET)
    if maxtemp == None:
        maxtemp = (np.max(spacial_corrected)  - KELVIN_OFFSET)

    # Map to color scale
    with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
        task = progress.add_task("Mapping to Color Scale...", total = len(spacial_corrected) * (len(spacial_corrected[0]) - 1))
        imageData = np.zeros((len(spacial_corrected), len(spacial_corrected[0]) - 1, 3), dtype = np.uint8)

        for x in range(len(spacial_corrected)):
            for y in range(len(spacial_corrected[0]) - 1):
                if roi != None and onBorder(x, y, top_left, bottom_right):
                    imageData[x,y] = (255,255,255)  # Highlight ROI in white
                else:
                    imageData[x,y] = colorTemp((spacial_corrected[x,y]) - KELVIN_OFFSET, mintemp, maxtemp, cmap)
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


def renderImage(filename, title = None, mintemp = None, maxtemp = None, cmap = 'magma'):
    if cmap == 'iron':
        cmap = IRON
    
    # Load raw data from CSV
    with open(filename, newline='') as csvfile:
        raw = np.array(list(csv.reader(csvfile, delimiter=';')))

    temperature_data = np.zeros((len(raw), len(raw[0])), dtype = np.float64)

    for x in range(len(raw)):
        for y in range(len(raw[0]) - 1):
            temperature_data[x,y] = float(raw[x,y].replace(',', '.'))

    if mintemp == None:
        mintemp = np.min(temperature_data[temperature_data > 0])
    if maxtemp == None:
        maxtemp = np.max(temperature_data)

    imageData = np.zeros((len(temperature_data), len(temperature_data[0]) - 1, 3), dtype = np.uint8)
    for x in range(len(temperature_data)):
            for y in range(len(temperature_data[0]) - 1):
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