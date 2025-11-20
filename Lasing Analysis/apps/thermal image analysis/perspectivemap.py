'''
This module contains code to account for the warped perspective
caused by aiming the camera off-axis to avoid measuring its own reflection.
It uses a set of manually determined calibration points stored in REFERENCE_PATH
to generate an image/coordinate transformation which maps points on the camera image
to points within the chip coordinate system.


'''
import cv2
import numpy as np
from collections.abc import Iterable
import os

os.chdir(os.path.dirname(__file__))
REFERENCE_PATH = "./CameraPerspectivePoints.txt"

# EDIT THESE FOR RECALIBRATION

# exact WIDTH and HEIGHT (mm) of the rectangular feature used for calibration.
# as of October 2025 we are using the small outline of the aluminum chip-holding brick.
# Diagram: https://docs.google.com/drawings/d/1t9eOAOHU1ThwDQBZjNbz-dM8aimdOd7iz81cOJ7B_L8/edit?usp=sharing

TRUE_WIDTH = 48.2  # mm
TRUE_HEIGHT = 37.2  # mm
CHIP_SIDELENGTH = 32.25  # mm, perhaps redundant with CHIP_DIM below but CAD gives this value
FTRUE_WIDTH = 62 # mm, width of the full brick
FTRUE_HEIGHT = 45 # mm, height of the full brick
SCALING_FACTOR = 5  # factor to scale up physical dimensions to image pixel dimensions

WIDTH = int(TRUE_WIDTH * SCALING_FACTOR)
HEIGHT = int(TRUE_HEIGHT * SCALING_FACTOR)
SIDELENGTH = int(CHIP_SIDELENGTH * SCALING_FACTOR)

CHIP_DIM = (32, 32)  # physical dimensions of the chip
CHIP_ORIGIN = (154, 260)  # sensor bottom-left corner in image
CHIP_EXTREMUM = (307, 110)  # sensor top-right corner

# used to inverting axes
OPTRIS_IMGHEIGHT = 288


# "offset"; the position of a reference in image coordinates
# This is currently the top-left corner of the rectangle.
# this ensures the transformed image is aligned; this point will remain unmoved
ox = 102
oy = 94

# Original values for ox, oy (Allen's calibration)
# ox = 73
# oy = 83

# END CONSTANTS

# in-image positions of the rectangle's corners
actual = np.loadtxt(REFERENCE_PATH, delimiter=",", dtype="float32")

# the new, "squared" positions that you want your corners to be at
target = np.float32([[ox, oy], [ox + WIDTH, oy],
                     [ox, oy + HEIGHT], [ox + WIDTH, oy + HEIGHT]])

# build transformation matrix
image_transform = cv2.getPerspectiveTransform(actual, target)


def get_target_corners():
    '''
    Returns the target corners of the calibration rectangle in image coordinates.
    '''
    return target

def get_chip_corners():
    '''
    Returns the corners of the chip in image coordinates.
    '''
    chip = np.float32([[ox + WIDTH - SIDELENGTH, oy + HEIGHT - SIDELENGTH], [ox + WIDTH, oy + HEIGHT - SIDELENGTH],
                      [ox + WIDTH - SIDELENGTH, oy + HEIGHT], [ox + WIDTH, oy + HEIGHT]])
    return chip

def get_fbrick_corners():
    '''
    Returns the corners of the full brick in image coordinates.
    '''
    brick = np.float32([[ox - (FTRUE_WIDTH - TRUE_WIDTH) * SCALING_FACTOR, oy - (FTRUE_HEIGHT - TRUE_HEIGHT) * SCALING_FACTOR], [ox + WIDTH,  oy - (FTRUE_HEIGHT - TRUE_HEIGHT) * SCALING_FACTOR],
                      [ox - (FTRUE_WIDTH - TRUE_WIDTH) * SCALING_FACTOR, oy + HEIGHT], [ox + WIDTH, oy + HEIGHT]])
    return brick

def get_calibration_dimensions():
    '''
    Returns the physical dimensions (width, height) of the calibration rectangle in mm.
    '''
    return TRUE_WIDTH, TRUE_HEIGHT, SCALING_FACTOR

def refresh_reference():
    '''
    Reloads the reference points from file and recalculates the perspective transform.
    Call this function if you have edited the reference points file.
    '''
    actual = np.loadtxt(REFERENCE_PATH, delimiter=",", dtype="float32")
    target = np.float32([[ox, oy], [ox + WIDTH, oy],
                     [ox, oy + HEIGHT], [ox + WIDTH, oy + HEIGHT]])
    global image_transform
    image_transform = cv2.getPerspectiveTransform(actual, target)

def cv_xy_package(x, y):
    '''
    package a point and/or sequences of coordinates corresponding to points
    in an opencv-digestible manner
    '''

    if isinstance(x, Iterable):
        pts = np.array([np.zeros((len(x), 2))], dtype="float32")
        pts[:, :, 0] = x
        pts[:, :, 1] = y

    else:  # single value case
        pts = np.array([[[x, y]]], dtype="float32")

    return pts


def perspective_map_points(x, y, transform):
    '''
    Wrapper function for PerspectiveTransform to be more flexible with inputs
    Returns an x, y of newly transformed points

    Params:
    x, y: float or Iterable[float]: Coordinate or iterable of coordinates along
    either the x or y axis. x and y should have the same length.

    transform: the transformation matrix to be applied to x and y.
    '''
    # format points for cv2

    pts = cv_xy_package(x, y)

    # apply the perspective transform to the points
    transformed = cv2.perspectiveTransform(pts, m=transform)
    xnew, ynew = transformed[:, :, 0], transformed[:, :, 1]
    return xnew, ynew


def image_coords_to_cartesian(x, y, imgheight=OPTRIS_IMGHEIGHT):
    '''
    Converts image pixel coordinates (y=0 at top) to cartesian (y=0 at bottom).
    '''

    return x, -y + imgheight


def to_arbitrary_coords(x, y, dimensions, origin, extremum):
    '''

    Using a reference origin and extremum point defining a rectangular ROI
    as well as its physical dimensions, take arbitrary points and map them
    to physical space in a cartesian coordinate system centered on the origin.

    params:
    x, y float or Array[float]: x and y coordinates (should be paired) to be converted

    dimensions Tuple(float, float): reference dimensions describing the width and height of
                                    a rectangular ROI.

    origin, extremum Tuple(float, float): reference points to define the new coordinate system with.
                                          the origin will be (0, 0), and the extremum will be at the max dimensions
                                          of the rectangle.
    '''
    # shift everything so that (0, 0) is the origin

    ox, oy = origin
    ex, ey = extremum

    x, y = x - ox, y - oy
    ex, ey = ex - ox, ey - oy

    # normalize everything out of the shifted extremum and then scale by desired dimensions

    dim_x, dim_y = dimensions

    x = (x / ex) * dim_x
    y = (y / ey) * dim_y

    return x[0], y[0]


def camera_to_roi(x, y, transform=image_transform, roi_dim=CHIP_DIM, origin=CHIP_ORIGIN, extremum=CHIP_EXTREMUM, imgheight=OPTRIS_IMGHEIGHT):
    '''
    Wraps the entire mapping process into a convenient function.
    Default values can be configured at the top of the file.
    '''
    # pre-calibrate by correcting for barrel distortion
    points = cv_xy_package(x, y)

    # undistorted = cv2.undistortPoints(points, cam, distCoeff)
    # x_prec, y_prec = undistorted[:, :, 0], undistorted[:, :, 1]

    # perspective map
    x_trans, y_trans = perspective_map_points(x, y, transform)

    # change y axis direction
    x_flip, y_flip = image_coords_to_cartesian(x_trans, y_trans, imgheight)

    # do the same thing with the reference points
    ox, oy = origin
    ex, ey = extremum

    # for now, distortion correction is disabled - does weird things to coordinates
    # undistorted = cv2.undistortPoints(points, cam, distCoeff)
    # x_prec, y_prec = undistorted[:, :, 0].flatten(), undistorted[:, :, 1].flatten()

    xref_trans, yref_trans = perspective_map_points(
        [ox, ex], [oy, ey], transform)
    refx_flip, refy_flip = image_coords_to_cartesian(
        xref_trans, yref_trans, imgheight)

    refx_flip = refx_flip[0]
    refy_flip = refy_flip[0]
    origin_converted = (refx_flip[0], refy_flip[0])
    extremum_converted = (refx_flip[1], refy_flip[1])

    # convert everything to chip coordinates
    cx, cy = to_arbitrary_coords(
        x_flip, y_flip, roi_dim, origin_converted, extremum_converted)

    # return cx, cy
    return x_trans[0][0], y_trans[0][0]
