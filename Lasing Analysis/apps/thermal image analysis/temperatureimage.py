import perspectivemap as pm
import optrisROI
import camera
import cv2
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from abc import ABC, abstractmethod
import plotly.graph_objects as go
from pylab import *
from rich.progress import Progress
from rich.console import Console


AMBIENT_TEMPERATURE = 299.15 # Kelvin, ambient temperature
KELVIN_OFFSET = 273.15

# Create "iron" colormap
IRON_RAW = np.flipud(np.asarray(Image.open("C:/Users/ssuub/Desktop/EDET80k_Damage/Lasing Analysis/apps/thermal image analysis/Iron Color Palette.png")))
IRON = LinearSegmentedColormap.from_list('iron', IRON_RAW / 255)

def intensity(temperatures):
    # Convert temperature values to intensity values using Stefan-Boltzmann law
    return (temperatures + KELVIN_OFFSET) ** 4

def temperature(intensities):
    # Convert intensity values to temperature values using Stefan-Boltzmann law
    return intensities ** 0.25 - KELVIN_OFFSET


class TemperatureSet:
    """Creates a complete dataset object (temperature reading and camera position).

    Parameters:
        data (array_like or string):
            The temperature data, either as a numpy array or a path to a CSV file.
        label (str):
            A label for the dataset. This will be the title of any rendered images.
        cp (camera.CameraPosition, optional):
            The position of the camera when the temperature data was collected.

    Returns:
        TemperatureSet: a TemperatureSet object containing the temperature data and camera position.
    """

    def __init__(self, data, label, cp:camera.CameraPosition = None):
        self.label = label
        self.cp = cp
        if isinstance(data, str):
            # Load raw data from CSV
            with open(data, newline='') as csvfile:
                raw = np.array(list(csv.reader(csvfile, delimiter=';')))

            temperature_data = np.zeros((len(raw), len(raw[0]) - 1), dtype = np.float64)

            for x in range(len(raw)):
                for y in range(len(raw[0]) - 1):
                    temperature_data[x,y] = float(raw[x,y].replace(',', '.'))
        else:
            temperature_data = data
        self.data = temperature_data
        self.initial_width = int(len(self.data[0]))
        self.initial_height = int(len(self.data))
        self.maxtemp = np.max(temperature_data)
        self.mintemp = np.min(temperature_data)

        # Correction flags
        self.perspective = False
        self.lambertian = False
        self.occlusion = False

        # Fitting storage
        self.fits = {}



    #=======================================================================================
    # Basic arithmetic operations for TemperatureSet objects
    #=======================================================================================
    
    def __add__(self, other):
        if isinstance(other, TemperatureSet):
            if self.data.shape != other.data.shape:
                raise Exception("TemperatureSets must have the same shape to be added")
            new_data = self.data + other.data
            return TemperatureSet(new_data, self.label + " + " + other.label)
        elif isinstance(other, (int, float)):
            new_data = self.data + other
            return TemperatureSet(new_data, self.label)
        else:
            raise Exception("Can only add TemperatureSet or scalar to TemperatureSet")
        
    def __sub__(self, other):
        if isinstance(other, TemperatureSet):
            if self.data.shape != other.data.shape:
                raise Exception("TemperatureSets must have the same shape to be subtracted")
            new_data = self.data - other.data
            return TemperatureSet(new_data, self.label + " - " + other.label)
        elif isinstance(other, (int, float)):
            new_data = self.data - other
            return TemperatureSet(new_data, self.label)
        else:
            raise Exception("Can only subtract TemperatureSet or scalar from TemperatureSet")
        
    def __mul__(self, other):
        if isinstance(other, TemperatureSet):
            if self.data.shape != other.data.shape:
                raise Exception("TemperatureSets must have the same shape to be multiplied")
            new_data = self.data * other.data
            return TemperatureSet(new_data, self.label + " * " + other.label)
        elif isinstance(other, (int, float)):
            new_data = self.data * other
            return TemperatureSet(new_data, self.label + " * " + str(other))
        else:
            raise Exception("Can only multiply TemperatureSet by TemperatureSet or a scalar")
        
    def __truediv__(self, other):
        if isinstance(other, TemperatureSet):
            if self.data.shape != other.data.shape:
                raise Exception("TemperatureSets must have the same shape to be divided")
            new_data = self.data / other.data
            return TemperatureSet(new_data, self.label + " / " + other.label)
        elif isinstance(other, (int, float)):
            new_data = self.data / other
            return TemperatureSet(new_data, self.label + " / " + str(other))
        else:
            raise Exception("Can only divide TemperatureSet by TemperatureSet or a scalar")
    
    #=======================================================================================
    # Basic data management functions
    #=======================================================================================

    def save(self, filepath:str):
        """Save temperature data to CSV

        Parameters:
            filepath :string
                The path to the CSV file where the temperature data will be saved.
        """
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            for x in range(len(self.data)):
                row = [f"{self.data[x,y]:.4f}".replace('.', ',') for y in range(len(self.data[0]))]
                row.append(';')
                writer.writerow(row)
    

    def setTitle(self, title:str):
        """Set the label for the dataset. This will be the title of any rendered images.
        
        Parameters:
            title (string):
                The new label for the dataset.
        """
        self.label = title
        
    def shape(self):
        """Return the shape of the temperature data as a tuple (height, width).
        
        Returns:
            tuple: A tuple containing the height and width of the temperature data.
        """
        return self.data.shape

    def _refreshTempExtrema(self):
        self.mintemp = np.min(self.data[self.data > AMBIENT_TEMPERATURE - KELVIN_OFFSET])
        self.maxtemp = np.max(self.data)

    def getMaxTemp(self, roi = None):
        """Get the maximum temperature in the dataset, optionally within a specified ROI.
        
        Parameters:
            roi (optrisROI.ROI, optional):
                The region of interest within which to calculate the maximum temperature. If None, the maximum temperature will be calculated for the entire dataset.

        Returns:
            float: The maximum temperature.
        """

        if roi != None:
            maxTemp = -np.inf
            for x in range(len(self.data)):
                for y in range(len(self.data[0])):
                    if roi.inRoi(x,y):
                        if self.data[x,y] > maxTemp:
                            maxTemp = self.data[x,y]
            return maxTemp
        else:
            return self.maxtemp
        
    def getMinTemp(self, roi = None):
        """Get the minimum temperature in the dataset, optionally within a specified ROI.
        
        Parameters:
            roi (optrisROI.ROI, optional):
                The region of interest within which to calculate the minimum temperature. If None, the minimum temperature will be calculated for the entire dataset.

        Returns:
            float: The minimum temperature.
        """
        if roi != None:
            minTemp = np.inf
            for x in range(len(self.data)):
                for y in range(len(self.data[0])):
                    if roi.inRoi(x,y):
                        if self.data[x,y] < minTemp:
                            minTemp = self.data[x,y]
            return minTemp
        else:
            return self.mintemp
        
    #=======================================================================================
    # Correction functions (perspective, lambertian, occlusion)
    #=======================================================================================
        
    def showCorrections(self):
        """Print which corrections have been applied to the dataset."""
        print("Perspective Correction Applied: ", self.perspective)
        print("Lambertian Correction Applied: ", self.lambertian)
        print("Occlusion Correction Applied: ", self.occlusion)

    def perspectiveCorrect(self, cp: camera.CameraPosition = None):
        """Apply perspective correction to the dataset using the camera position.
        
        Parameters:
            cp (camera.CameraPosition, optional):
                The camera position used for perspective correction. If None, the camera position associated with the dataset will be used.
        """
        if cp == None:
            if self.cp == None:
                raise Exception("TemperatureSet.perspectiveCorrect() missing 1 required positional argument: 'cp'")
            else:
                cp = self.cp

        if self.perspective:
            print("Warning: Perspective correction has already been applied to this dataset")

        transform = cp.imageTransform()

        # Preliminary calculations to determine location of corners and check for image flips
        corners = np.array([[[0,0], [self.initial_width, 0], [self.initial_width, self.initial_height], [0, self.initial_height]]], dtype= np.float32)
        transformed = cv2.perspectiveTransform(corners, transform)
        if (transformed[0,0,0] > transformed[0,1,0]) or (transformed[0,3,0] > transformed[0,2,0]):
            print("Warning: Image Flipped Horizontally")
        if (transformed[0,0,1] > transformed[0,3,1]) or (transformed[0,1,1] > transformed[0,2,1]):
            print("Warning: Image Flipped Vertically")

        # Correct intensity values for perspective
        spacial_corrected = cv2.warpPerspective(
            intensity(self.data), transform, (int(np.ceil(max(transformed[0,:,0]))), int(np.ceil(max(transformed[0,:,1])))), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.data = temperature(spacial_corrected)
        self.perspective = True


    def lambertianCorrect(self, roi:optrisROI.ROI = None, cp:camera.CameraPosition = None):
        """Corrects for lower intensity readings for a lambertian emitter (eg. paper) based on the camera position.
        
            The intensity of light emitted by a lambertian emitter is given by

            .. math:: 
                I=I_0\cos(θ)

            where :math:`θ` is the angle between the surface normal and the line of sight to the viewer. 
            This function corrects for this effect by dividing the measured intensity by the cosine of this angle, which is calculated using the camera position.
        
        Parameters:
            roi (optrisROI.ROI, optional):
                The region of interest within which to apply lambertian correction. If None, lambertian correction will be applied to the entire dataset.
            cp (camera.CameraPosition, optional):
                The camera position used for lambertian correction. If None, the camera position associated with the dataset will be used.
        """
        if cp == None:
            if self.cp == None:
                raise Exception("TemperatureSet.lambertianCorrect() missing 1 required positional argument: 'cp'")
            else:
                cp = self.cp
        if not self.perspective:
            print("Warning: Perspective correction has not been applied to this dataset. Lambertian correction may be inaccurate.")
        
        if self.lambertian:
            print("Warning: Lambertian correction has already been applied to this dataset")
        for x in range(len(self.data)):
                    for y in range(len(self.data[0])):
                        if self.data[x,y] <= KELVIN_OFFSET:
                            continue
                        elif roi != None and not roi.inROI(x, y):
                            self.data[x,y] = self.data[x,y]
                        else:
                            # Deterimes the angle of the point on the camera to the chip from the vertical
                            rx, ry, rz = cp.cameraChipVector(x, y)
                            theta = np.atan2(np.sqrt(rx**2 + ry**2), rz)
                            self.data[x,y] = temperature(intensity(self.data[x,y])/ np.cos(theta))
        self._refreshTempExtrema()
        self.lambertian = True
        # Invalidate any existing fits
        self.fits.clear()

        
    def occlusionCorrect(self, roi: optrisROI.ROI, cp: camera.CameraPosition = None, resolution = 5):
        """Corrects for lower intensity readings due to the vacuum chamber lid partially obscuring the view.
        
            Estimates the true temperature of a point based on the number of rays from the camera that are occluded by the lid. 
            Multiple ray paths are traced from each point of the dataset (assumed to be in the same plane as the chip) to a point on the camera lens, and the number of rays that are occluded by the lid is counted.
            Occluded rays are assumed to be "replaced" by emission from the lid, which is assumed to be at ambient temperature.
            The intensity is then corrected based on the fraction of rays that are occluded. 

            .. admonition:: Point Selection
                :collapsible: closed

                For each point in the dataset, :math:`R^2` rays are traced to the camera lens where :math:`R` is the resolution. Higher resolutions will result in more accurate corrections but will take longer to compute (:math:`O(R^2)` time complexity).
                The set of points :math:`P_R` on the camera lens that are traced to is given by

                .. math::
                    P_R :=\\left\\{\\left(\\frac{\\operatorname{floor}\\left(\\sqrt{n}\\right)}{R-1}\\sin\\left(\\phi\\left(n\\right)\\right),\\frac{\\operatorname{floor}\\left(\\sqrt{n}\\right)}{R-1}\\cos\\left(\\phi\\left(n\\right)\\right)\\right)\\bigg|n=0,1,\\dots,R^2-1\\right\\}

                where

                .. math::
                    \\phi\\left(n\\right) := \\frac{2\\pi n}{\\operatorname{floor}\\left(\\sqrt{n}\\right)}

                For an interactive visualization of the selected points on the lens, see
                https://www.desmos.com/calculator/rfuweg7pkd

            .. admonition:: Occlusion Detection
                :collapsible: closed

                To determine whether a line from the camera to a point on the chip is occluded by the lid, we first find the equation of the circle(s) that describes the hole in our lid. 
                If the intersection of the line with the plane of the circle is outside the circle, then the ray is occluded.

                The diameter and position of the holes have been measured and documented in the CAD files, but the lid angle and camera angle need to be measured for each new camera position.

                For an interactive visualization of the occlusion, see
                https://www.desmos.com/3d/4agexlbubf
            
        Parameters:
            roi (optrisROI.ROI, optional):
                The region of interest within which to apply occlusion correction. If None, occlusion correction will be applied to the entire dataset.
            cp (camera.CameraPosition, optional):
                The camera position used for occlusion correction. If None, the camera position associated with the dataset will be used.
            resolution (int, optional):
                The resolution of the occlusion correction.
        """
        if cp == None:
            if self.cp == None:
                raise Exception("TemperatureSet.occlusionCorrect() missing 1 required positional argument: 'cp'")
            else:
                cp = self.cp
                
        if not self.perspective:
            print("Warning: Perspective correction has not been applied to this dataset. Lambertian correction may be inaccurate.")

        if self.occlusion:
            print("Warning: Occlusion correction has already been applied to this dataset")
        # count how many rays are occluded given the resolution
        maxRho = cp.getMaxRho()

        numPoints = resolution ** 2
        deltaPoints = lambda x : 2 * x - 1


        for x in range(len(self.data)):
                    for y in range(len(self.data[0])):
                        if roi != None and not roi.inROI(x, y):
                            self.data[x,y] = self.data[x,y]
                        else:
                            occludedCounter = 0
                            for i in range(resolution):
                                rho = maxRho * i / resolution
                                for j in range(deltaPoints(i)):
                                    phi = 2 * np.pi * (j + 0.5) / deltaPoints(i)
                                    radius_l, radius_c, radius_u = cp.apertureRadii(x, y, rho, phi)
                                    if (radius_l > camera.LOWER_APERTURE_DIAMETER / 2) or (radius_c > camera.LOWER_APERTURE_DIAMETER / 2) or (radius_u > camera.UPPER_APERTURE_DIAMETER / 2):
                                        occludedCounter += 1
                            
                            if occludedCounter >= numPoints:
                                self.data[x,y] = self.data[x,y]  # All rays occluded; return measured temperature
                            elif occludedCounter == 0:
                                self.data[x,y] = self.data[x,y] # No occlusion; return measured temperature
                            else:
                                self.data[x,y] = temperature((intensity(self.data[x,y]) * numPoints - AMBIENT_TEMPERATURE ** 4 * occludedCounter) / (numPoints - occludedCounter))
        self._refreshTempExtrema()
        self.occlusion = True
        # Invalidate any existing fits
        self.fits.clear()

    def median_filter(self, kernel_size = 3):
        """Apply a median filter to the temperature data to reduce noise.
        
        Parameters:
            kernel_size (int, optional):
                The size of the kernel used for the median filter. Must be an odd integer. Default value is 3."""
        
        filtered = median_filter(self.data, kernel_size)
        self.data = filtered

    #=======================================================================================
    # Fitting functions
    #=======================================================================================

    def polyfit(self, degree:int = 3, roi: optrisROI.ROI = optrisROI.CHIP):    
        """ Fit a polynomial surface to the temperature data within the ROI

        Parameters:
            degree (int, optional):
                The degree of the polynomial fit. Default value is 3.
            roi (optrisROI.ROI, optional):
                The region of interest within which to apply the polynomial fit. If None, the polynomial fit will be applied to the entire dataset.

        Returns:
            tuple (TemperatureSet, function): A tuple containing a TemperatureSet with the fitted temperature data and the fitting function.
        """
        xmin, xmax, ymin, ymax = [int(bound) for bound in roi.getBounds()]
        xvals = np.linspace(xmin + 1, xmax, xmax - xmin - 1)
        yvals = np.linspace(ymin + 1, ymax, ymax - ymin - 1)
        X, Y = np.meshgrid(np.array(xvals), np.array(yvals))
        Z = self.data[xmin + 1:xmax, ymin + 1:ymax]

        if degree in self.fits:
            fitfunc = self.fits[degree].function
            fit = fitfunc(X, Y)
            print("Existing fit of degree" + degree + " loaded")

        else:
            def poly2d(M, *args):
                x, y = M
                prms = np.array([a for a in args])
                x0, y0 = prms[len(prms) - 2:]
                prms = np.delete(prms, [len(prms) - 3, len(prms) - 1])
                degree = int(np.sqrt(len(prms)))
                c = prms.reshape(degree, degree)
                z = np.polynomial.polynomial.polyval2d(x - x0, y - y0, c)
                return z

            # Initial guesses to the fit parameters.
            guess_prms = np.zeros((degree,degree))
            guess_prms[0,0] = self.maxtemp
            # Flatten the initial guess parameter list.
            p0 = np.append([p for prms in guess_prms for p in prms], (0, 160))

            # We need to ravel the meshgrids of X, Y points to a pair of 1-D arrays.
            xdata = np.vstack((X.ravel(), Y.ravel()))

            # Do the fit, using our custom _gaussian function which understands our
            # flattened (ravelled) ordering of the data points.
            popt, pcov = curve_fit(poly2d, xdata, Z.ravel(), p0)
            def fitfunc(x, y):
                return poly2d((x, y), *popt)
            fit = fitfunc(X, Y)
            rms = np.sqrt(np.mean((Z - fit)**2))
            print('RMS residual =', rms)

            self.fits[degree] = TemperaturePolyFit(self.data[xmin + 1:xmax, ymin + 1:ymax], degree, popt)

        return TemperatureSet(fit, self.label + " (Polyfit Degree " + str(degree) + ")"), fitfunc

    #=======================================================================================
    # Rendering functions
    #=======================================================================================

    def render(self, roi:optrisROI.ROI = None, cmap = IRON, vmin = None, vmax = None, crop = optrisROI.CHIP):
        """Render the temperature data as a 2D image.
        
        Parameters:
            roi (optrisROI.ROI, optional):
                The region of interest within which to render the temperature data. If None, the entire dataset will be rendered.
            cmap (string or matplotlib colormap, optional):
                The colormap to use for rendering. Default is 'iron'.
            vmin (float, optional):
                The minimum temperature for the color scale. If None, the minimum temperature in the dataset will be used.
            vmax (float, optional):
                The maximum temperature for the color scale. If None, the maximum temperature in the dataset will be used.
            crop (optrisROI.ROI or str, optional):
                The region to crop the image to. If an ROI is provided, it will be centered in the image. If 'FULL', the full image will be rendered.
        """

        img = self.data
        offset = [0,0]
        if cmap != IRON:
            cmap = matplotlib.colormaps.get_cmap(cmap)
        if vmin is None:
            vmin = self.mintemp
        if vmax is None:
            vmax = self.maxtemp

        # If crop is an ROI, center it in the image and crop to the size of the initial image. If crop is "FULL", render the full image without cropping.
        if isinstance(crop, optrisROI.ROI):
            roi_center = np.array([(crop.xmin + crop.xmax) / 2, (crop.ymin + crop.ymax) / 2])
            image_center = np.array([self.initial_height / 2, self.initial_width / 2])
            offset = (roi_center - image_center).astype(int)
            # Fill in out of bounds pixels with ambient temperature
            img = self.data[offset[0]:offset[0] + self.initial_height, offset[1]:offset[1] + self.initial_width]
            img = np.pad(img, ((max(0, -offset[0]), max(0, offset[0] + self.initial_height - len(self.data))), (max(0, -offset[1]), max(0, offset[1] + self.initial_width - len(self.data[0])))), constant_values = 0)
        elif crop != "FULL":
            raise Exception("Invalid crop parameter. Use an optrisROI.ROI object or 'FULL'.")
        
        if roi != None:
            imageData = np.zeros((len(img), len(img[0]), 3), dtype = np.uint8)
            colorTemp = lambda temp: (int(cmap((temp - vmin) / (vmax - vmin))[0]*255), 
                                      int(cmap((temp - vmin) / (vmax - vmin))[1]*255), 
                                      int(cmap((temp - vmin) / (vmax - vmin))[2]*255))
            for x in range(len(img)):
                    for y in range(len(img[0])):
                        if roi != None and roi.onBorder(x + offset[0], y + offset[1]):
                            imageData[x,y] = (255,255,255)  # Highlight ROI in white
                        # elif roi != None and roi.inROI(x, y):
                        #     imageData[x,y] = (0,255,0)
                        else:
                            imageData[x,y] = colorTemp(img[x,y])
            img = Image.fromarray(imageData)          


        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar(label='Temperature (°C)')
        plt.title(self.label)
        plt.show()

        # max = np.max(img)
        # min = np.min(img)
        # return ((img - min) / (max - min)) * 256


    def render3D(self, roi: optrisROI.ROI = None):
        """Render the temperature data as a 3D surface plot using the plotly library.

        .. note::
            The plot often will not render / update. Restarting VS Code sometimes fixes the issue. Requires more investigation.
        
        Parameters:
            roi (optrisROI.ROI, optional):
                The region of interest within which to render the temperature data. If None, the entire dataset will be rendered.
                """
        if roi != None:
            data = self.data[int(roi.xmin + 1):int(roi.xmax), int(roi.ymin + 1):int(roi.ymax)]
        else:
            data = self.data
        fig = go.Figure(data=[go.Surface(z=data, colorbar=dict(title="Temperature (°C)"))])
        fig.update_layout(title=dict(text=self.label), autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90))

        fig.show()

    


        


class TemperatureComposite(TemperatureSet):
    """Creates a composite temperature dataset by combining multiple TemperatureSet objects. 
    Can be used to create a more complete temperature map by combining multiple images taken from different angles or with different ROIs.
    
    Parameters:    
        datasets (list of TemperatureSet): 
            The list of TemperatureSet objects to be combined. All datasets must have the same shape.
        rois (list of optrisROI.ROI or None):
            The list of regions of interest for each dataset over which to combine data (as indicated by the mode). If None, the entire dataset will be used.
        label (str):
            The label for the composite temperature dataset. This will be the title of any rendered images.
        mode (str, optional):
            The mode for combining the datasets. Can be 'median', 'mean', or 'count'. Default is 'median'.
    """

    def __init__(self, datasets:list, rois:list, label:str, mode:str = 'median'):
        if rois != None and len(datasets) != len(rois):
            raise Exception("Number of datasets must match number of ROIs")
        self.datasets = datasets
        self.rois = rois
        self.label = label

        totaldata = np.zeros((len(datasets), len(datasets[0].data), len(datasets[0].data[0])))
        with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
            task = progress.add_task("Loading Data", total = len(datasets) * (len(datasets[0].data) * len(datasets[0].data[0])))
            for i in range(len(datasets)):
                totaldata[i] = datasets[i].data
                progress.update(task, advance = len(datasets[0].data) * len(datasets[0].data[0]))

        # Create a composite dataset by combining the datasets
        composite_data = []
        if mode == 'median':
            combine_func = np.median
        elif mode == 'mean':
            combine_func = np.mean
        elif mode == 'count':
            combine_func = len
        else:
            raise Exception("Invalid mode for compositeTemperatureSet. Use 'median', 'mean', or 'count'.")
        
        with Progress(console=Console(force_terminal=True, force_jupyter=False)) as progress:
            task = progress.add_task("Compositing Data", total = len(totaldata[0,0]) * len(totaldata[0]))
            for y in range(len(totaldata[0,0])):
                xvals = []
                for x in range(len(totaldata[0])):
                    list = []
                    for i in range(len(totaldata)):
                        if rois[i] != None and not rois[i].inROI(x, y):
                            continue
                        list.append(totaldata[i,x,y])
                    if len(list) == 0:
                        xvals.append(AMBIENT_TEMPERATURE - KELVIN_OFFSET)
                    else:
                        xvals.append(combine_func(list))
                    progress.update(task, advance = 1)
                composite_data.append(xvals)
            

            composite_data = np.swapaxes(np.array(composite_data),0,1)

        self.data = composite_data
        self.maxtemp = np.max(composite_data)
        self.mintemp = np.min(composite_data)

        self.perspective = False
        self.lambertian = False
        self.occlusion = False


class TemperatureModel(TemperatureSet):
    """Creates a TemperatureSet object based on a mathematical model of the temperature distribution. Can be used to test the effects of different distortions on a known temperature distribution.
    
    Parameters:
        model_function (function):
            A function that takes in x and y coordinates and returns a temperature value. The function should also take in any parameters needed for the model as keyword arguments.
        parameters (dict):
            A dictionary of parameters for the model function.
        shape (tuple):
            The shape of the array to be used for the temperature dataset.
        label (str):
            A label for the temperature dataset. This will be the title of any rendered images.
    """

    def __init__(self, model_function, parameters, shape:tuple, label:str):
        self.model_function = model_function
        self.parameters = parameters
        self.label = label

        model_data = np.zeros(shape, dtype = np.float64)
        for x in range(shape[0]):
            for y in range(shape[1]):
                model_data[x,y] = model_function(x, y, **parameters)
        
        self.data = model_data
        self.maxtemp = np.max(model_data)
        self.mintemp = np.min(model_data)

        self.perspective = False
        self.lambertian = False
        self.occlusion = False

    #=======================================================================================
    # Error modelling functions (lambertian, occlusion)
    #=======================================================================================
    
    def applyLambertian(self, roi:optrisROI.ROI, cp:camera.CameraPosition):
        """Applies a lambertian distortion to the temperature data based on the camera position. This is the inverse of the lambertian correction function.

            Parameters:
                roi (optrisROI.ROI):
                    The region of interest within which to apply the lambertian distortion. If None, lambertian distortion will be applied to the entire dataset.
                cp (camera.CameraPosition):
                    The camera position to be used for the lambertian distortion.
        """

        for x in range(len(self.data)):
            for y in range(len(self.data[0])):
                if roi != None and not roi.inROI(x, y):
                    self.data[x,y] = self.data[x,y]
                else:
                    # Deterimes the angle of the point on the camera to the chip from the vertical
                    rx, ry, rz = cp.cameraChipVector(x, y)
                    theta = np.atan2(np.sqrt(rx**2 + ry**2), rz)
                    trueTemp = (self.data[x,y] ** 4 * np.cos(theta)) ** 0.25
                    self.data[x,y] = trueTemp
        self._refreshTempExtrema()

        

    def _pointOcclusion(self, x:int, y:int, cp:camera.CameraPosition, resolution = 5):
        # count how many rays are occluded given the resolution
        # helper function for applyOcclusion to calculate the occlusion distortion for a single point
        maxRho = cp.getMaxRho()

        numPoints = resolution ** 2
        deltaPoints = lambda x : 2 * x - 1
        netIntensity = 0

        # occludedCounter = 0
        # for i in range(resolution):
        #     rho = maxRho * i / resolution
        #     for j in range(deltaPoints(i)):
        #         phi = 2 * np.pi * (j + 0.5) / deltaPoints(i)
        #         radius_l, radius_u = cp.apertureRadii(x, y, rho, phi)
        #         if (radius_l > camera.APERTURE_DIAMETER / 2) or (radius_u > camera.APERTURE_DIAMETER / 2):
        #             occludedCounter += 1
        
        # if occludedCounter >= numPoints:
        #     return self.data[x,y]  # All rays occluded; return measured temperature
        # elif occludedCounter == 0:
        #     return self.data[x,y] # No occlusion; return measured temperature
        # else:
        #     return temperature((intensity(self.data[x,y]) * numPoints - AMBIENT_TEMPERATURE ** 4 * occludedCounter) / (numPoints - occludedCounter))

        for i in range(resolution):
                rho = maxRho * i / resolution
                for j in range(deltaPoints(i)):
                    phi = 2 * np.pi * (j + 0.5) / deltaPoints(i)
                    radius_l, radius_c, radius_u = cp.apertureRadii(x, y, rho, phi)
                    if (radius_u > camera.UPPER_APERTURE_DIAMETER / 2) or (radius_c > camera.LOWER_APERTURE_DIAMETER / 2) or (radius_l > camera.LOWER_APERTURE_DIAMETER / 2):
                        netIntensity += AMBIENT_TEMPERATURE ** 4
                    else:
                        netIntensity += intensity(self.data[x,y])
        avgIntensity = netIntensity / resolution ** 2
        return temperature(avgIntensity)
    
    def applyOcclusion(self, roi:optrisROI.ROI, cp:camera.CameraPosition, resolution = 5):
        """Applies occlusion distortion to the temperature data based on the camera position. This is an approximate inverse of the occlusion correction function.

            Parameters:
                roi (optrisROI.ROI):
                    The region of interest within which to apply the occlusion distortion. If None, occlusion distortion will be applied to the entire dataset.
                cp (camera.CameraPosition):
                    The camera position to be used for the occlusion distortion.
                resolution (int, optional):
                    The resolution of the occlusion distortion. Higher resolutions will result in more accurate distortions but will take longer to compute (:math:`O(R^2)` time complexity). Default value is 5.
        """
        for x in range(len(self.data)):
            for y in range(len(self.data[0])):
                if roi != None and not roi.inROI(x, y):
                    self.data[x,y] = self.data[x,y]
                else:
                    self.data[x,y] = self._pointOcclusion(x, y, cp, resolution)
        self._refreshTempExtrema()

class TemperatureFit:
    """A class representing a fit to a temperature dataset. This is a base class for arbitrary fits.
    
        Parameters:
            data (np.array): 
                The temperature data used for the fit.
            function: 
                A function that takes in xy coordinates and returns a temperature value.
            parameters: 
                Fitting parameters. Exists as a reference to keep track of parameters but is not actually used. 
    """
    def __init__(self, data: np.array, function, parameters):
        self.data = data
        self.function = function
        self.parameters = parameters


class TemperaturePolyFit(TemperatureFit):
    """A class representing a 2D polynomial fit with respect to space (xy coordinates) to a temperature dataset.
    
        Parameters:
            data (np.array): 
                The temperature data used for the fit.
            degree (int): 
                The degree of the polynomial fit.
            parameters: 
                Polynomial coefficients and offsets for the fit. The polynomial coefficients should be in a flattened array of length degree^2, and the last two parameters should be the x and y offsets for the fit.
    """

    def __init__(self, data: np.array, degree:int, parameters):
        self.degree = degree
        self.parameters = parameters
        super().__init__(data, lambda x: self._poly2d(x, self.parameters), self.parameters)

    def _poly2d(self, M, *args):
        x, y = M
        prms = np.array([a for a in args])
        x0, y0 = prms[len(prms) - 2:]
        prms = np.delete(prms, [len(prms) - 3, len(prms) - 1])
        c = prms.reshape(self.degree, self.degree)
        z = np.polynomial.polynomial.polyval2d(x - x0, y - y0, c)
        return z
    
class SuperFit(ABC):

    @abstractmethod
    def __init__(self, fits:list):
        self.fits = fits
        self.function = None


class SuperPolyFit(SuperFit):
    """A class representing a polynomial fit with respect to temperature based on multiple underlying polynomial fits with respect to space. 

            The goal of this class is to create a function that takes in xy coordinates and a temperature value and returns a temperature value, while keeping track of the datesets used to create the fit.
    
        Parameters:
            temperatures (list of float): 
                The temperatures at which the fits were taken.
            fits (list of TemperaturePolyFit): 
                The underlying polynomial fits. Each polynomial fit must have a corresponding temperature in the temperatures list.
            degree (int): 
                The degree of the polynomial fit with respect to temperature.
        """
    
    def __init__(self, temperatures:list, fits:list, degree:int = 1):
        if len(temperatures) != len(fits):
            raise Exception("Number of temperatures must match number of fits")
        self.temperatures = temperatures
        self.subdegree = - np.inf
        for fit in fits:
            if not isinstance(fit, TemperaturePolyFit):
                raise Exception("All fits must be TemperaturePolyFit objects")
            if fit.degree > self.subdegree:
                self.subdegree = fit.degree
        self.coefficients = np.zeros((self.subdegree, self.subdegree, degree + 1))
        self.offsets = np.zeros(2)
        for c1 in range(self.subdegree):
            for c2 in range(self.subdegree):
                coeffs = []
                for fit in fits:
                    if fit.degree > c1 and fit.degree > c2:
                        coeffs.append(fit.parameters[c1, c2])
                    else:
                        coeffs.append(0)
                popt, pcov = np.polyfit(temperatures, coeffs, degree)
                self.coefficients[c1, c2] = popt

        def _superpoly2d(M, T):
            x, y = M
            degree = self.subdegree
            c = np.zeros((degree, degree))
            for c1 in range(degree):
                for c2 in range(degree):
                    p = self.coefficients[c1, c2]
                    c[c1, c2] = sum([p[i] * T ** (degree - i - 1) for i in range(len(p))])
            z = np.polynomial.polynomial.polyval2d(x - self.offsets[0], y - self.offsets[1], c)
            return z

        self.function = _superpoly2d        
        



