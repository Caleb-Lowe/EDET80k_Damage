import perspectivemap as pm
import optrisROI
import camera
import cv2
from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
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
    def __init__(self, data, label):
        self.label = label
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
        self.maxtemp = np.max(temperature_data)
        self.mintemp = np.min(temperature_data)

        self.perspective = False
        self.lambertian = False
        self.occlusion = False

    # def getData(self):
    #     return self.data
    
    def setTitle(self, title:str):
        self.label = title
    
    def getMaxTemp(self, roi = None):
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
        
    def showCorrections(self):
        print("Perspective Correction Applied: ", self.perspective)
        print("Lambertian Correction Applied: ", self.lambertian)
        print("Occlusion Correction Applied: ", self.occlusion)


    def refreshTempExtrema(self):
        self.mintemp = np.min(self.data[self.data > AMBIENT_TEMPERATURE - KELVIN_OFFSET])
        self.maxtemp = np.max(self.data)

    def perspectiveCorrect(self):
        if self.perspective:
            print("Warning: Perspective correction has already been applied to this dataset")
        # Correct intensity values for perspective
        transform = pm.image_transform
        spacial_corrected = cv2.warpPerspective(
            intensity(self.data), transform, (len(self.data[0]), len(self.data)), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.data = temperature(spacial_corrected)
        self.perspective = True


    def lambertianCorrect(self, roi:optrisROI.ROI, cp:camera.CameraPosition):
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
        self.refreshTempExtrema()
        self.lambertian = True


        
    def occlusionCorrect(self, roi: optrisROI.ROI, cp, resolution = 5):
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
                                    radius_l, radius_u = cp.apertureRadii(x, y, rho, phi)
                                    if (radius_l > camera.APERTURE_DIAMETER / 2) or (radius_u > camera.APERTURE_DIAMETER / 2):
                                        occludedCounter += 1
                            
                            if occludedCounter >= numPoints:
                                self.data[x,y] = self.data[x,y]  # All rays occluded; return measured temperature
                            elif occludedCounter == 0:
                                self.data[x,y] = self.data[x,y] # No occlusion; return measured temperature
                            else:
                                self.data[x,y] = temperature((intensity(self.data[x,y]) * numPoints - AMBIENT_TEMPERATURE ** 4 * occludedCounter) / (numPoints - occludedCounter))
        self.occlusion = True
        self.refreshTempExtrema()

    def polyfit(self, degree:int = 3, roi: optrisROI.ROI = optrisROI.CHIP):
        # Fit a polynomial surface to the temperature data within the ROI
        # xvals = []
        # yvals = []
        # zvals = []
        # for x in range(len(self.data)):
        #     for y in range(len(self.data[0])):
        #         if roi != None and roi.inROI(x, y):
        #             xvals.append(x)
        #             yvals.append(y)
        #             zvals.append(self.data[x,y])
        xmin, xmax, ymin, ymax = [int(bound) for bound in roi.getBounds()]
        xvals = np.linspace(xmin + 1, xmax, xmax - xmin - 1)
        yvals = np.linspace(ymin + 1, ymax, ymax - ymin - 1)
        X, Y = np.meshgrid(np.array(xvals), np.array(yvals))
        Z = self.data[xmin + 1:xmax, ymin + 1:ymax]

        def poly2d(M, *args):
            x, y = M
            prms = np.array([a for a in args])
            x0, y0 = prms[len(prms) - 2:]
            prms = np.delete(prms, [len(prms) - 3, len(prms) - 1])
            degree = int(np.sqrt(len(prms)))
            c = prms.reshape(degree, degree)
            z = np.polynomial.polynomial.polyval2d(x - x0, y - y0, c)
            # z[x <= x0] = np.polynomial.polynomial.polyval2d(x - x0, y - y0, c)
            # z[y >= y0] = np.polynomial.polynomial.polyval2d(x - x0, y - y0, c)
            return z

        # Initial guesses to the fit parameters.
        guess_prms = np.zeros((degree,degree))
        guess_prms[0,0] = 120
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

        return TemperatureSet(fit, self.label + " (Polyfit Degree " + str(degree) + ")"), fitfunc


    def render(self, roi:optrisROI.ROI = None, cmap = IRON, vmin = None, vmax = None):
        if cmap != IRON:
            cmap = matplotlib.colormaps.get_cmap(cmap)
        if vmin is None:
            vmin = self.mintemp
        if vmax is None:
            vmax = self.maxtemp
        if roi != None:
            imageData = np.zeros((len(self.data), len(self.data[0]), 3), dtype = np.uint8)
            colorTemp = lambda temp: (int(cmap((temp - vmin) / (vmax - vmin))[0]*255), 
                                      int(cmap((temp - vmin) / (vmax - vmin))[1]*255), 
                                      int(cmap((temp - vmin) / (vmax - vmin))[2]*255))
            for x in range(len(self.data)):
                    for y in range(len(self.data[0])):
                        if roi != None and roi.onBorder(x, y):
                            imageData[x,y] = (255,255,255)  # Highlight ROI in white
                        # elif roi != None and roi.inROI(x, y):
                        #     imageData[x,y] = (0,255,0)
                        else:
                            imageData[x,y] = colorTemp(self.data[x,y])
            img = Image.fromarray(imageData)          
        else:
            img = self.data
        
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar(label='Temperature (°C)')
        plt.title(self.label)
        plt.show()


    def render3D(self, roi: optrisROI.ROI = None):
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