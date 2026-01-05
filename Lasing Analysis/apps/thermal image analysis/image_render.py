from PIL import Image
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
from pylab import *


# Create "iron" colormap (used by the Optris Camera)
# Path needs to be updated for new computers
IRON_RAW = np.flipud(np.asarray(Image.open("C:/Users/ssuub/Desktop/EDET80k_Damage/Lasing Analysis/apps/thermal image analysis/Iron Color Palette.png")))
IRON = LinearSegmentedColormap.from_list('iron', IRON_RAW / 255)


def colorTemp(temperature, min, max, cmap = 'magma'):
    # Returns an RGB tuple corresponding to the temperature value
    progress = (temperature - min) / (max - min)
    if cmap != IRON:
        cmap = matplotlib.colormaps.get_cmap(cmap)
    color_val = cmap(int(progress * 255))
    return (int(color_val[0] * 255), int(color_val[1] * 255), int(color_val[2] * 255))


def renderImage(filename, title = None, vmin = None, vmax = None, cmap = 'iron'):
    if cmap == 'iron':
        cmap = IRON
    
    # If filename is a path (to a csv file), load file
    # Otherwise assume it is already an array of temperatures
    if isinstance(filename, str):
        # Load raw data from CSV
        with open(filename, newline='') as csvfile:
            raw = np.array(list(csv.reader(csvfile, delimiter=',')))

        temperature_data = np.zeros((len(raw), len(raw[0]) - 1), dtype = np.float64)

        for x in range(len(raw)):
            for y in range(len(raw[0]) - 1):
                temperature_data[x,y] = float(raw[x,y].replace(',', '.'))
    else:
        temperature_data = filename

    # Determine the temperature range of the data
    if vmin == None:
        vmin = np.min(temperature_data)
    if vmax == None:
        vmax = np.max(temperature_data)

    # Map each temperature to a color
    imageData = np.zeros((len(temperature_data), len(temperature_data[0]), 3), dtype = np.uint8)
    for x in range(len(temperature_data)):
            for y in range(len(temperature_data[0])):
                imageData[x,y] = colorTemp((temperature_data[x,y]), vmin, vmax, cmap)


    # Render image using matplotlib
    plt.close("Thermal Image")

    img = Image.fromarray(imageData)

    fig, ax = plt.subplots(1, 1, num = "Thermal Image")

    if title != None:
        plt.title(title)
    
    imgplot = ax.imshow(img, vmin = vmin, vmax = vmax, cmap = cmap)
    ax.set_axis_off()

    plt.axis('off')
    plt.colorbar(imgplot, ax = ax, label = 'Temperature (°C)')
    plt.show()