import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import lasinglib as ll
import simulationlib as sl
import shapes

import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt

import optris_csv as ocsv
from scipy.signal import medfilt

TIME_CUTOFF = 9000
OPTRIS_SIGMA = 0.2
kernel_size = 15


def tempPlot(figure_num, right_up = None, left_down = None, right_down = None, left_up = None, delim = ";", trigger1 = 5, trigger2 = 5, key = "Detector Area"):

    plt.close(figure_num)

    data = []

    if right_up != None:
        ru_data = ocsv.OptrisDataset(right_up,delim).build_array_data()
        ru_data = ru_data[ru_data["time"] <= TIME_CUTOFF]
        ru_temp = medfilt(ru_data[key], kernel_size = kernel_size)
        data.append(tempMap(getMaxima(ru_temp, trigger1), "Right", "Up") )

    if left_down != None:
        ld_data = ocsv.OptrisDataset(left_down,delim).build_array_data()
        ld_data = ld_data[ld_data["time"] <= TIME_CUTOFF]
        ld_temp = medfilt(ld_data[key], kernel_size = kernel_size)
        data.append(tempMap(getMaxima(ld_temp, trigger2), "Left", "Down") )

    if right_down != None:
        rd_data = ocsv.OptrisDataset(right_down,delim).build_array_data()
        rd_data = rd_data[rd_data["time"] <= TIME_CUTOFF]
        rd_temp = medfilt(rd_data[key], kernel_size = kernel_size)
        data.append(tempMap(getMaxima(rd_temp, trigger1), "Right", "Down") )

    if left_up != None:
        lu_data = ocsv.OptrisDataset(left_up,delim).build_array_data()
        lu_data = lu_data[lu_data["time"] <= TIME_CUTOFF]
        lu_temp = medfilt(lu_data[key], kernel_size = kernel_size)
        data.append(tempMap(getMaxima(lu_temp, trigger1), "Left", "Up") )
    
    sample = np.mean(data, axis = 0)

    fig, ax = plt.subplots(figsize = (5,5), num=figure_num)
    ax.set_title("Maximum Temperature (15x15)")
    img = ax.imshow(sample,interpolation='nearest',
                        cmap = 'magma',
                        origin='lower')

    plt.colorbar(img, cmap = 'magma')

    plt.show()
    return sample

def tempMap(local_maxima, d1, d2):
    grid_length = int(np.sqrt(len(local_maxima)))
    maxima_map = np.zeros((grid_length,grid_length))

    for i in range(len(local_maxima)):
        if d1 == "Left":
            y = grid_length - (i % grid_length + 1)
        elif d1 == "Right":
            y = i % grid_length
        elif d1 == "Up":
            x = i % grid_length
        elif d1 == "Down":
            x = grid_length - (i % grid_length + 1)

        if d2 == "Left":
            y = grid_length - (i // grid_length + 1)
        elif d2 == "Right":
            y = i // grid_length
        elif d2 == "Up":
            x = i // grid_length
        elif d2 == "Down":
            x = grid_length - (i // grid_length + 1)

        maxima_map[x, y] = local_maxima[i]

    return maxima_map

def getMaxima(temperature, trigger = 5):
    local_maxima = []
    i = 0
    current_maximum = 0
    current_minimum = np.inf

    for t in temperature:
        if i % 2 == 1:
            if t >= current_maximum - trigger:
                current_maximum = max(t, current_maximum)
            else:
                local_maxima.append(current_maximum)
                current_maximum = 0
                i += 1
        else: 
            if t <= current_minimum + trigger:
                current_minimum = min(t, current_minimum)
            else:
                current_minimum = np.inf
                i += 1

    print(f"{len(local_maxima)} local maxima found")
    return local_maxima
