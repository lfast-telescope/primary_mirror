'''
measure_display_mirror.py
Generic interface for mirror measurement and output
Take measurements using interferometer, compute surface and display
1/9/2025 warrenbfoster
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from matplotlib import cm
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize
import pickle
import cv2 as cv
from matplotlib.widgets import EllipseSelector
from General_zernike_matrix import *
from tec_helper import *
from LFAST_TEC_output import *
from LFAST_wavefront_utils import *
from hcipy import *
from interferometer_utils import *
import os
from matplotlib import patches as mpatches
import csv
from LFASTfiber.libs.libNewport import smc100
from LFASTfiber.libs import libThorlabs
from plotting_utils import *
#%%
#Mirror parameters
in_to_m = 25.4e-3

OD = 32*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.5*OD
clear_aperture_inner = 0.5*ID

number_frames_avg = 20 #Number of frames to take for on-board averaging
number_measurements = 5 #Number of averaged measurements to save
remove_normal_coef = [0, 1, 2, 4] #Zernike coefficients to subtract
s_gain = 0.5 #Gain for tip/tilt/focus correction based on Zernike fitted wavefront

#Object for Newport interface for tip/tilt/focus correction
s = smc100('COM3',nchannels=3)

#Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

#%% Set up path to folder holding measurements
base_path = 'C:/Users/lfast-admin/Documents/mirrors/'
mirror_num = 11

#Path for mirror
mirror_path = base_path + 'M' + str(mirror_num) + '/'
if not os.path.exists(mirror_path): os.mkdir(mirror_path)

#Path for daily measurements
folder_name = datetime.datetime.now().strftime('%Y%m%d')
save_path = mirror_path + folder_name + '/'
if not os.path.exists(save_path): os.mkdir(save_path)

#Path for current measurement iteration
list_of_measurements = os.listdir(save_path)

take_new_measurement = True

if take_new_measurement:
    measurement_number = len(list_of_measurements)
else:
    measurement_number = len(list_of_measurements)-1

save_subfolder = save_path + str(measurement_number) + '/'
if not os.path.exists(save_subfolder): os.mkdir(save_subfolder)

if take_new_measurement:
    #Align beam and take measurements
    start_alignment(7, number_frames_avg, s, s_gain)

    for num in np.arange(number_measurements):
        take_interferometer_measurements(save_subfolder, num_avg=number_frames_avg, onboard_averaging=True, savefile=str(num))


#%% Load measurements sequentially to perform circle detection. The pupil is then defined based on the coordinate average of the set.
test1 = 'C:/Users/lfast-admin/Documents/mirrors/M11/20250205/0/'
test2 =  'C:/Users/lfast-admin/Documents/mirrors/M11/20250207/1/'
test_holder = []

for test in [test1, test2]:
    save_subfolder = test
    data_holder = []
    coord_holder = []
    ID_holder = []
    for file in os.listdir(save_subfolder):
        if file.endswith(".h5"):
            data, circle_coord, ID = measure_h5_circle(save_subfolder + file)
            data_holder.append(data)
            coord_holder.append(circle_coord)
            ID_holder.append(ID)

    avg_circle_coord = np.mean(coord_holder, axis=0)
    avg_ID = np.mean(ID_holder)

    #Based on the defined pupil, process the measurements
    increased_ID_crop = 1.25

    wf_maps = []

    for data in data_holder:
        wf_maps.append(format_data_from_avg_circle(data, avg_circle_coord, clear_aperture_outer, clear_aperture_inner*increased_ID_crop, Z, normal_tip_tilt_power=True)[1])

    surface = np.flip(np.mean(wf_maps, 0), 0)

    test_holder.append(surface)
    #np.save(fig_path + 'surface_v' + str(step_num) + '.npy', surface)
#%%
delta = test_holder[1] - test_holder[0]
M,C = get_M_and_C(delta, Z)
remove_coef = [ 0,  1,  2,  4]
updated_surface = remove_modes(M,C,Z,remove_coef)
plot_mirror_and_cs('Delta',updated_surface,Z=Z,C=C)


plot_single_mirror('Radially symmetric surface changes',updated_surface,include_rms=False)