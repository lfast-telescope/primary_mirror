import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import optimize
from matplotlib import cm
from scipy import interpolate
import cv2 as cv
from matplotlib.widgets import EllipseSelector
from General_zernike_matrix import *
from tec_helper import *
from LFAST_TEC_output import *
from LFAST_wavefront_utils import *
from plotting_utils import *
import pickle
from hcipy import *
import os
from matplotlib import patches as mpatches
import csv

path = "C:/Users/lfast-admin/Documents/mirrors/M9/20240815/0_rerun_characterization/"

in_to_m = 25.4e-3

OD = 31.9*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.47*OD
clear_aperture_inner = ID
remove_normal_coef = [0,1,2,4]

#%%Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))
#%%
inhouse_averaging = True
ref = []

list_of_steps = os.listdir(path)[:-2]
array_of_steps = np.array([int(z) for z in list_of_steps])
flat_maps = np.where(np.array(array_of_steps)%3==0)[0]
if inhouse_averaging:
    for index, stepnum in enumerate(list_of_steps):
            savefile =stepnum + '_' + os.listdir(path + stepnum)[-1][:-5]
            print(savefile)
            output_ref = process_wavefront_error(path + stepnum + '/', Z, remove_normal_coef, clear_aperture_outer, clear_aperture_inner, compute_focal=False)

            if int(stepnum) % 3 == 0:
                if stepnum == str(array_of_steps[np.max(flat_maps)]):
                    ref = np.mean([ref,output_ref.copy()],0)
                else:
                    next_step_index = np.min(flat_maps[flat_maps > index])
                    next_stepnum = list_of_steps[next_step_index]
                    if len(next_stepnum) < 2:
                        next_stepnum = '0' + next_stepnum
                    next_ref = process_wavefront_error(path + next_stepnum + '/', Z, remove_normal_coef, clear_aperture_outer, clear_aperture_inner, compute_focal=False)
                    ref = np.mean([output_ref.copy(),next_ref.copy()],0)

            plt.imshow(output_ref-ref,vmin=-1.5,vmax=1.5)
            plt.xticks([])
            plt.yticks([])
            plt.title(savefile)
            plt.colorbar()
            plt.savefig(path + 'fig/' + savefile + '.png')
            plt.show()
            np.save(path + 'processed/' + savefile + '.npy', output_ref)
else:
    for timestamp in os.listdir(path):
        output_ref = process_wavefront_error(path + timestamp + '/', Z, remove_normal_coef, clear_aperture_outer, clear_aperture_inner, compute_focal=False)
        np.save(path + 'processed/' + timestamp + '.npy', output_ref)

#%%
if not os.path.exists(path + 'fig/'): os.mkdir(path + 'fig/')

file_holder = []
file_list = os.listdir(path + 'processed/')
for file in file_list:
    file_holder.append(np.load(path + 'processed/' + file))

ref = np.mean(file_holder[0:3],0)
deltas = np.subtract(ref,file_holder)

start_time = file_list[4]

for num,x in enumerate(deltas):
    if num == 4:
        delta_time = 0
    else:
        delta_hour = int(file[:2]) - int(start_time[:2])
        delta_min = int(file[2:4]) - int(start_time[2:4])
        delta_sec = int(file[4:6]) - int(start_time[4:6])
        delta_time = delta_hour*3600 + delta_min*60 + delta_sec - 20
    file = file_list[num]
    plt.imshow(x,vmin=-1,vmax=0.6)
    plt.colorbar(label = 'um')
    plt.xticks([])
    plt.yticks([])
    plt.title('Mirror change ' + str(delta_time) + 's after turning on TEC')
    plt.savefig(path + 'fig/' + file[:-4] + '.png')
    plt.close()

