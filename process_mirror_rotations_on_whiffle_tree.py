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

#Mirror parameters
in_to_m = 25.4e-3

OD = 31.9*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.47*OD
clear_aperture_inner = ID/2

new_load_method = True

base_path = 'C:/Users/lfast-admin/Documents/mirrors/M14/20250516/'
whiffle_tree_path = 'C:/Users/lfast-admin/Documents/mirrors/M16/whiffle_tree/'
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

#%%
rotation_holder = []
path_subfolder_int = np.sort([int(subfolder) for subfolder in os.listdir(base_path)])
list_of_subfolders = list(path_subfolder_int.astype('str'))

for rotation in list_of_subfolders:
    save_subfolder = base_path + rotation + '/'
    test = {"file_path":save_subfolder,"rotation":rotation}
    data_holder = []
    coord_holder = []
    ID_holder = []
    for file in os.listdir(save_subfolder):
        if file.endswith(".h5"):
            data, circle_coord, ID = measure_h5_circle(save_subfolder + file, use_optimizer = True)
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
    test.update({"surface":surface})

    plot_single_mirror(rotation,surface,include_rms=True)

    rotation_holder.append(test)

#%%
copy_holder = rotation_holder.copy()
# for test in rotation_holder:
#     if int(test["rotation"])%1 == 0:
#         copy_holder.append(test)
#copy_holder.remove(copy_holder[0])

surface_holder = []
for test in copy_holder:
    plot_ref = test["surface"]
    vals = plot_ref[~np.isnan(plot_ref)]*1000
    rms = np.sqrt(np.sum(np.power(vals,2))/len(vals))
    test.update({"rms_raw":rms})

    surface_holder.append(test["surface"])
#%%
avg_surface = np.mean(surface_holder,0)
plot_single_mirror('Average surface', avg_surface, include_rms = True)

Mavg,Cavg = get_M_and_C(avg_surface,Z)
remove_radial = [0,1,2,4,12,24,40]
avg_surface_sans_radial = remove_modes(Mavg,Cavg,Z,remove_radial)
plot_single_mirror('Avg surface sans radial', avg_surface_sans_radial, include_rms = True)

np.save(whiffle_tree_path + 'whiffle_tree_contribution.npy', avg_surface)
np.save(whiffle_tree_path + 'whiffle_tree_sans_radial.npy', avg_surface_sans_radial)


#%%



for test in copy_holder:
    plot_ref = test["surface"] - avg_surface_sans_radial
    vals = plot_ref[~np.isnan(plot_ref)]*1000
    rms = np.sqrt(np.sum(np.power(vals,2))/len(vals))
    test.update({"rms_avg":rms})

    plot_single_mirror('Mirror rotated ' + test["rotation"] + ' degrees', plot_ref, include_rms=True)

#%%


#%%
rms_raw = [test["rms_raw"] for test in copy_holder]
rms_avg = [test["rms_avg"] for test in copy_holder]
rotation = [test["rotation"] for test in copy_holder]

plt.plot(rotation,rms_raw,label='Raw rms error',color='r')
plt.plot(rotation,rms_avg,label='Mean-subtracted rms error',color='blue')
plt.xlabel('Rotation position')
plt.ylabel('Rms error(nm)')
plt.legend()
plt.title('Total normalized error with/without whiffle tree contribution')
plt.show()


#%%
plot_single_mirror('Whiffle tree contribution', avg_surface, include_rms=True)




