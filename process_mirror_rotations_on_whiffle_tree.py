import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import optimize
from matplotlib import cm
from scipy import interpolate
from scipy import ndimage
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

if False:
    base_path = 'C:/Users/lfast-admin/Documents/mirrors/M14/20250515/'
else:
    base_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M14/'
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))
mirror_num = base_path.split('/')[-2]
#%%
day_holder = []
list_of_days = os.listdir(base_path)[7:9]
for day in list_of_days:
    if os.path.isdir(base_path + day):
        day_path = base_path + day + '/'
        if os.path.exists(day_path + 'rotation_holder.npy'):
            rotation_holder = np.load(day_path + 'rotation_holder.npy', allow_pickle=True)
        else:
            rotation_holder = []

            day_path = base_path + day + '/'
            list_of_tests = os.listdir(day_path)
            test_boolean = [test.isnumeric() for test in list_of_tests]
            tests_for_rotations = list(np.array(list_of_tests)[test_boolean])

            path_subfolder_int = np.sort([int(subfolder) for subfolder in tests_for_rotations])
            list_of_subfolders = list(path_subfolder_int.astype('str'))

            for rotation in list_of_subfolders:
                save_subfolder = day_path + rotation + '/'
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
            np.save(day_path + 'rotation_holder.npy', rotation_holder, allow_pickle=True)
        day_holder.append(rotation_holder)

#%%
label_iter = 0
for i, day in enumerate(day_holder):
    copy_holder = day.copy()
    list_of_rotations = [test["rotation"] for test in copy_holder]
    array_of_rotations = np.array(list_of_rotations).astype(int)
    array_of_mod_rotation = np.mod(array_of_rotations,360)

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
plt.ylabel('Rms error (nm)')
plt.title('Rms error at different rotations on whiffle tree')
#plt.legend(['1','','','2'])
plt.show()
#%%
surfaces = [test["surface"] for test in rotation_holder]
avg_ref = np.mean(surfaces[:-1],0)
Mavg, Cavg = get_M_and_C(avg_ref, Z)
surface_without_radial = remove_modes(Mavg, Cavg, Z, [0,1,2,4,12,24,40])
plot_single_mirror('Average from rotation', surface_without_radial, include_rms=True)
np.save(day_path + '/whiffle_tree_contribution.npy',surface_without_radial)

#%%

for day_under_test in day_holder:
    title = day_under_test[0]["file_path"].split('/')[-3]
    difference_holder = []
    morph_holder = []
    starting_surface = day_under_test[0]["surface"]
    if len(day_under_test) > 8:
        day_under_test = day_under_test[:-1]
    for test in day_under_test:
        considered_surface = test["surface"]
        angle = -int(test["rotation"])
        rotated_surface = ndimage.rotate(considered_surface, angle, order=0, reshape=False)
        rotated_surface[rotated_surface==0] = np.nan
        difference = rotated_surface - starting_surface
    #    plot_single_mirror(test["rotation"],difference,include_rms = True)
        difference_holder.append(difference)
        morph_holder.append(rotated_surface)
        #plot_mirrors_side_by_side(rotated_surface,starting_surface,angle, include_difference_plot= True)

    plot_single_mirror(mirror_num,np.mean(morph_holder,0), include_rms=True, save_path = day_path + '/mirror_surface.jpg')
    np.save(day_path + '/morphed_result.npy', np.mean(morph_holder,0))

#%%
remove_correctable = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 20]

M,C = get_M_and_C(np.mean(morph_holder,0),Z)
updated_surface = remove_modes(M,C,Z,remove_correctable)
plot_single_mirror('N19 with edge modes removed',updated_surface, include_rms=True)
