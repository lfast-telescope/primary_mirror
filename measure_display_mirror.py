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
mirror_num = '14'

whiffle_tree_contribution = np.load(base_path + 'M' + mirror_num + '/whiffle_tree_contribution.npy')

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
#%%
if not os.path.exists(save_subfolder): os.mkdir(save_subfolder)

if take_new_measurement:
    #Align beam and take measurements
    start_alignment(1, number_frames_avg, s, s_gain)

    for num in np.arange(number_measurements):
        take_interferometer_measurements(save_subfolder, num_avg=number_frames_avg, onboard_averaging=True, savefile=str(num))
#%%

# Load measurements sequentially to perform circle detection. Tsdfgdfsgdfsgfdhe pupil is then defined based on the coordinate average of the set.
if mirror_num == "8":
    test1 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250116/2/', #reference, starting figure
             'title':'Start'}

    test2 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250312/0/', #after day 1
             'title':'3/12 PM'}

    test3 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250313/1/', #same figure, stabilized overnight
             'title':'3/13 AM'}

    test4 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250313/3/', #mirror after 4 hrs polishing
             'title':'3/13 PM'}

    test5 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250314/2/', ##mirror after 4 hrs polishing
             'title':'3/14 PM'}

    test6 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250317/1/',  #AM, after weekend stabilization
             'title':'3/17 AM'}

    test7 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250317/3/', #PM, after run
             'title':'3/17 PM'}

    test8 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250318/1/', #PM, after run
             'title':'3/18 PM'}

    test9 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250319/2/', #PM, after run
             'title':'3/19 PM'}

    test10 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250320/3/', #PM, after run
             'title':'3/20 PM'}

    test11 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250321/0/',  # PM, after run
              'title': '3/21 PM'}

    test12 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M8/20250324/1/',  # AM, after weekend
              'title': '3/21 PM'}

    test_suite = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test12]
    subtests = [test10, test12]

elif mirror_num == "11":
    test1 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M11/20250224/4/',  # reference, starting figure
             'title': '2/24 PM'}
    test2 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M11/20250225/0/',  # reference, starting figure
             'title': '2/25 AM'}
    subtests = [test1, test2]

elif mirror_num == "10":
    test1 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M10/20250307/1/',  # reference, starting figure
             'title': '3/7 PM'}
    subtests = [test1]

elif mirror_num == "14":
    test1 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250326/2/',  #starting figure
             'title': '3/26 PM'}
    test2 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250326/5/',  # figure with 90 deg CCW rotation
             'title': '3/26 PM'}
    test3 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250328/0/',  # after two hours with 90 deg clock on runner
             'title': '3/28 AM'}
    test4 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250328/1/',  # test 3 with 5 degree rotation to point N->N, also waiting 20min
             'title': '3/28 AM'}
    test5 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250328/2/',  # Rotate back to the position of test 3 to see if it repeats
             'title': '3/28 AM'}
    test6 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250328/6/',  # After two more hours with -90 deg clock on runner (opposite direction)
             'title': '3/28 PM'}
    test7 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250331/0/',  # After two more hours with -90 deg clock on runner (opposite direction)
             'title': '3/31 PM'}
    test8 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250331/360/',  # After two more hours with -90 deg clock on runner (opposite direction)
             'title': '3/28 PM'}
    test9 = {'path': 'C:/Users/lfast-admin/Documents/mirrors/M14/20250402/0/',  # After two more hours with -90 deg clock on runner (opposite direction)
             'title': '4/1 PM'}
    last_test = {'path': save_subfolder,  # Last  test saved
                'title': save_subfolder.split('/')[-3]}

    subtests = [test6]

for test in subtests:
    save_subfolder = test['path']
    data_holder = []
    coord_holder = []
    ID_holder = []
    for file in os.listdir(save_subfolder):
        if file.endswith(".h5"):
            print('Now processing ' + file)
            data, circle_coord, ID = measure_h5_circle(save_subfolder + file, use_optimizer=True)
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

    surface = np.flip(np.mean(wf_maps, 0), 0) - whiffle_tree_contribution

    test.update({'surface': surface})
    #np.save(fig_path + 'surface_v' + str(step_num) + '.npy', surface)

#surface = subtests[-1]["surface"]
M,C = get_M_and_C(surface, Z)
remove_coef = [ 0,  1,  2, 3, 4,5,6,7,8,9,10,11,13,14]
remove_coef = [0,1,2,4]
remove_astig = [0,1,2,3,4,5]
coef_correctable = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 20]
updated_surface = remove_modes(M,C,Z,remove_coef)

if False:
    X,Y = np.meshgrid(np.linspace(-OD/2,OD/2,surface.shape[0]),np.linspace(-OD/2,OD/2,surface.shape[0]))
    distance_from_center = np.sqrt(np.square(X)+np.square(Y))
    pupil_boolean = (distance_from_center > 3*25.4e-3) * (distance_from_center < 15*25.4e-3)
    updated_surface[~pupil_boolean] = np.nan


plot_single_mirror('N14 ',updated_surface,include_rms=True)
#plot_mirror_and_cs('N14 beyond clear aperture',updated_surface,Z=Z,C=C, OD=OD)
#%%
#%%
updated_surface = remove_modes(M,C,Z,coef_correctable)
wf_foc, throughput, x_foc, y_foc = propagate_wavefront(updated_surface, clear_aperture_outer, clear_aperture_inner,
                                                       Z, use_best_focus=True)
plot_mirror_and_psf('Corrected N8 on 20250321',updated_surface,wf_foc,throughput,x_foc,y_foc,foc_scale=[-3,-5.5])

#%%
test_suite = subtests.copy()
defocus_amount = [0]*len(test_suite)
offset = [0]*len(test_suite)

test_num_1 = -2
test_num_2 = -1

compared_surfaces = [test_num_1,test_num_2]

augmented_surfaces = []
for num in compared_surfaces:
    starting_surface = test_suite[num]["surface"]
    M,C = get_M_and_C(starting_surface, Z)
    remove_coef = [ 0,  1,  2, 3, 4,5,6,7,8,9,10,11,13,14]
    remove_coef = [0,1,2,4]
    coef_correctable = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 20]
    updated_surface = remove_modes(M,C,Z,remove_coef)

    new_surface = add_defocus(updated_surface,Z,defocus_amount[num]) + offset[num]


    if True:
        X, Y = np.meshgrid(np.linspace(-OD / 2, OD / 2, surface.shape[0]),
                           np.linspace(-OD / 2, OD / 2, surface.shape[0]))
        distance_from_center = np.sqrt(np.square(X) + np.square(Y))
        pupil_boolean = (distance_from_center > 3 * 25.4e-3) * (distance_from_center < 15 * 25.4e-3)
        new_surface[~pupil_boolean] = np.nan

    augmented_surfaces.append(new_surface)

delta = augmented_surfaces[1] - augmented_surfaces[0]

title = 'N14 figure changes ' + str(test_suite[test_num_2]["path"].split('/')[-3])
name_set = [test_suite[test_num_1]["title"],test_suite[test_num_2]["title"]]
plot_many_mirror_cs(title,augmented_surfaces,name_set,include_reference = None,Z=None,C=None,OD=OD)
plot_mirrors_side_by_side(augmented_surfaces[1], augmented_surfaces[0], title, subtitles=['After: ','Before: '])
plot_mirror_and_cs('Delta ' + str(test_suite[test_num_2]["path"].split('/')[-3]),delta)
