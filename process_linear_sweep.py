# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:01:42 2024

@author: warre

Compute/ calibrate height changes from TEC poke
"""

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
import os
from matplotlib import patches as mpatches
import csv

# Path to the folder of influence functions (delta maps)
path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M9/linear_sweeps/'
tec_response_path = path + 'tec_responses/'

#Mirror parameters
in_to_m = 25.4e-3

OD = 31.9*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.47*OD
clear_aperture_inner = ID

#%%Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

#%%
remove_normal_coef = [0,1,2,4]
neighborhood_size = 0.01
tec_holder = [[] for i in range(24)]
delta_holder = [[] for i in range(530)]

for subfolder in os.listdir(path)[5:]:
    if subfolder.startswith('linear_sweep'):
        reference_map_index = []
        short_path = path + subfolder + '/'
        list_of_steps = os.listdir(short_path)

        path_to_first_step = short_path + list_of_steps[0] + '/'
        path_to_last_step = short_path + '/' + list_of_steps[-1] + '/'
        with open(path_to_first_step + 'step_info.txt', 'r') as file:
            first_step_info = file.read()
        with open(path_to_last_step + 'step_info.txt', 'r') as file:
            last_step_info = file.read()
        first_tec = int(first_step_info.split('TEC: ')[-1].split(',')[0])
        last_tec = int(last_step_info.split('TEC: ')[-1].split(',')[0])

        #First, change step name to have three digits because I should have done this originally
        for step_str in list_of_steps:
            if len(step_str) ==1:
                os.replace(short_path + step_str + '/', short_path + '00' + step_str + '/')
            elif len(step_str) == 2:
                os.replace(short_path + step_str + '/', short_path + '0' + step_str + '/')

        list_of_steps = os.listdir(short_path)

        #Then, select the reference maps
        for num, step_str in enumerate(list_of_steps):
            step = int(step_str)
            print('Now running step ' + str(step))
            step_path = short_path + step_str + '/'
            with open(step_path + 'step_info.txt') as file:
                step_info = file.read()
            split_step_info = step_info.split(': ')
            step_num = int(split_step_info[1].split(',')[0])
            tec_num = int(split_step_info[3].split(',')[0])
            tec_cmd = float(split_step_info[4].split(',')[0])

            if tec_cmd == 0:
                reference_map_index.append(step)

            # if os.path.exists(step_path + 'data.npy'):
            #     output_ref = np.load(step_path + 'data.npy')
            # else:
            output_ref = process_wavefront_error(step_path,Z,remove_normal_coef,clear_aperture_outer,clear_aperture_inner, compute_focal=False, mirror_type = 'coated')
            np.save(step_path + 'data.npy', output_ref)

            step_dict = {
                "Step number": step_num,
                "TEC number": tec_num,
                "TEC cmd": tec_cmd,
                "Height map": output_ref,
                "delta": []
            }
            delta_holder[step-1] = step_dict

        ref_map_array = np.array(reference_map_index)
        #Then, go through all the steps and create deltas
        for step_str in list_of_steps:
            step = int(step_str)
            if step in reference_map_index:
                ref = delta_holder[step - 1]['Height map']
            else:
                ref_left_ind = np.max(ref_map_array[ref_map_array < step]) - 1
                ref_right_ind = np.min(ref_map_array[ref_map_array >= step]) - 1
                ref = np.nanmean([delta_holder[ref_left_ind]['Height map'], delta_holder[ref_right_ind]['Height map']], 0)

            delta = delta_holder[step - 1]['Height map'] - ref
            delta_holder[step - 1]['delta'] = delta

            tec_num = delta_holder[step-1]["TEC number"] -1

            if tec_num != last_tec:
                tec_holder[tec_num-1].append(delta_holder[step-1])
            if len(tec_holder[tec_num-1]) >= 22:
                np.save(tec_response_path + 'tec' + str(tec_num+1) + '.npy', tec_holder[tec_num-1])

#Highly optimistically, we should get to this point over the weekend

#%%Create list that holds responses for every TEC
tec_responses = [[] for i in range(24)]
for file in os.listdir(tec_response_path):
    tec_response = np.load(tec_response_path + file,allow_pickle=True)
    tec_responses[tec_response[0]['TEC number'] - 1] = tec_response
#%%
nominal_TEC_locs = import_TEC_centroids()
x_linspace = np.linspace(-clear_aperture_outer/2,clear_aperture_outer/2, tec_responses[0][0]['delta'].shape[0])
#Resume troubleshooting here
angle_diff,roe = find_global_rotation(tec_responses, nominal_TEC_locs, x_linspace)

#%%
deflection_holder = []
for tec_num, tec_response in enumerate(tec_responses):
    deflection = []
    tec_inputs = []
    tec = nominal_TEC_locs.loc[tec_num]
    for cmd_num, tec_cmd in enumerate(tec_response):
        if cmd_num > 0 and not ((tec_response[-1] == tec_cmd or cmd_num==1) and tec_cmd['TEC cmd'] == 0):
            angle = np.arctan2(tec['Y (m)'], tec['X (m)']) + angle_diff
            x_loc = np.cos(angle) * roe
            y_loc = np.sin(angle) * roe
            avg_height = return_neighborhood(tec_cmd['delta'], x_linspace, x_loc, y_loc, neighborhood_size)
            deflection.append(avg_height)
            tec_inputs.append(tec_cmd['TEC cmd'])
    deflection_holder.append(deflection)
    print(np.round(tec_inputs,1))

#%%
deflection_avg = np.mean(deflection_holder,0)
rms_holder = []
filtered_holder = []
subset_index = np.abs(tec_inputs) <= 0.5

for num, deflection in enumerate(deflection_holder):
    filtered_holder.append(ndimage.gaussian_filter1d(deflection,3))
    diff = np.subtract(deflection, deflection_avg)
    rms = np.sqrt(np.mean(np.square(diff)))
    plt.plot(np.array(tec_inputs)[subset_index], np.array(deflection)[subset_index])
    #   plt.plot(tec_inputs,filtered_avg)
    plt.xlabel('TEC command')
    plt.ylabel('Wavefront deflection (um)')
    #    plt.show()
    rms_holder.append(rms)
plt.title('Peak mirror change in response to TEC input')
plt.show()
#%%
filtered_avg = np.mean(filtered_holder,0)
subset_index = np.abs(tec_inputs) <= 0.5

for num, filtered in enumerate(filtered_holder):
    diff = np.subtract(filtered,filtered_avg)
    rms = np.sqrt(np.mean(np.square(diff)))
    plt.plot(np.array(tec_inputs)[subset_index], np.array(filtered)[subset_index])
 #   plt.plot(tec_inputs,filtered_avg)
    plt.xlabel('TEC command')
    plt.ylabel('Wavefront change (um)')
    plt.title('TEC ' + str(num+1) + ' has rms=' + str(np.round(rms,2)))
#    plt.show()
    rms_holder.append(rms)
plt.title('Filtered mirror change in response to TEC input')
plt.show()

#%%
subset_index = np.abs(tec_inputs) <= 0.5
rise = [np.max(x[subset_index]) - np.min(x[subset_index]) for x in filtered_holder]
run = np.max(np.array(tec_inputs)[subset_index]) - np.min(np.array(tec_inputs)[subset_index])

#%%
fig, ax = plt.subplots(1,1)
pcm = ax.pcolormesh(x_linspace,x_linspace,delta)
ax.set_aspect('equal')
ax.add_artist(mpatches.Circle([y_loc,x_loc], color = 'r', radius=0.01,fill=False,linewidth = 0.5))
fig.colorbar(pcm,ax=ax)
ax.set_title('TEC' + str(test_holder[step-1]['TEC number']) + ' at cmd=' + str(np.round(test_holder[step-1]['TEC cmd'],1)))
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(fig_path + str(step) + '.jpg')
plt.close()

#%%
eigenvectors = []
for tec_num, tec_response in enumerate(tec_responses):
    tec = nominal_TEC_locs.loc[tec_num]
    for cmd_num, tec_cmd in enumerate(tec_response):
        if np.round(tec_response[cmd_num]['TEC cmd'],1) == 0.5:
            eigenvectors.append(tec_response[cmd_num]['delta'])

np.save(path + 'eigenvectors.npy', eigenvectors)
np.save(path + 'eigen_slopes.npy',rise)

#%%

neighborhood_size = 0.01
max_height = [[] for i in range(5)]

x_coord = []
y_coord = []

for test in test_holder:
    if test['TEC number'] < 5:
        x_holder = []
        y_holder = []
        delta = test['Height map']
        cmd = test['TEC cmd']
        max_def_arg = np.where(np.abs(delta) == np.nanmax(np.abs(delta)))
        x_linspace = np.linspace(-clear_aperture_outer, clear_aperture_outer, delta.shape[0])
        x_loc = x_linspace[max_def_arg[0][0]]
        y_loc = x_linspace[max_def_arg[1][0]]
        avg_height = return_neighborhood(delta, x_linspace, x_loc, y_loc, neighborhood_size)
        max_height[test['TEC number']-1].append(avg_height)
        if np.abs(test['TEC cmd']) == 1:
            x_holder.append(x_loc)
            y_holder.append(y_loc)
    x_coord.append(np.mean(x_holder))
    y_coord.append(np.mean(y_holder))

#%%
x_vals = np.linspace(-1,1,21)
for vals in max_height:
    plt.plot(x_vals,vals[1:])
plt.show()
