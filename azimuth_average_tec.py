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
from hcipy import *
import os
from matplotlib import patches as mpatches
import csv

#%% Set up training system: create Zernike matrix and create a set of images from the h5 file

#Path to the folders of h5 from the interferometer

path = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M9/20240531/20/'
reference_path = path + 'unpowered/'

#Mirror parameters
in_to_m = 25.4e-3

OD = 31.9*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.47*OD
clear_aperture_inner = ID / 2


#%%Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(89,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

#%%
remove_normal_coef = [0,1,2,4]
output_ref, output_foc,throughput,x_foc,y_foc = process_wavefront_error(reference_path,Z,remove_normal_coef,clear_aperture_outer,clear_aperture_inner)

mirror = 'M9'
if True: #tip/tilt/focus subtraction
    remove_coef=[ 0,  1,  2,  4]
    title = mirror + ' without edge correction'

elif False: #modes to astigmatism subtraction
    remove_coef=[ 0,  1,  2,  3, 4, 5]
    title = mirror + ' (astigmatism removed)'
    
elif False: #astigmatism + spherical subtraction
    remove_coef = [ 0,  1,  2,  3, 4,  5,  6,  9, 10, 14]
    title = mirror + ' with edge correction'

else: #all non-coma modes to quatrefoil subtracted
    remove_coef = [ 0,  1,  2,  3, 4,  5,  6,  9, 10, 12, 14]
    title = mirror + ' with spherical and edge modes removed'
    
M,C = get_M_and_C(output_ref, Z)
unpowered_surface = remove_modes(M,C,Z,remove_coef)


#%%
TEC_locs = import_TEC_centroids()
x_linspace = np.linspace(-clear_aperture_outer,clear_aperture_outer,unpowered_surface.shape[0])
TEC_loc_shrink = 0.9
neighborhood_size = 0.01
deg_to_rad = np.pi / 180
rad_to_deg = 180 / np.pi


azimuth_profiles = []
delta_holder = []

for tec in os.listdir(path):
    if tec.isdigit():
        tec_int = int(tec)
        
        for cmd in os.listdir(path + tec):
            path_to_poke = path + tec + '/'+ cmd + '/'
            cmd_int = int(cmd.split('cmd')[-1])
            output_ref = process_wavefront_error(path_to_poke,Z,remove_normal_coef,clear_aperture_outer,clear_aperture_inner, compute_focal = False)
            M,C = get_M_and_C(output_ref, Z)
            poke_surface = remove_modes(M,C,Z,remove_coef)
            delta = poke_surface - unpowered_surface
            TEC_entry = TEC_locs.loc[TEC_locs['TEC #'] == tec_int]
            x_loc = TEC_entry['X (m)'].values[0]*TEC_loc_shrink
            y_loc = TEC_entry['Y (m)'].values[0]*TEC_loc_shrink
            
            avg_height = return_neighborhood(delta,x_linspace,x_loc,y_loc,neighborhood_size)
            #%%
            fig,ax = plt.subplots()
            pcm = ax.pcolormesh(x_linspace,x_linspace,delta/2)
            #ax.add_artist(mpatches.Circle([x_loc,y_loc], color = 'r', radius=0.01,fill=False,linewidth = 0.5))
            fig.suptitle('Surface height changes from single TEC activation')
            ax.set_xlabel('um',x=1.1,)
            ax.xaxis.set_label_coords(1.1, -.02)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(pcm,ax=ax)
            plt.show()
            
            delta_holder.append(delta)
 #%%
TEC_loc_shrink = 0.85
x_loc = TEC_entry['X (m)'].values[0]*TEC_loc_shrink
y_loc = TEC_entry['Y (m)'].values[0]*TEC_loc_shrink

starting_angle = np.arctan2(y_loc,x_loc) * rad_to_deg
 
outer_radius = np.sqrt(x_loc**2 + y_loc**2)
inner_radius = clear_aperture_inner*1.3

azimuth_profiles = []
#fig,ax = plt.subplots()
for delta in delta_holder:
   # ax.pcolormesh(x_linspace,x_linspace,delta)
    for radius in [outer_radius]:#,inner_radius]:
         
         sample_height = []
         angle_range = np.arange(starting_angle-180,starting_angle+180) 
         for angle in np.arange(starting_angle-180,starting_angle+180):
             sample_x = radius * np.cos(angle * deg_to_rad)
             sample_y = radius * np.sin(angle * deg_to_rad)
             sample_height.append(return_neighborhood(delta,x_linspace,sample_x,sample_y,neighborhood_size))
             #ax.add_artist(mpatches.Circle([sample_x,sample_y], color = 'r', radius=0.01,fill=False,linewidth = 0.5))
         plt.plot(angle_range - starting_angle,np.array(sample_height)/2)
         plt.plot(angle_range - starting_angle,[0]*len(angle_range),'k',linewidth=0.5)
         plt.xlabel('Rotation from TEC (degrees)')
         plt.ylabel('Height deflection (um)')
         plt.title('Radial height changes from single TEC activation')
         azimuth_profiles.append(sample_height)
    plt.show()
    
#%% This is a deviation from the intention of this script, but idgaf. 
# I am going to rotate the delta map from a single TEC poke 24 times. I will scale it from [0,1] and then apply it to the unpowered surface
# Using optimization, I will find the combination of values that minimizes mirror rms.
# This "semi-synthetic" TEC correction is semi-bullshit and would be completely unnecessary if we could get a single fucking image out of the interferometer
# But I will try to make the best of the results and embrace the stoic oppurtunity.

angle_holder = []
for val in np.arange(0,24):
    x_loc = TEC_locs['X (m)'][val]
    y_loc = TEC_locs['Y (m)'][val]
    angle = np.arctan2(y_loc,x_loc) * rad_to_deg
    angle_holder.append(angle)

delta_angle_holder = angle_holder - angle_holder[19] #Difference from TEC20 which I used for the individual TEC response

#%%

eigenvectors = []

for rotation in delta_angle_holder:
    trial = delta.copy()
    trial[np.isnan(trial)] = 0
    old_dims = trial.shape
    rot_im = ndimage.rotate(trial,rotation)
    new_dims = rot_im.shape
    if rot_im.shape[0] % 2 == 1:
        zoom_im = ndimage.zoom(rot_im,(new_dims[0]+3)/new_dims[0])
    else:
        zoom_im = ndimage.zoom(rot_im,(new_dims[0]+2)/new_dims[0])
    crop_size = int((zoom_im.shape[0] - old_dims[0])/2)
    crop_im = zoom_im[crop_size:-crop_size,crop_size:-crop_size]
    crop_im[np.isnan(delta)] = np.nan
    # fig,ax = plt.subplots()
    # ax.pcolormesh(x_linspace,x_linspace,crop_im)
    # plt.show()
    
    eigenvectors.append(crop_im)

eigenvalues = [0]*24    

#eigenvalue_bounds = optimize.Bounds(lb=[-1]*24,ub=[1]*24)

eigenvalue_bounds = []
for i in np.arange(len(eigenvalues)):
    eigenvalue_bounds.append([-1,1])
#%%

eigenvalue_path = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M9/eigenvalues/'
plot_intermediate = False

no_superimposed_eigenvalues = np.load(eigenvalue_path + 'rms_optimized.npy')
no_superimpose_surface = add_tec_influences(unpowered_surface,eigenvectors,no_superimposed_eigenvalues)

magnitude = 0.1

coef_list = [3,5,6,9,10,14,15,20,21,27,28,35,36,44,45,54,55,65,66,77,78,90]
coef_list = [28]

for coef in coef_list:
    eigenvalues = no_superimposed_eigenvalues.copy()
    term = (Z[1].transpose(2,0,1)[coef])*magnitude 
    desired_surface = unpowered_surface + term

    if plot_intermediate:
        plt.imshow(unpowered_surface)
        plt.title('Z' + str(coef) + ' unpowered')
        plt.colorbar()
        plt.show()

    if plot_intermediate:
        plt.imshow(term)
        plt.title('Z' + str(coef) + ' raw term')
        plt.colorbar()
        plt.show()

    if plot_intermediate:
        plt.imshow(desired_surface)
        plt.title('Z' + str(coef) + ' desired')
        plt.colorbar()
        plt.show()
    
    print('Optimizing for Z' + str(coef))
    reduced_surface,eigenvalues = optimize_TECs(desired_surface,eigenvectors,eigenvalues,eigenvalue_bounds,clear_aperture_outer,clear_aperture_inner,Z,metric='rms')
    superimposed_surface = add_tec_influences(unpowered_surface,eigenvectors,eigenvalues)
    delta_surface = superimposed_surface - no_superimpose_surface
    error_surface = delta_surface - (-term)
    
    if plot_intermediate:
        plt.imshow(reduced_surface)
        plt.title('Z' + str(coef) + ' reduced')
        plt.colorbar()
        plt.show()

    if plot_intermediate:
        plt.imshow(imposed_surface)
        plt.title('Z' + str(coef) + ' imposed')
        plt.colorbar()
        plt.show()

    if plot_intermediate:
        plt.imshow(error_surface)
        plt.title('Z' + str(coef) + ' error')
        plt.colorbar()
        plt.show()
    
    vals = error_surface[~np.isnan(error_surface)]
    rms = np.sqrt(np.sum(np.power(vals,2))/len(vals))*1000
    
    if False:
        title = 'Optimized Z' + str(coef) + ' with ' + str(round(rms)) + 'nm rms error'
        plot_mirrors_side_by_side(delta_surface, -term,title,subtitles=['TEC-generated surface', 'Desired surface'])
        plt.savefig(eigenvalue_path + 'Z' + str(coef) + '.jpg')
        plt.show()
        
        file_name = 'Z' + str(coef) + '.npy'
        np.save(eigenvalue_path + file_name, eigenvalues)
    else:
        zern_name = return_zernike_name(coef)
        if zern_name:
            zern_name = zern_name[0].lower() + zern_name[1:]
        else: 
            zern_name = 'Z' + str(coef)
        title = 'Idealized correction portrays ' + zern_name + ' with ' + str(round(rms)) + 'nm rms error'
        plot_single_mirror(title,delta_surface/2)
   #%% 
    if len(eigenvalues[np.abs(eigenvalues)>1]) > 0:
        print('Z' + str(coef) + ' went beyond the limits')
#%%
wave = np.linspace(400e-9,1.7e-6,14)

flatter_path = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M9/20240528/unpowered/'
output_ref, output_foc,throughput,x_foc,y_foc = process_wavefront_error(flatter_path,Z,remove_normal_coef,clear_aperture_outer,clear_aperture_inner)
M,C = get_M_and_C(output_ref, Z)
uncorrected_surface = remove_modes(M,C,Z,remove_coef)

foc_scale = float(np.max(output_foc))

uncorrected_title = 'Mirror without TEC correction'
vals = uncorrected_surface[~np.isnan(uncorrected_surface)]*1000
bounds = compute_cmap_and_contour(vals)

output_foc,throughput,x_foc,y_foc = propagate_wavefront(uncorrected_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True,wavelengths = wave)

plot_mirror_and_psf(uncorrected_title,uncorrected_surface,output_foc,throughput,x_foc,y_foc,bounds = bounds)

corrected_title = 'Mirror with idealized TEC correction'
corrected_eigenvalues = np.load(eigenvalue_path + 'rms_optimized.npy')
corrected_surface = add_tec_influences(unpowered_surface,eigenvectors,corrected_eigenvalues)

output_foc,throughput,x_foc,y_foc = propagate_wavefront(corrected_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True,wavelengths = wave)
plot_mirror_and_psf(corrected_title,corrected_surface,output_foc,throughput,x_foc,y_foc,bounds = bounds, foc_scale = foc_scale)

#%%


#%%
# plot_mirrors_side_by_side(imposed_surface, term,title,subtitles=['TEC-generated surface', 'Desired surface'])


# #%%

# M,C = get_M_and_C(reduced_surface, Z)

# remove_coef=[ 0,  1,  2,  4]
# title = mirror + ' with simulated TEC edge correction'
# corrected_surface = remove_modes(M,C,Z,remove_coef)
# output_foc,throughput,x_foc,y_foc = propagate_wavefront(corrected_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True)
# plot_mirror_and_psf(title,corrected_surface,output_foc,throughput,x_foc,y_foc)
# plot_mirror_and_cs(title,corrected_surface,include_reference = [12,24,40,60],Z=Z,C=C)

# remove_coef=[ 0,  1,  2,  4, 12,24,40,60]
# title = mirror + ' with TEC and spherical correction'
# sph_corrected_surface = remove_modes(M,C,Z,remove_coef)
# output_foc,throughput,x_foc,y_foc = propagate_wavefront(sph_corrected_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True)
# plot_mirror_and_psf(title,updated_surface,output_foc,throughput,x_foc,y_foc)
# plot_mirror_and_cs(title,sph_corrected_surface,include_reference = [12,24,40,60],Z=Z,C=C)


#     #%%
    
