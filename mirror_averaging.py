
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:35:20 2023

@author: warre
"""

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

#%% Set up training system: create Zernike matrix and create a set of images from the h5 file

#Path to the folders of h5 from the interferometer
#path = 'C:/Users/lfast-admin/Documents/mirrors/M9/20241015/uncorrected/'
path = 'C:/Users/lfast-admin/Documents/mirrors/M9/20241015/v4/'
#path = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M8/20240308/'

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
output_ref, output_foc,throughput,x_foc,y_foc = process_wavefront_error(path,Z,remove_normal_coef,clear_aperture_outer,clear_aperture_inner)


#%%
mirror = 'M1N9'
if True: #tip/tilt/focus subtraction
    remove_coef=[ 0,  1,  2,  4]
    title = mirror + ' without TEC correction'

elif False: #modes to astigmatism subtraction
    remove_coef=[ 0,  1,  2,  3, 4, 5]
    title = mirror + ' (astigmatism removed)'
    
elif False: #astigmatism + spherical subtraction
    remove_coef = [ 0,  1,  2,  3, 4,  5,  6,  9, 10, 14]
    title = mirror + ' with edge correction'

else: #all non-coma modes to quatrefoil subtracted
    remove_coef = [ 0,  1,  2,  3, 4,  5,  6,  9, 10, 12, 14,24,40]
    title = mirror + ' with spherical and edge modes removed'
    
M,C = get_M_and_C(output_ref, Z)
updated_surface = remove_modes(M,C,Z,remove_coef)

#plot_zernike_modes_as_bar_chart(C,num_modes=66)
wave = np.linspace(400e-9,1.6e-6,13)
output_foc,throughput,x_foc,y_foc = propagate_wavefront(updated_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True,wavelengths = wave)

pupil_vmin = -400
pupil_vmax = 400
foc_vmax = 0.0007219130189887503
contour_intervals = np.arange(pupil_vmin,pupil_vmax+1,100)
plot_mirror_and_psf(title,updated_surface,output_foc,throughput,x_foc,y_foc,bounds=[pupil_vmin,pupil_vmax,contour_intervals],foc_scale=foc_vmax)

#%%
title = 'M7 (astigmatism removed)'
remove_coef = [ 0,  1,  2,  3, 4, 5]
updated_surface = remove_modes(M,C,Z,remove_coef)
plot_mirror_and_cs(title,updated_surface,include_reference = [12,24,40,60],Z=Z,C=C)
#%%
if False:
    title = 'M7 (astigmatism removed) with defocus added'
    defocus = -0.085
    plot_mirror_and_cs(title,add_defocus(updated_surface,Z,defocus),include_reference = [12],Z=Z,C=C)    

if False:
    for gain in np.linspace(0.1,1,10):
        reduced_surface = output_ref * gain
        M,C = get_M_and_C(reduced_surface, Z)
        updated_surface = remove_modes(M,C,Z,remove_coef)
        output_foc,throughput,x_foc,y_foc = propagate_wavefront(updated_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True)
        title = 'M1 with gain = ' + str(round(gain,1))
        plot_mirror_and_psf(title,updated_surface,output_foc,throughput,x_foc,y_foc)
    
    #%%
    #%%
    plt.plot(x,cs_remove*1e3)
    plt.plot(x,cs_original*1e3)
    plt.xlabel('Distance (m)')
    plt.ylabel('Height error (nm)')
    plt.legend(['RTV removed','Original surface'])
    #%%
 #   x = np.linspace(-OD/2,OD/2,updated_surface.shape[0])
    
    path_start = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M1_1_TEC/8-12/'
    path_end = ['M7/2-5-24/hub_removed/', 'M8/20240306/', 'M8/20240308/','M9/20240321/','M9/20240326/','M9/20240327/','M9/20240329/']
    path_end = ['references/','send-3(best)/']
    #path_end = os.listdir(path_start)

    #remove_coef= [ 0,  1,  2,  3, 4,  5,  6,  7,  8,  9, 10,12,14]
    remove_holder = [ 0,  1,  2, 4]
    
    output_ref_holder = []
    C_holder = []
    
    for i in np.arange(0,len(path_end)):
        print(path_end[i])
        path = path_start + path_end[i]
        
        if False:
            path = path_start + path_end[0]
            remove_coef = remove_holder[i]
        
        output_ref = process_wavefront_error(path,Z,remove_normal_coef,clear_aperture_outer,clear_aperture_inner,compute_focal = False)
        output_ref_holder.append(output_ref)
        M,C = get_M_and_C(output_ref, Z)
        C_holder.append(C)
#%%

    holder = []
    remove_coef=[ 0,  1,  2, 4]
#    remove_coef = [ 0,  1,  2,  3, 4,  5,  6,  7,  8,  9, 10, 12, 14]

    
    titles = ['M7','M8 after 5hrs polish', 'M8 after 15 hrs polish','M9 after 5hrs polish', 'M9 after 15 hrs polish', 'M9 after 20 hrs polish', 'M9 after 25 hrs polish']
    
    for i in np.arange(0,len(output_ref_holder)):        
        M,C = get_M_and_C(output_ref_holder[i], Z)
        updated_surface = remove_modes(M,C,Z,remove_coef)
        plot_mirror_and_cs(titles[i],updated_surface,include_reference = [12,24,40],Z=Z,C=C)
        
        output_foc,throughput,x_foc,y_foc = propagate_wavefront(updated_surface,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=False)
        plot_mirror_and_psf(titles[i],updated_surface,output_foc,throughput,x_foc,y_foc)

        holder.append(updated_surface)
#        error,residual = calculate_error_per_order(M,C,Z)
 #       print(error)

        #%%
    plot_mirrors_side_by_side(holder[0], holder[1], 'TEC edge correction improves mirror wavefront')
    #plot_single_mirror('Height changes from additional polish',holder[1]-holder[0])
#%%
plot_zernike_modes_as_bar_chart(C_holder[5],C_holder[4],num_modes = 44,labels = ['20hrs','15hrs'])
#%%
name_copy = ['12-12-23', '12-19-23','1-18-24','1-29-24']
name_copy = ['Start','5hr','10hr','15hr']
name_copy = ['5hrs','15hrs']
name_copy = ['M7','M8 5hrs', 'M8 15hrs', 'M9 5hrs', 'M9 15hrs','M9 20hrs','M9 25hrs']
plot_many_mirror_cs('Radially symmetric wavefront error on M9',output_ref_holder,name_copy,include_reference = None,Z=Z,C=C)