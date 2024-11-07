# Third attempt at performing closed loop control of M9 using the 24 measured influence functions
# v2 works but requires too much "man in the loop"
# I want to track the existing TEC correction, and have recommendations be updated to that
# This can be done in a while loop or something, idk
# Once this is established, I'll write a script to transition from proportional control to integration
#   after we hit a certain threshold of correction
#
# Using optimization, I will find the combination of values that minimizes mirror rms.
# Uses eigenvectors file from more extensive linearization data
import time

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

# Path to the folder of influence functions (delta maps)
path = 'C:/Users/lfast-admin/Documents/mirrors/M9/'
tec_path = 'C:/Users/lfast-admin/Documents/LFAST_TEC/PMC_GUI/tables/'

folder_name = datetime.datetime.now().strftime('%Y%m%d')
if not os.path.exists(path + folder_name): os.mkdir(path + folder_name)
if not os.path.exists(tec_path + folder_name): os.mkdir(tec_path + folder_name)

folder_path = path + folder_name + '/'
tec_path = tec_path + folder_name + '/'

fig_path = folder_path + 'figures/'
if not os.path.exists(fig_path): os.mkdir(fig_path)

# Mirror parameters
in_to_m = 25.4e-3
OD = 31.9 * in_to_m  # Outer mirror diameter (m)
ID = 3 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

#Interferometer parameters
number_frames_avg = 30
number_averaged_frames = 5

# %%Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44, int(clear_aperture_outer * 1e6), int(clear_aperture_inner * 1e6))
remove_normal_coef = [0, 1, 2, 4]

s_gain = 0.5
s = smc100('COM3',nchannels=3)

# %%

eigenvectors = np.load(path + 'eigenvectors.npy')

eigenvector_cmd_ref = 0.5
#eigenvalues = [0 for i in range(24)]
#eigenvalues = [0.16506464, 0.048271351, -0.05514196, 0.211742062, 0.099669819, 0.129454181, 0.200889152, 0.260121459, 0.009595516, 0.047980494, 0.288285892, 0.131411055, 0.108104213, 0.162067565, 0.018735804, 0.063295173, 0.159772269, 0.02991954, 0.108047584, 0.122649578, 0.074381552, 0.125742855, 0.050129896, 0.209152127]
#eigenvalues = [0.177892168, -0.006043927, -0.028166022, 0.003563217, -0.018615902, 0.002713767, 0.0880891, 0.174854869, -0.087945185, -0.026298161, -0.026891576, -0.048932701, -0.00398623, -0.004685945, -0.128899643, -0.119739366, 0.017259612, -0.000602331, 0.090038439, -0.009922718, 0.01777506, -0.006849058, -0.014295127, 0.195476407]
#eigenvalues = [ 0.190973  , -0.00216194, -0.0846456 ,  0.00925242,  0.00027523, 0.02580604,  0.04483602,  0.00803905, -0.05959679, -0.1310124 , 0.07965635, -0.10351066, -0.00454872,  0.02780594, -0.15824744,       -0.02867862, -0.03660763, -0.00141162,  0.01439509, -0.00471691,       -0.00337181,  0.00211778, -0.00039394,  0.21574715]
eigenvalues = [ 0.21436123, -0.00495113, -0.11757119, -0.00494711,  0.00661357, 0.00801154,  0.01040047,  0.06070638, -0.1089327 , -0.06495096,   -0.01855315, -0.10009754, -0.00949694,  0.00325726, -0.1335276 ,  -0.12330648,  0.00989675, -0.00595356,  0.00632757, -0.01411828, -0.00646934, -0.00605925,  0.00190873,  0.23893678]
eigen_gain = 0.2
inter_eigen_gain = 0.6
integrator_flag = True
surface_rms_holder = []
optimized_rms_holder = []
integrated_error_holder = []
number_steps = 21
integrated_error = np.zeros([500,500])
only_integrate_correctable = True
neutral_tec_current = True
remove_spherical = False
filter_surface = False
filter_radius = 11
spline_surface = True
#%%
if filter_surface:
    print('gmm')
    filtered_eigenvectors = []
    for eigenvector in eigenvectors:
        eig_copy = eigenvector.copy()
        eig_copy[np.isnan(eigenvector)] = 0
        eig_filt = ndimage.gaussian_filter(eig_copy, sigma=filter_radius)
        eig_filt[np.isnan(eigenvector)] = np.nan

        filtered_eigenvectors.append(eig_filt)

        if False:
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(eigenvector)
            ax[1].imshow(filtered_eigenvectors[-1])
            for i in [0,1]:
                ax[i].set_xticks([])
                ax[i].set_yticks([])
            plt.show()
elif spline_surface:
    x_linspace = np.linspace(-clear_aperture_outer, clear_aperture_outer, eigenvectors[0].shape[0])
    X, Y = np.meshgrid(x_linspace, x_linspace)
    filtered_eigenvectors = []
    for eigenvector in eigenvectors:
        coord = np.where(~np.isnan(eigenvector))
        tck,fp,ier,msg = interpolate.bisplrep(x_linspace[coord[0]],x_linspace[coord[1]],eigenvector[~np.isnan(eigenvector)],full_output=1)
        znew = interpolate.bisplev(x_linspace,x_linspace,tck)
        znew[np.isnan(eigenvector)] = np.nan
        filtered_eigenvectors.append(znew)

eigenvalue_bounds = []
for i in np.arange(len(eigenvalues)):
    eigenvalue_bounds.append([-0.6, 0.6])
#%%
start_alignment(3,number_frames_avg,s,s_gain)

#%%
write_eigenvalues_to_csv(tec_path + 'updating_eigenvalues.csv', eigenvalues)
input('Set the TECs to automatically update')
if np.abs(np.sum(eigenvalues)) > 0:
    hold_alignment(180, number_frames_avg, s, s_gain)

#%%
if remove_spherical:
    remove_coef = [0, 1, 2, 4, 12, 24, 40]

for step_num in np.arange(number_steps):

    save_path = folder_path + str(step_num) + '/'
    if not os.path.exists(save_path): os.mkdir(save_path)

    for num in np.arange(number_averaged_frames):
        take_interferometer_measurements(save_path, num_avg=number_frames_avg, onboard_averaging=True, savefile=str(step_num) + '_' + str(num))

    data_holder = []
    coord_holder = []
    wf_maps = []
    for file in os.listdir(save_path):
        if file.endswith(".h5"):
            data, circle_coord = measure_h5_circle(save_path + file)
            data_holder.append(data)
            coord_holder.append(circle_coord)

    for data in data_holder:
        wf_maps.append(format_data_from_avg_circle(data, circle_coord, Z, normal_tip_tilt_power=True)[1])

    surface = np.flip(np.mean(wf_maps, 0), 0)
    np.save(fig_path + 'surface_v' + str(step_num) + '.npy',surface)

    if remove_spherical: #remove spherical modes for the fitting / optimization
        M,C = get_M_and_C(surface,Z)
        surface = remove_modes(M, C, Z, remove_coef)

    output_foc, throughput, x_foc, y_foc = propagate_wavefront(surface, clear_aperture_outer, clear_aperture_inner, Z=Z, use_best_focus=True)

    if remove_spherical:
        plot_mirror_and_psf('Sph subtracted iteration ' + str(step_num), surface, output_foc, throughput, x_foc, y_foc)
    else:
        plot_mirror_and_psf('Iteration ' + str(step_num), surface, output_foc, throughput, x_foc, y_foc)
    surface_vals = surface[~np.isnan(surface)] * 1000
    surface_rms = np.sqrt(np.sum(np.power(surface_vals,2))/len(surface_vals))
    surface_rms_holder.append(surface_rms)

    if filter_surface:
        surface_copy = surface.copy()
        surface_copy[np.isnan(surface_copy)] = 0
        desired_surface = ndimage.gaussian_filter(surface_copy,filter_radius)
        desired_surface[np.isnan(desired_surface)] = np.nan
    elif spline_surface:
        coord = np.where(~np.isnan(surface))
        tck,fp,ier,msg = interpolate.bisplrep(x_linspace[coord[0]],x_linspace[coord[1]],surface[~np.isnan(surface)],full_output=1)
        znew = interpolate.bisplev(x_linspace,x_linspace,tck)
        znew[np.isnan(eigenvector)] = np.nan
        desired_surface = znew.copy()
    else:
        desired_surface = surface.copy()

    if not integrator_flag:

        if neutral_tec_current:

            reduced_surface, eigenvalue_delta = optimize_TECs(desired_surface, eigenvectors, eigenvalues, eigenvalue_bounds,
                                                     clear_aperture_outer, clear_aperture_inner, Z, metric='rms_neutral')
        reduced_vals = reduced_surface[~np.isnan(reduced_surface)]
        reduced_rms = np.sqrt(np.sum(np.power(reduced_vals,2))/len(reduced_vals)) * 1000

        if np.sum(eigenvalues) == 0:
            eigenvalues = eigenvalue_delta * eigenvector_cmd_ref
        else:
            eigenvalues = eigenvalues + eigenvalue_delta * eigenvector_cmd_ref * eigen_gain

        optimized_surface = add_tec_influences(surface, eigenvectors, eigenvalues)

        optimized_vals = optimized_surface[~np.isnan(optimized_surface)]
        optimized_rms = np.sqrt(np.sum(np.power(optimized_vals, 2)) / len(optimized_vals)) * 1000
        optimized_rms_holder.append(optimized_rms)

        #if surface_rms < reduced_rms * 1.2:
        #   integrator_flag = True

    else:
        if only_integrate_correctable:
            reduced_surface, eigenvalue_delta = optimize_TECs(desired_surface, eigenvectors, eigenvalues, eigenvalue_bounds, clear_aperture_outer, clear_aperture_inner, Z, metric='rms')
            correctable = desired_surface - reduced_surface
            correctable[np.isnan(reduced_surface)] = 0
            correctable_filtered = ndimage.gaussian_filter(correctable, filter_radius)
            correctable_filtered[np.isnan(reduced_surface)] = np.nan

            integrated_error = integrated_error + correctable_filtered
            plot_mirrors_side_by_side(correctable_filtered,integrated_error,title='Iteration ' + str(step_num), subtitles = ['Surface','Integrated surface'])

        else:
            integrated_error = integrated_error + surface
        integrated_error_holder.append(integrated_error)

        if filter_surface or spline_surface:
            reduced_integration, eigenvalue_delta = optimize_TECs(integrated_error, filtered_eigenvectors, eigenvalues, eigenvalue_bounds,
                                                              clear_aperture_outer, clear_aperture_inner, Z, metric='rms_neutral')
        else:
            reduced_integration, eigenvalue_delta = optimize_TECs(integrated_error, eigenvectors, eigenvalues, eigenvalue_bounds,
                                                              clear_aperture_outer, clear_aperture_inner, Z, metric='rms_neutral')

        optimized_surface = add_tec_influences(surface, eigenvectors, eigenvalues)

        optimized_vals = optimized_surface[~np.isnan(optimized_surface)]
        optimized_rms = np.sqrt(np.sum(np.power(optimized_vals, 2)) / len(optimized_vals)) * 1000
        optimized_rms_holder.append(optimized_rms)
        eigenvalues = eigenvalues + eigenvalue_delta * eigenvector_cmd_ref * inter_eigen_gain

    write_eigenvalues_to_csv(tec_path + 'corrections_based_on_v' + str(step_num) + '.csv', eigenvalues)
    write_eigenvalues_to_csv(tec_path + 'updating_eigenvalues.csv', eigenvalues)

    if step_num < number_steps-1:
        print('TECs are heating')
        hold_alignment(240,number_frames_avg,s,s_gain)

#%%

tic = time.time()
while time.time() - tic < 30:
    coef_filename = take_interferometer_coefficients(number_frames_avg)
    coef_file = "C:/inetpub/wwwroot/output/" + coef_filename
    zernikes = np.fromfile(coef_file, dtype=np.dtype('d'))
    correct_tip_tilt_power(zernikes, s, s_gain)
    time.sleep(10)

#%%
for filter_radius in [3, 7, 11]:
    surface_copy = surface.copy()
    surface_copy[np.isnan(surface_copy)] = 0
    desired_surface = ndimage.gaussian_filter(surface_copy, filter_radius)
    desired_surface[np.isnan(surface)] = np.nan
    reduced_surface, eigenvalue_delta = optimize_TECs(desired_surface, eigenvectors, eigenvalues, eigenvalue_bounds,
                                                      clear_aperture_outer, clear_aperture_inner, Z, metric='rms')
    changes = desired_surface - reduced_surface
    changes[np.isnan(reduced_surface)] = 0
    changes_filtered = ndimage.gaussian_filter(changes, filter_radius)
    changes_filtered[np.isnan(reduced_surface)] = np.nan

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(desired_surface)
    ax[1].imshow(desired_surface - reduced_surface)
    ax[2].imshow(changes_filtered)
    for i in range(3):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    fig.suptitle(filter_radius)
    plt.tight_layout()
    plt.show()

# %%
wave = np.linspace(400e-9, 1.7e-6, 14)

flatter_path = 'C:/Users/warre/OneDrive/Documents/LFAST/mirrors/M9/20240528/unpowered/'
output_ref, output_foc, throughput, x_foc, y_foc = process_wavefront_error(flatter_path, Z, remove_normal_coef,
                                                                           clear_aperture_outer, clear_aperture_inner)
M, C = get_M_and_C(output_ref, Z)
uncorrected_surface = remove_modes(M, C, Z, remove_coef)

foc_scale = float(np.max(output_foc))

uncorrected_title = 'Mirror without TEC correction'
vals = uncorrected_surface[~np.isnan(uncorrected_surface)] * 1000
bounds = compute_cmap_and_contour(vals)

output_foc, throughput, x_foc, y_foc = propagate_wavefront(uncorrected_surface, clear_aperture_outer,
                                                           clear_aperture_inner, Z, use_best_focus=True,
                                                           wavelengths=wave)

plot_mirror_and_psf(uncorrected_title, uncorrected_surface, output_foc, throughput, x_foc, y_foc, bounds=bounds)

corrected_title = 'Mirror with idealized TEC correction'
corrected_eigenvalues = np.load(eigenvalue_path + 'rms_optimized.npy')
corrected_surface = add_tec_influences(unpowered_surface, eigenvectors, corrected_eigenvalues)

output_foc, throughput, x_foc, y_foc = propagate_wavefront(corrected_surface, clear_aperture_outer,
                                                           clear_aperture_inner, Z, use_best_focus=True,
                                                           wavelengths=wave)
plot_mirror_and_psf(corrected_title, corrected_surface, output_foc, throughput, x_foc, y_foc, bounds=bounds,
                    foc_scale=foc_scale)

# %%