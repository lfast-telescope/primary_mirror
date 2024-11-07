# Second attempt at performing closed loop control of M9 using the 24 measured influence functions
# Using optimization, I will find the combination of values that minimizes mirror rms.
# Uses eigenvectors file from more extensive linearization data

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

# Path to the folder of influence functions (delta maps)
path = 'C:/Users/lfast-admin/Documents/mirrors/M9/'
tec_path = 'C:/Users/lfast-admin/Documents/LFAST_TEC/PMC_GUI/tables/20241015/'

folder_name = datetime.datetime.now().strftime('%Y%m%d')
if not os.path.exists(path + folder_name): os.mkdir(path + folder_name)
folder_path = path + folder_name + '/'

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
eigenvalues = [0 for i in range(24)]

eigenvalue_bounds = []
for i in np.arange(len(eigenvalues)):
    eigenvalue_bounds.append([-0.6, 0.6])

for step_num in ['uncorrected','v1','v2','v3','v4','v5']:
    initial_save_path = folder_path + step_num + '/'
    if not os.path.exists(initial_save_path): os.mkdir(initial_save_path)

    for num in np.arange(number_averaged_frames):
        take_interferometer_measurements(initial_save_path, num_avg=number_frames_avg, onboard_averaging=True, savefile=step_num + '_' + str(num))

    surface = process_wavefront_error(initial_save_path, Z, remove_normal_coef, clear_aperture_outer, clear_aperture_inner, compute_focal=False)
    vals = surface[~np.isnan(surface)] * 1000
    rms = np.sqrt(np.sum(np.power(vals,2))/len(vals))

    plt.imshow(surface)
    plt.xticks([])
    plt.yticks([])
    plt.title(step_num + ' has ' + str(round(rms)) + 'nm rms error')
    plt.show()

    desired_surface = surface.copy()

    reduced_surface, eigenvalues = optimize_TECs(desired_surface, eigenvectors, eigenvalues, eigenvalue_bounds,
                                                 clear_aperture_outer, clear_aperture_inner, Z, metric='rms')
    improved_surface = add_tec_influences(surface, eigenvectors, eigenvalues)

    vals = improved_surface[~np.isnan(improved_surface)]
    rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals)) * 1000

    eigenvalue_output = eigenvalues * eigenvector_cmd_ref
    with open(tec_path + 'corrections_based_on_'+step_num+'.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(eigenvalue_output)

    input('Apply the new eigenvectors to the mirror')

    print('TECs are heating')
    tic = time.time()
    while time.time() - tic < 300:
        coef_filename = take_interferometer_coefficients(number_frames_avg)
        coef_file = "C:/inetpub/wwwroot/output/" + coef_filename
        zernikes = np.fromfile(coef_file, dtype=np.dtype('d'))
        correct_tip_tilt_power(zernikes, s, s_gain)
        time.sleep(10)

#%%
tic = time.time()
while time.time() - tic < 30:
    coef_filename = take_interferometer_coefficients(number_frames_avg)
    coef_file = "C:/inetpub/wwwroot/output/" + coef_filename
    zernikes = np.fromfile(coef_file, dtype=np.dtype('d'))
    correct_tip_tilt_power(zernikes, s, s_gain)
    time.sleep(10)


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