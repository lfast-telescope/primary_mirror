"""
Evaluate ability to portray Zernike modes on mirror

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from matplotlib import cm
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize
from General_zernike_matrix import *
from tec_helper import *
from LFAST_TEC_output import *
from LFAST_wavefront_utils import *
import os
from plotting_utils import *

path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M9/'

spline_surface = True

# Mirror parameters
in_to_m = 25.4e-3
OD = 31.9 * in_to_m  # Outer mirror diameter (m)
ID = 3 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

Z = General_zernike_matrix(44, int(clear_aperture_outer * 1e6), int(clear_aperture_inner * 1e6))
remove_normal_coef = [0, 1, 2, 4]

eigenvectors = np.load(path + 'eigenvectors.npy')

x_linspace = np.linspace(-clear_aperture_outer, clear_aperture_outer, eigenvectors[0].shape[0])

if spline_surface:
    filtered_eigenvectors = []
    for eigenvector in eigenvectors:
        coord = np.where(~np.isnan(eigenvectors[0]))
        tck, fp, ier, msg = interpolate.bisplrep(x_linspace[coord[0]], x_linspace[coord[1]], eigenvector[~np.isnan(eigenvector)],
                                                 full_output=1)
        znew = interpolate.bisplev(x_linspace, x_linspace, tck)
        znew[np.isnan(eigenvector)] = np.nan
        desired_surface = znew.copy()
        filtered_eigenvectors.append(desired_surface)
    eigenvectors = filtered_eigenvectors

M, C = get_M_and_C(eigenvectors[0], Z)
#%%

flat_surface = np.zeros(eigenvectors[0].shape)
zernscale = 0.1
eigenvalues = [0 for i in range(24)]
eigenvalue_bounds = [-1,1]
error_holder = []
appx_surface_holder = []
eigenvalue_holder = []
for i in np.arange(21):
    if not i in [0,1,2]:
        zern_surface = Z[1].transpose(2,0,1)[i]*zernscale
        reduced_surface, eigenvalue_delta = optimize_TECs(zern_surface*-1, eigenvectors, eigenvalues, eigenvalue_bounds, clear_aperture_outer, clear_aperture_inner, Z, metric='rms_bounded')

        approximated_surface = add_tec_influences(flat_surface, eigenvectors, eigenvalue_delta)
        error = np.sqrt(np.divide(np.nansum(np.square(reduced_surface)),len(reduced_surface[~np.isnan(reduced_surface)])))*1e3
        plot_mirrors_side_by_side(zern_surface, approximated_surface,title = 'Mirror portrays Z' + str(i) + ' with ' + str(round(error,1)) + 'nm rms error',subtitles = ['Zernike mode','Representation'])

        error_holder.append(error)
        appx_surface_holder.append(approximated_surface)
        eigenvalue_holder.append(eigenvalue_delta)
#%%
counter = -1
for i in np.arange(22):
    if not i in [0,1,2]:
        counter += 1
        zern_surface = Z[1].transpose(2,0,1)[i]*zernscale
        approximated_surface = add_tec_influences(flat_surface, eigenvectors, eigenvalue_holder[counter])
        reduced_surface = zern_surface - approximated_surface
        error = np.sqrt(np.divide(np.nansum(np.square(reduced_surface)),len(reduced_surface[~np.isnan(reduced_surface)])))*1e3

        plot_mirrors_side_by_side(zern_surface, approximated_surface,title = 'Mirror portrays Z' + str(i) + ' with ' + str(round(error,1)) + 'nm rms error',subtitles = ['Zernike mode','Representation'])

        error_holder.append(error)
        appx_surface_holder.append(approximated_surface)
        eigenvalue_holder.append(eigenvalue_delta)

