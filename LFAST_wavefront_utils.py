# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:17:02 2024

@author: warre

Collection of utility algorithms for mirror profiles etc
"""

# === Standard library imports ===
import os
import sys
import csv
import warnings

# === Third-party imports ===
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.optimize import minimize, minimize_scalar
from hcipy import *

#Add parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- mirror_control imports ---
try:
    from mirror_control.shared.wavefront_propagation import propagate_wavefront
    from mirror_control.shared.General_zernike_matrix import General_zernike_matrix
    from mirror_control.interferometer.surface_processing import (
        import_4D_map_auto,
        import_cropped_4D_map,
        measure_h5_circle,
        format_data_from_avg_circle
    )
    from mirror_control.shared.zernike_utils import get_M_and_C, remove_modes, return_coef
    from mirror_control.interferometer import interferometer_utils
    from mirror_control.shared import wavefront_propagation as _shared_wavefront_propagation
except ImportError as e:
    warnings.warn(f"mirror_control import failed: {e}. Some functionality will be unavailable.", ImportWarning)
    propagate_wavefront = None
    General_zernike_matrix = None
    import_4D_map_auto = None
    import_cropped_4D_map = None
    measure_h5_circle = None
    format_data_from_avg_circle = None
    _func = None
    interferometer_utils = None
    _shared_wavefront_propagation = None


#%%Synthetic TEC functions based on 4D measurements

def find_transfer_functions(csv_path):
    # Compute transfer function of current to heat load (A/W) based on FEA
    current = []
    edge_heat = []
    interior_heat = []

    #Load csv containing the values from FEA
    with open(csv_path, newline='') as csvfile:
        read = csv.reader(csvfile, delimiter=',', quotechar='|')
        for num, row in enumerate(read):
            if num != 0:
                current.append(float(row[0]))
                edge_heat.append(float(row[1]))
                interior_heat.append(float(row[2]))

    edge_min = np.argmin(edge_heat)
    interior_min = np.argmin(interior_heat)

    edge_heat_to_current = interpolate.CubicSpline(edge_heat[edge_min:], current[edge_min:])
    interior_heat_to_current = interpolate.CubicSpline(interior_heat[interior_min:], current[interior_min:])

    return edge_heat_to_current, interior_heat_to_current


def add_tec_influences(updated_surface, eigenvectors, eigenvalues):
    #Apply combination of [surface changes caused by TECs] * [amplitude for these changes] to mirror surface
    for num, eigenvalue in enumerate(eigenvalues):
        updated_surface = updated_surface + eigenvectors[num] * eigenvalue
    return updated_surface
    
def optimize_TECs(updated_surface,eigenvectors,eigenvalues,eigenvalue_bounds,clear_aperture_outer,clear_aperture_inner,Z,metric = 'rms'):
    #Choose set of eigenvalues that minimize mirror surface error, either based on rms error or encircled energy
    current_eigenvalues = eigenvalues.copy()
    if metric == 'rms':
        res = minimize(rms_objective_function,x0 = eigenvalues, args = (updated_surface,eigenvectors))#, method='bounded', bounds = eigenvalue_bounds)
    elif metric == 'EE':
        res = minimize(EE_objective_function,x0 = eigenvalues, args = (updated_surface,eigenvectors,clear_aperture_outer,clear_aperture_inner,Z))#, method='bounded', bounds = eigenvalue_bounds)
    elif metric == 'rms_neutral':
        res = minimize(rms_neutral_objective_function,x0 = eigenvalues, args = (updated_surface,eigenvectors,current_eigenvalues))#, method='bounded', bounds = eigenvalue_bounds)
    elif metric == 'rms_bounded':
        res = minimize(rms_bounded_objective_function,x0 = eigenvalues, args = (updated_surface,eigenvectors,current_eigenvalues))#, method='bounded', bounds = eigenvalue_bounds)
    else:
        print('Bruh, how many methods do you want?')
        return updated_surface, eigenvalues
        
    eigenvalues = res.x
    reduced_surface = add_tec_influences(updated_surface, eigenvectors, eigenvalues)
    return reduced_surface, eigenvalues


def rms_objective_function(eigenvectors, updated_surface,
                           eigenvalues):  #takes input, applies operations, returns a single number
    #Objective function for TEC optimization, reduces mirror rms error
    reduced_surface = add_tec_influences(updated_surface, eigenvectors, eigenvalues)
    vals = reduced_surface[~np.isnan(reduced_surface)]
    rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals)) * 1000
    if False:
        print('rms error is ' + str(round(rms, 3)) + 'nm')
    return rms

def rms_neutral_objective_function(eigenvectors, updated_surface,
                           eigenvalues, current_eigenvalues):  #takes input, applies operations, returns a single number
    #Objective function for TEC optimization, reduces mirror rms error
    reduced_surface = add_tec_influences(updated_surface, eigenvectors, eigenvalues)
    vals = reduced_surface[~np.isnan(reduced_surface)]
    rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals)) * 1000
    eigen_sum = np.nansum(eigenvectors + current_eigenvalues) #This is supposed to be eigenvalues but I've gotten confused somewhere about the order for variables in minimize() so don't judge
    merit = rms * (1 + eigen_sum**4)**0.1
    if True:
        print('rms=' + str(round(rms, 3)) + 'nm, sum=' + str(round(np.abs(eigen_sum), 3)) + ', merit=' + str(round(merit, 3)))
    return merit

def rms_bounded_objective_function(eigenvectors, updated_surface, eigenvalues, current_eigenvalues):  #takes input, applies operations, returns a single number
    #Objective function for TEC optimization, reduces mirror rms error
    desired_max_current = 0.6

    reduced_surface = add_tec_influences(updated_surface, eigenvectors, eigenvalues)
    vals = reduced_surface[~np.isnan(reduced_surface)]
    rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals)) * 1000
    eigensum = np.nansum(eigenvectors + current_eigenvalues)

    penalty = 1 + np.sum(np.power(np.divide(eigensum,desired_max_current), 100))

    maxval = np.nanmax(eigensum) #This is supposed to be eigenvalues but I've gotten confused somewhere about the order for variables in minimize() so don't judge
    merit = rms * penalty
    if True:
        print('rms=' + str(round(rms, 3)) + 'nm, max=' + str(round(maxval, 3)) + ', penalty=' + str(round(penalty, 2)))
    return merit

def EE_objective_function(eigenvectors, updated_surface, eigenvalues, clear_aperture_outer, clear_aperture_inner,
                          Z):  #takes input, applies operations, returns a single number
    #Objective function for TEC optimization, maximizes throughput. Slow because need to do focus compensation inside optimization
    reduced_surface = add_tec_influences(updated_surface, eigenvectors, eigenvalues)
    output_foc, throughput, x_foc, y_foc = propagate_wavefront(reduced_surface, clear_aperture_outer,
                                                               clear_aperture_inner, Z, use_best_focus=True)
    if False:
        print('EE is ' + str(round(float(throughput) * 100, 2)) + '%')
    return -float(throughput)


# %%This code disappeared into the aether. I have no idea.
def find_global_rotation(tec_responses, nominal_TEC_locs, x_linspace):
    output_plots = False
    X, Y = np.meshgrid(x_linspace, x_linspace)
    number_points = 10
    angle_diff_holder = []
    roe_holder = []

    for tec_response in tec_responses:
        tec_num = tec_response[0]['TEC number']
        nominal_x = nominal_TEC_locs.loc[tec_num - 1]['X (m)']
        nominal_y = nominal_TEC_locs.loc[tec_num - 1]['Y (m)']
        nominal_angle = np.arctan2(nominal_y, nominal_x)
        angle = []
        roe = []
        for index in [1]:
            x_holder = []
            y_holder = []
            delta = tec_response[index]['delta']
            sorted_vals = np.sort(np.abs(delta), None)
            sorted_vals = sorted_vals[~np.isnan(sorted_vals)]
            for i in np.arange(number_points):
                coord = np.where(np.abs(delta) == sorted_vals[-i])
                x_holder.append(x_linspace[coord[1][0]])
                y_holder.append(x_linspace[coord[0][0]])

            x_loc = np.median(x_holder)
            y_loc = np.median(y_holder)
            angle.append(np.arctan2(y_loc, x_loc))
            roe.append(np.sqrt(np.sum([np.power(x_loc, 2), np.power(y_loc, 2)])))

            if output_plots:
                fig, ax = plt.subplots()
                ax.pcolormesh(x_linspace, x_linspace, tec_response[index]['delta'])
                ax.set_aspect('equal')
                plt.title(
                    'Angle is ' + str(round(angle[-1] * 180 / np.pi)) + ' and distance is ' + str(round(roe[-1], 2)))
                plt.scatter(x_loc, y_loc)
                plt.show()

        angle_diff = nominal_angle - np.mean(angle)
        angle_diff_holder.append(angle_diff)
        roe_holder.append(np.mean(roe))

    angle_diff = np.mean(angle_diff_holder)
    roe = np.mean(roe_holder)
    return angle_diff, roe

# ============================================================================
# BACKWARD COMPATIBILITY IMPORTS - DEPRECATED
# ============================================================================
"""
The following functions have been moved to shared/zernike_utils.py for better
modularity and reuse across different measurement methods.

These imports are provided for backward compatibility only.
For new code, import directly from shared.zernike_utils.

WARNING: These compatibility imports will be removed in a future version.
"""

def _deprecation_warning(func_name, new_location):
    """Issue standardized deprecation warning for moved functions."""
    warnings.warn(
        f"Importing {func_name} from primary_mirror.LFAST_wavefront_utils is deprecated. "
        f"Use 'from {new_location} import {func_name}' instead. "
        "This compatibility import will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

def get_M_and_C(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.zernike_utils.get_M_and_C instead."""
    _deprecation_warning('get_M_and_C', 'mirror_control.shared.zernike_utils')
    try:
        from mirror_control.shared.zernike_utils import get_M_and_C as _func
        return _func(*args, **kwargs)
    except ImportError:
        raise ImportError("mirror_control.shared.zernike_utils not found.")

def return_zernike_nl(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.zernike_utils.return_zernike_nl instead."""
    _deprecation_warning('return_zernike_nl', 'mirror_control.shared.zernike_utils')
    try:
        from mirror_control.shared.zernike_utils import return_zernike_nl as _func
        return _func(*args, **kwargs)
    except ImportError:
        raise ImportError("mirror_control.shared.zernike_utils not found.")

def calculate_error_per_order(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.zernike_utils.calculate_error_per_order instead."""
    _deprecation_warning('calculate_error_per_order', 'mirror_control.shared.zernike_utils')
    try:
        from mirror_control.shared.zernike_utils import calculate_error_per_order as _func
        return _func(*args, **kwargs)
    except ImportError:
        raise ImportError("mirror_control.shared.zernike_utils not found.")

def return_coef(*args, **kwargs):
    """DEPRECATED: Use shared.data_processing.return_coef instead."""
    _deprecation_warning('return_coef', 'shared.zernike_utils')
    try:
        from mirror_control.shared.zernike_utils import return_coef as _func
        return _func(*args, **kwargs)
    except ImportError:
        raise ImportError("shared.zernike_utils not found.")

def save_image_set(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils.save_image_set instead."""
    _deprecation_warning('save_image_set', 'mirror_control.interferometer.interferometer_utils')
    if interferometer_utils is not None:
        return interferometer_utils.save_image_set(*args, **kwargs)
    raise ImportError("mirror_control.interferometer.interferometer_utils not found.")

def process_wavefront_error(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils.process_wavefront_error instead."""
    _deprecation_warning('process_wavefront_error', 'mirror_control.interferometer.interferometer_utils')
    if interferometer_utils is not None:
        return interferometer_utils.process_wavefront_error(*args, **kwargs)
    raise ImportError("mirror_control.interferometer.interferometer_utils not found.")

def load_interferometer_maps(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils.load_interferometer_maps instead."""
    _deprecation_warning('load_interferometer_maps', 'mirror_control.interferometer.interferometer_utils')
    if interferometer_utils is not None:
        return interferometer_utils.load_interferometer_maps(*args, **kwargs)
    raise ImportError("mirror_control.interferometer.interferometer_utils not found.")

def add_defocus(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.add_defocus instead."""
    _deprecation_warning('add_defocus', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.add_defocus(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")

def propagate_wavefront(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.propagate_wavefront instead."""
    _deprecation_warning('propagate_wavefront', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.propagate_wavefront(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")

def compute_fiber_throughput(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.compute_fiber_throughput instead."""
    _deprecation_warning('compute_fiber_throughput', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.compute_fiber_throughput(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")

def deravel(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.deravel instead."""
    _deprecation_warning('deravel', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.deravel(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")

def find_best_focus(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.find_best_focus instead."""
    _deprecation_warning('find_best_focus', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.find_best_focus(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")

def optimize_focus(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.optimize_focus instead."""
    _deprecation_warning('optimize_focus', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.optimize_focus(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")

def objective_function(*args, **kwargs):
    """DEPRECATED: Use mirror_control.shared.wavefront_propagation.objective_function instead."""
    _deprecation_warning('objective_function', 'mirror_control.shared.wavefront_propagation')
    if _shared_wavefront_propagation is not None:
        return _shared_wavefront_propagation.objective_function(*args, **kwargs)
    raise ImportError("mirror_control.shared.wavefront_propagation module not found.")
