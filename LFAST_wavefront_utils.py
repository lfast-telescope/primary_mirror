# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:17:02 2024

@author: warre

Collection of utility algorithms for mirror profiles etc
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import csv
import os
from hcipy import *
from scipy.optimize import minimize, minimize_scalar
from General_zernike_matrix import General_zernike_matrix
from primary_mirror.LFAST_TEC_output import (
    import_4D_map_auto,
    import_cropped_4D_map, 
    measure_h5_circle,
    format_data_from_avg_circle
)


#%% Low level h5 processing and Zernike fitting

def save_image_set(folder_path,Z,remove_coef = [],mirror_type='uncoated'):
    #Store a folder containing h5 files as a tuple
    output = []
    for file in os.listdir(folder_path):
        if file.endswith(".h5"):
            try:
                if mirror_type == 'uncoated':
                    if len(remove_coef) == 0:
                        surf = import_4D_map_auto(folder_path + file,Z)
                    else:
                        surf = import_4D_map_auto(folder_path + file,Z,normal_tip_tilt_power=False,remove_coef = remove_coef)
                else:
                    surf = import_cropped_4D_map(folder_path + file,Z,normal_tip_tilt_power=False,remove_coef = remove_coef)
                output.append(surf[1])

                if False:
                    plt.imshow(surf[1])
                    plt.colorbar()
                    plt.title(file)
                    plt.show()
            except OSError as e:
                print('Could not import file ' + file)
    return output

def load_interferometer_maps(array_of_paths, Z, clear_aperture_outer, clear_aperture_inner, remove_coef=[0, 1, 2, 4], new_load_method=False, pupil_size=None):
    array_of_outputs = []
    for path in array_of_paths:
        if new_load_method:
            data_holder = []
            coord_holder = []
            wf_maps = []
            wf_maps = []
            for file in os.listdir(path):
                if file.endswith(".h5"):
                    data, circle_coord = measure_h5_circle(path + file)
                    data_holder.append(data)
                    coord_holder.append(circle_coord)

            for data in data_holder:
                if remove_coef == [0, 1, 2, 4]:
                    wf_maps.append(format_data_from_avg_circle(data, circle_coord, Z, normal_tip_tilt_power=True)[1])
                else:
                    wf_maps.append(format_data_from_avg_circle(data, circle_coord, Z, normal_tip_tilt_power=False,
                                                               remove_coef=remove_coef)[1])
            output_ref = np.flip(np.mean(wf_maps, 0), 0)

        else:
            output_ref = process_wavefront_error(path, Z, remove_coef, clear_aperture_outer, clear_aperture_inner,
                                                 compute_focal=False)

        array_of_outputs.append(output_ref)
    return array_of_outputs

def process_wavefront_error(path,Z,remove_coef,clear_aperture_outer,clear_aperture_inner,compute_focal = True,mirror_type='uncoated'): #%% Let's do some heckin' wavefront analysis!
    #Load a set of mirror height maps in a folder and average them
    references = save_image_set(path,Z,remove_coef,mirror_type)
    avg_ref = np.flip(np.mean(references,0),0)
    output_ref = avg_ref.copy()
     
    if compute_focal:
        output_foc,throughput,x_foc,y_foc = propagate_wavefront(avg_ref,clear_aperture_outer,clear_aperture_inner,Z,use_best_focus=True)     
        return output_ref, output_foc,throughput,x_foc,y_foc
    else:
        return output_ref

def return_neighborhood(surface, x_linspace, x_loc, y_loc, neighborhood_size):
    #For an input coordinate on the mirror [x_loc,y_loc], return the average pixel value less than neighborhood_size away  
    [X, Y] = np.meshgrid(x_linspace, x_linspace)
    dist = np.sqrt((X - x_loc) ** 2 + (Y - y_loc) ** 2)
    neighborhood = dist < neighborhood_size
    return np.nanmean(surface[neighborhood])  
  
#%% Wavefront analysis and propagation routines

def add_defocus(avg_ref, Z, amplitude=1):
    #Adds an "amplitude" amount of power to surface map; useful for focus optimization
    power = (Z[1].transpose(2, 0, 1)[4]) * amplitude
    left = np.min(avg_ref)
    right = np.max(avg_ref)

    if False:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(avg_ref, vmin=left, vmax=right)
        ax[1].imshow(avg_ref + power, vmin=left, vmax=right)
        plt.show()
    return avg_ref + power  #return 1D flattened surface and 2D surface


def propagate_wavefront(avg_ref, clear_aperture_outer, clear_aperture_inner, Z=None, use_best_focus=False,
                        wavelengths=[632.8e-9], fiber_diameters = None):
    #Define measured surface as a wavefront and do Fraunhofer propagation to evaluate at focal plane

    prop_ref = avg_ref.copy()
    prop_ref[np.isnan(prop_ref)] = 0

    if use_best_focus:
        if Z == None:
            Z = General_zernike_matrix(36, int(clear_aperture_radius * 1e6), int(ID * 1e6))

        prop_ref = optimize_focus(prop_ref, Z, clear_aperture_outer, clear_aperture_inner, wavelength=[1e-6])

    focal_length = clear_aperture_outer * 3.5
    grid = make_pupil_grid(500, clear_aperture_outer)

    if fiber_diameters is None:
        #Fiber parameters
        fiber_radius = 18e-6/2
        fiber_subtense = fiber_radius / focal_length
        focal_grid = make_focal_grid(15, 15, spatial_resolution=632e-9 / clear_aperture_outer)
        eemask = Apodizer(evaluate_supersampled(make_circular_aperture(fiber_subtense * 2), focal_grid, 8))
        prop = FraunhoferPropagator(grid, focal_grid, focal_length=focal_length)
    else:
        focal_grid = make_focal_grid(30, 20, spatial_resolution=632e-9 / clear_aperture_outer)
        prop = FraunhoferPropagator(grid, focal_grid, focal_length=focal_length)

    output_foc_holder = []
    throughput_holder = []

    if type(wavelengths) != list and type(wavelengths) != np.ndarray:
        wavelengths = [wavelengths]
        
    for wavelength in wavelengths:    

        wf = Wavefront(make_obstructed_circular_aperture(clear_aperture_outer,clear_aperture_inner/clear_aperture_outer)(grid),wavelength)
        wf.total_power = 1

        opd = Field(prop_ref.ravel() * 1e-6, grid)
        mirror = SurfaceApodizer(opd, 2)
        wf_opd = mirror.forward(wf)
        wf_foc = prop.forward(wf_opd)
        if fiber_diameters is None:
            throughput_holder.append(eemask.forward(wf_foc).total_power)
        else:
            EE_holder = []
            for diameter in fiber_diameters:
                fiber_radius = diameter/2
                EE_holder.append(compute_fiber_throughput(wf_foc, fiber_radius, focal_length, focal_grid))
            throughput_holder.append(EE_holder)
        size_foc = [int(np.sqrt(wf_foc.power.size))] * 2
        output_foc_holder.append(np.reshape(wf_foc.power, size_foc))

    if fiber_diameters is None:
        throughput = np.mean(throughput_holder)
    else:
        throughput = np.mean(throughput_holder, 0)

    if len(wavelengths) == 1:
        output_foc = output_foc_holder[0]
    else:
        output_foc = np.mean(output_foc_holder, 0)

    grid_dims = [int(np.sqrt(wf_foc.power.size))] * 2
    x_foc = 206265 * np.reshape(wf_foc.grid.x, grid_dims)
    y_foc = 206265 * np.reshape(wf_foc.grid.y, grid_dims)
    return output_foc, throughput, x_foc, y_foc
#%%
def compute_fiber_throughput(wf_foc, fiber_radius, focal_length, focal_grid):
    """Compute energy coupled into a circular fiber of given radius."""
    fiber_subtense = fiber_radius / focal_length
    fiber_mask = Apodizer(
        evaluate_supersampled(make_circular_aperture(fiber_subtense * 2), focal_grid, 8)
    )
    return fiber_mask.forward(wf_foc).total_power

def deravel(field,dims=None):
    if not dims:
        dims = [np.sqrt(field.size).astype(int)]*2
    new_shape = np.reshape(field,dims)
    return np.array(new_shape)
#%%

def find_best_focus(output_ref, Z, centerpoint, scale, num_trials, clear_aperture_outer, clear_aperture_inner):
    #Dumb focus compensation algorithm: just evaluate PSF with different applied defocus
    defocus_range = np.linspace(centerpoint - scale, centerpoint + scale, num_trials)
    throughput_holder = []
    for amplitude in defocus_range:
        title = 'Adding ' + str(round(amplitude,2)) + ' focus '
        defocused_avg = add_defocus(output_ref,Z,amplitude)
        output_foc,throughput,x_foc,y_foc = propagate_wavefront(defocused_avg,clear_aperture_outer,clear_aperture_inner)
        throughput_holder.append(throughput)         
    if True:
        plt.plot(defocus_range, throughput_holder)
        plt.xlabel('Defocus')
        plt.ylabel('Throughput')
    best_focus = defocus_range[np.argmax(throughput_holder)]
    return best_focus


def optimize_focus(updated_surface, Z, clear_aperture_outer, clear_aperture_inner, wavelength):
    #Focus optimizer
    res = minimize_scalar(objective_function,method='bounded', bounds=[-1,1], args = (updated_surface,Z,clear_aperture_outer,clear_aperture_inner, wavelength))
    defocused_surf= add_defocus(updated_surface,Z,amplitude=res.x)
    defocused_surf[np.isnan(defocused_surf)] = 0 

    return defocused_surf
    
def objective_function(amplitude,output_ref,Z,clear_aperture_outer,clear_aperture_inner, wavelength): #takes input, applies operations, returns a single number
    #Optimization function for minimization optimization: returns negative throughput in range [0-1]
    defocused_avg = add_defocus(output_ref, Z, amplitude)
    output_foc, throughput, x_foc, y_foc = propagate_wavefront(defocused_avg, clear_aperture_outer,
                                                               clear_aperture_inner, wavelengths=wavelength)

    if False:
        print('Amplitude is ' + str(amplitude) + ' and throughput is ' + str(throughput*100))
    return -throughput


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

import warnings
import sys

def _zernike_deprecation_warning(func_name):
    """Issue deprecation warning for moved Zernike functions."""
    warnings.warn(
        f"Importing {func_name} from primary_mirror.LFAST_wavefront_utils is deprecated. "
        f"Use 'from shared.zernike_utils import {func_name}' instead. "
        "This compatibility import will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

def get_M_and_C(*args, **kwargs):
    """DEPRECATED: Use shared.zernike_utils.get_M_and_C instead."""
    _zernike_deprecation_warning('get_M_and_C')
    
    # Add the mirror-control path to sys.path if not already there
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from shared.zernike_utils import get_M_and_C as _func
    return _func(*args, **kwargs)

def return_zernike_nl(*args, **kwargs):
    """DEPRECATED: Use shared.zernike_utils.return_zernike_nl instead."""
    _zernike_deprecation_warning('return_zernike_nl')
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from shared.zernike_utils import return_zernike_nl as _func
    return _func(*args, **kwargs)

def calculate_error_per_order(*args, **kwargs):
    """DEPRECATED: Use shared.zernike_utils.calculate_error_per_order instead."""
    _zernike_deprecation_warning('calculate_error_per_order')
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from shared.zernike_utils import calculate_error_per_order as _func
    return _func(*args, **kwargs)

def return_coef(*args, **kwargs):
    """DEPRECATED: Use shared.data_processing.return_coef instead."""
    warnings.warn(
        f"Importing return_coef from primary_mirror.LFAST_wavefront_utils is deprecated. "
        f"Use 'from shared.data_processing import return_coef' instead. "
        "This compatibility import will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from shared.data_processing import return_coef as _func
    return _func(*args, **kwargs)


