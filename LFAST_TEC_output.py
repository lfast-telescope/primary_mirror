"""
Low level TEC control functions and FEA-based surface correction algorithms.
Mainly written by Nick Didato circa 2023
Deprecated functions written by Warren Foster have been moved to mirror_control/shared/zernike_utils.py and mirror_control/shared/General_zernike_matrix.py
"""

import pandas as pd
import numpy as np
from scipy import optimize, interpolate
import csv
import os
import pickle
import sys
from pathlib import Path


TEC_location_file = 'TEC_centroids.csv'
thermal_conductance_file = 'Thermal_conductances.csv'
Influence_function_file = 'Binaries/Influence_function matrix.obj'
Zernike_matrix_file = 'Binaries/Zernike_matrix.obj'


E_TEC = 0.476 #W/C Heat sink side thermal conductance
R = 3.2131 #TEC electrical resistance
K = 0.32131 #TEC thermal conductance
Alpha = 0.0298 #V/K Seebeck coefficient


def load_influence_function_matrix(): #load saved influence function matrix object
    
    fileobj = open(Influence_function_file,'rb')
    matrix = pickle.load(fileobj)
    fileobj.close()
    
    return matrix


def load_zernike_matrix(): #load saved Zernike matrix object
    
    fileobj = open(Zernike_matrix_file,'rb')
    matrix = pickle.load(fileobj)
    fileobj.close()
    
    return matrix


def import_TEC_centroids(chosen_file = None): #import x,y coordinates of each TEC
    
    if chosen_file == None:
        file = TEC_location_file
    else:
        file = chosen_file
    
    df = pd.read_csv(file).iloc[0:,:]
    
    return df


def import_thermal_conductance(): #import mirror/heat sink thermal conductance values 
    
    file = thermal_conductance_file
    
    df = pd.read_csv(file).iloc[0:,:]
    
    E_vals = list(df['E'])[0:24]
    E_vals.extend(list(np.ones(108)*E_TEC))
    
    return np.array(E_vals)


def Heat_loads(W,I,T_a):  #calculate heat loads, input (influence function matrix, Zernike fit to measured surface, ambient temperature in celsius)
    
    W_processed = W[0].copy()
    W_processed[np.isnan(W_processed)] = 0  #replaces NaN's with 0's
    
    I_processed = -I[0].copy()
    I_processed[np.isnan(I_processed)] = 0  #replaces NaN's with 0's
    
    W_t = W_processed.transpose() #
    
    A = np.dot(W_t,W_processed) #
    
    A_inv = np.linalg.inv(A) #
    
    B = np.dot(A_inv,W_t)       #
    
    Heat_loads = np.dot(B,I_processed) #Solves matrix equation:  Heat_loads = ((W_t*W)^-1)*W_t*I
    
    Surf = np.dot(W[1],Heat_loads) #Calculates best fit surface from heat loads
    Temp = np.dot(W[2],Heat_loads) + T_a
    TEC_temp = np.dot(W[3],Heat_loads) + T_a
    
    return Heat_loads,Surf,Temp,TEC_temp #returns the vector containing 132 heat loads and the generated surface in tuple form


def Heat_loads2(W,I,T_a):  #calculate heat loads such that sum of heat loads = 0, input (influence function matrix, Zernike fit to measured surface,ambient temperature in celsius)
    
    W_processed = W[0].copy()
    W_processed[np.isnan(W_processed)] = 0  #replaces NaN's with 0's
    
    I_processed = -I[0].copy()
    I_processed[np.isnan(I_processed)] = 0  #replaces NaN's with 0's
    
    W_t = W_processed.transpose()
    
    A = np.dot(W_t,W_processed)
    B = np.dot(W_t,I_processed)
    
    C = np.ones((1,132),dtype = 'float')
    C[0][0:24] = 0  #constrain only back TECs to sum to 0
    d = np.zeros((1,),dtype = 'float')
    f = np.zeros((1,),dtype = 'float')
    
    E = np.vstack((A,C))
    F = np.vstack((C.transpose(),f))
    G = np.hstack((B,d))
    
    H = np.hstack((E,F))
    
    H_inv = np.linalg.inv(H)

    Heat_loads = np.dot(H_inv,G)[0:132]
    
    Surf = np.dot(W[1],Heat_loads) #Calculates best fit surface from heat loads
    Temp = np.dot(W[2],Heat_loads) + T_a
    TEC_temp = np.dot(W[3],Heat_loads) + T_a
    
    return Heat_loads,Surf,Temp,TEC_temp #returns the vector containing 132 heat loads and the generated surface in tuple form         


def get_TEC_temperatures(df,temp_map): #extract TEC mirror side temperatures, df is TEC locations, temp_map is the fourth element in the heat_load output H[3]
    
    xs = ys = np.linspace(-0.45914,0.45914,500) #grid that extends out to OD TEC locations
    zs = temp_map.copy()
    zs[np.isnan(zs)] = 0
    
    Z = interpolate.interp2d(xs,ys,zs)
    
    zi = Z(xs,ys)
    
    X,Y = np.meshgrid(xs,-ys)
                
    test = np.sqrt(X**2 + Y**2)                         #
    inds = np.where((test > 0.45914) | (test < .0635))  #
    coords = list(zip(inds[0],inds[1]))                 #remove points outside OD and inside ID
    for j,m in enumerate(coords):                       #
        zi[coords[j][0]][coords[j][1]] = np.nan         #
                
                 
    xs_TEC = np.array(df['X (m)'])
    ys_TEC = np.array(df['Y (m)']) 
    
    TEC_ts = []
    
    for i,n in enumerate(xs_TEC):  #for each TEC location, extract temperature values within a 20 mm radius and average them
        TEC_point_list = []
        test = np.sqrt((X - xs_TEC[i])**2 + (Y - ys_TEC[i])**2)
        inds = np.where(test <= 0.02)
        coords = list(zip(inds[0],inds[1]))
        for j,m in enumerate(coords):
            TEC_point_list.append(zi[coords[j][0]][coords[j][1]])
        TEC_ts.append(np.nanmean(TEC_point_list))           
        
    return np.array(TEC_ts)


def TEC_electrical_load_solve_function_mirror(vals,Qh,T,E,T_a): #TEC implicit equations
    
    Th = T + 273.15
    
    T_o = T_a + 273.15

    Qc,I = vals
    
    a = ((Alpha*I)/(E) + (K)/(E) + 1)*Qc - Alpha*T_o*I + (1/2)*I**2*R + K*Th - K*T_o
    b = ((K)/(E))*Qc + Qh - Alpha*Th*I - (1/2)*I**2*R + K*Th - K*T_o
    
    return a,b



def TEC_electrical_load_mirror(Qh,T,E,T_a): #numerical solve function for current and electrical power
    
    Qc,I = optimize.fsolve(TEC_electrical_load_solve_function_mirror,(1,0.1),(Qh,T,E,T_a))
    
    P = Qh - Qc
    
    return I,P


def get_electrical_output(df,Qh,T,E,T_a): #calculate TEC currents.  Inputs are (TEC locations, heat loads, TEC temperature, thermal conductance, ambient temperature in celsius)
    
    currents = [TEC_electrical_load_mirror(Qh[i],T[i],E[i],T_a)[0] for i,n in enumerate(Qh)]
    power = [TEC_electrical_load_mirror(Qh[i],T[i],E[i],T_a)[1] for i,n in enumerate(Qh)]
    
    return currents,power
    

def Full_surface_correction(filename,T_a,n): #comprehensive function to solve for TEC currents.  Input (4D map file, ambient temperature in celsius, number of Zernike terms to correct)
    
    W = load_influence_function_matrix()

    Z = load_zernike_matrix()

    TEC_locs = import_TEC_centroids()
    
    thermal_c = import_thermal_conductance()

    M = import_4D_map_auto(filename, Z)

    Z_C = Zernike_decomposition(Z,M,n)

    H = Heat_loads2(W,Z_C,T_a)

    TEC_temps = get_TEC_temperatures(TEC_locs,H[3])

    current = get_electrical_output(TEC_locs,H[0],TEC_temps,thermal_c,T_a)[0]
    
    power = get_electrical_output(TEC_locs,H[0],TEC_temps,thermal_c,T_a)[1]
    
    return current,power,H[0]


def iterative_correction(filename,T_a,n,H_prev): #iterative correction function.  Input (4D map file, ambient temperature in celsius, number of Zernike modes to correct, previous set of heat loads)
    
    W = load_influence_function_matrix()

    Z = load_zernike_matrix()

    TEC_locs = import_TEC_centroids()
    
    thermal_c = import_thermal_conductance()

    M = import_4D_map_auto(filename, Z)

    Z_C = Zernike_decomposition(Z,M,n)

    H = Heat_loads2(W,Z_C,T_a)
    
    H_net = H_prev + H[0]
    
    TEC_temp_map = np.dot(W[3],H_net) + T_a
    
    TEC_temps = get_TEC_temperatures(TEC_locs,TEC_temp_map)

    current = get_electrical_output(TEC_locs,H_net,TEC_temps,thermal_c,T_a)[0]
   
    power = get_electrical_output(TEC_locs,H_net,TEC_temps,thermal_c,T_a)[1]
    
    return current,power,H_net

def write_eigenvalues_to_csv(write_path,eigenvalues):
    data = [['TEC','cmd','enabled']]
    for i in range(len(eigenvalues)):
        data.append([i+1,eigenvalues[i],1])

    with open(write_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for line in data:
            csvwriter.writerow(line)


# =============================================================================
# BACKWARD COMPATIBILITY IMPORTS
# The following functions have been moved to mirror_control/shared/zernike_utils.py
# These imports maintain compatibility for existing code
# =============================================================================

import warnings
import os
import sys

def _import_from_shared_utils():
    """
    Import functions from the shared zernike utilities with proper path resolution.
    """
    try:
        # Try relative import from mirror_control submodule first
        mirror_control_path = os.path.join(os.path.dirname(__file__), '..', 'mirror_control', 'shared')
        if os.path.exists(mirror_control_path) and mirror_control_path not in sys.path:
            sys.path.insert(0, mirror_control_path)
        
        from mirror_control.shared.zernike_utils import Zernike_decomposition as _Zernike_decomposition
        from mirror_control.shared.zernike_utils import remove_modes as _remove_modes
        return _Zernike_decomposition, _remove_modes
        
    except ImportError:
        # Fallback: try to import from mirror_control if it's in the same parent directory
        try:
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            mirror_control_shared = os.path.join(parent_dir, 'mirror_control', 'shared')
            if os.path.exists(mirror_control_shared) and mirror_control_shared not in sys.path:
                sys.path.insert(0, mirror_control_shared)
            
            from mirror_control.shared.zernike_utils import Zernike_decomposition as _Zernike_decomposition
            from mirror_control.shared.zernike_utils import remove_modes as _remove_modes
            return _Zernike_decomposition, _remove_modes
            
        except ImportError:
            raise ImportError(
                "Cannot import Zernike functions. Please ensure mirror_control/shared/zernike_utils.py "
                "is available. Functions have been moved from LFAST_TEC_output.py to the shared utilities."
            )

def _import_surface_processing_functions():
    """
    Import functions from the interferometer surface processing utilities.
    """
    try:
        # First, find the workspace root (repos directory)
        current_file = Path(__file__)
        workspace_root = None
        
        # Look for the repos directory by finding mirror_control
        for path in [current_file.parent] + list(current_file.parents):
            if (path / 'mirror_control').exists():
                workspace_root = str(path)
                break
        
        if workspace_root:
            # Add workspace root to path if not already there
            if workspace_root not in sys.path:
                sys.path.insert(0, workspace_root)
        
        from mirror_control.interferometer.surface_processing import (
            import_4D_map as _import_4D_map,
            import_4D_map_auto as _import_4D_map_auto,
            import_cropped_4D_map as _import_cropped_4D_map,
            measure_h5_circle as _measure_h5_circle,
            continuous_pupil_merit_function as _continuous_pupil_merit_function,
            define_pupil_using_optimization as _define_pupil_using_optimization,
            define_ID as _define_ID,
            format_data_from_avg_circle as _format_data_from_avg_circle,
            initial_crop as _initial_crop
        )
        return (_import_4D_map, _import_4D_map_auto, _import_cropped_4D_map, _measure_h5_circle,
                _continuous_pupil_merit_function, _define_pupil_using_optimization, _define_ID,
                _format_data_from_avg_circle, _initial_crop)
        
    except ImportError:
        # Fallback: try to import from mirror_control if it's in the same parent directory
        try:
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            interferometer_path = os.path.join(parent_dir, 'mirror_control', 'interferometer')
            if os.path.exists(interferometer_path) and interferometer_path not in sys.path:
                sys.path.insert(0, interferometer_path)
            
            from mirror_control.interferometer.surface_processing import (
                import_4D_map as _import_4D_map,
                import_4D_map_auto as _import_4D_map_auto,
                import_cropped_4D_map as _import_cropped_4D_map,
                measure_h5_circle as _measure_h5_circle,
                continuous_pupil_merit_function as _continuous_pupil_merit_function,
                define_pupil_using_optimization as _define_pupil_using_optimization,
                define_ID as _define_ID,
                format_data_from_avg_circle as _format_data_from_avg_circle,
                initial_crop as _initial_crop
            )
            return (_import_4D_map, _import_4D_map_auto, _import_cropped_4D_map, _measure_h5_circle,
                    _continuous_pupil_merit_function, _define_pupil_using_optimization, _define_ID,
                    _format_data_from_avg_circle, _initial_crop)
            
        except ImportError:
            raise ImportError(
                "Cannot import surface processing functions. Please ensure mirror_control/interferometer/surface_processing.py "
                "is available. Functions have been moved from LFAST_TEC_output.py to the interferometer module."
            )

# Import the functions with backward compatibility
_zernike_decomposition_func, _remove_modes_func = _import_from_shared_utils()
(_import_4D_map_func, _import_4D_map_auto_func, _import_cropped_4D_map_func, _measure_h5_circle_func,
 _continuous_pupil_merit_function_func, _define_pupil_using_optimization_func, _define_ID_func,
 _format_data_from_avg_circle_func, _initial_crop_func) = _import_surface_processing_functions()

def Zernike_decomposition(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/shared/zernike_utils.py
    Please update your imports to use: from mirror_control.shared.zernike_utils import Zernike_decomposition
    """
    warnings.warn(
        "Zernike_decomposition in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.shared.zernike_utils instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _zernike_decomposition_func(*args, **kwargs)

def remove_modes(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/shared/zernike_utils.py
    Please update your imports to use: from mirror_control.shared.zernike_utils import remove_modes
    """
    warnings.warn(
        "remove_modes in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.shared.zernike_utils instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _remove_modes_func(*args, **kwargs)

# =============================================================================
# SURFACE PROCESSING BACKWARD COMPATIBILITY FUNCTIONS
# The following functions have been moved to mirror_control/interferometer/surface_processing.py
# =============================================================================

def import_4D_map(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import import_4D_map
    """
    warnings.warn(
        "import_4D_map in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _import_4D_map_func(*args, **kwargs)

def import_4D_map_auto(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import import_4D_map_auto
    """
    warnings.warn(
        "import_4D_map_auto in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _import_4D_map_auto_func(*args, **kwargs)

def import_cropped_4D_map(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import import_cropped_4D_map
    """
    warnings.warn(
        "import_cropped_4D_map in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _import_cropped_4D_map_func(*args, **kwargs)

def measure_h5_circle(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import measure_h5_circle
    """
    warnings.warn(
        "measure_h5_circle in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _measure_h5_circle_func(*args, **kwargs)

def continuous_pupil_merit_function(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import continuous_pupil_merit_function
    """
    warnings.warn(
        "continuous_pupil_merit_function in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _continuous_pupil_merit_function_func(*args, **kwargs)

def define_pupil_using_optimization(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import define_pupil_using_optimization
    """
    warnings.warn(
        "define_pupil_using_optimization in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _define_pupil_using_optimization_func(*args, **kwargs)

def define_ID(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import define_ID
    """
    warnings.warn(
        "define_ID in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _define_ID_func(*args, **kwargs)

def format_data_from_avg_circle(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import format_data_from_avg_circle
    """
    warnings.warn(
        "format_data_from_avg_circle in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _format_data_from_avg_circle_func(*args, **kwargs)

def initial_crop(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/interferometer/surface_processing.py
    Please update your imports to use: from mirror_control.interferometer.surface_processing import initial_crop
    """
    warnings.warn(
        "initial_crop in LFAST_TEC_output.py is deprecated. "
        "Please import from mirror_control.interferometer.surface_processing instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _initial_crop_func(*args, **kwargs)