# =============================================================================
# BACKWARD COMPATIBILITY FOR General_zernike_matrix
# This module has been moved to mirror_control/shared/General_zernike_matrix.py
# This file maintains compatibility for existing imports
# =============================================================================

import warnings
import os
import sys

def _import_from_shared_zernike_matrix():
    """
    Import functions from the shared General_zernike_matrix module with proper path resolution.
    """
    try:
        # Try relative import from mirror_control submodule first
        mirror_control_path = os.path.join(os.path.dirname(__file__), '..', 'mirror_control', 'shared')
        if os.path.exists(mirror_control_path) and mirror_control_path not in sys.path:
            sys.path.insert(0, mirror_control_path)
        
        from General_zernike_matrix import General_zernike_matrix as _General_zernike_matrix
        from General_zernike_matrix import save_zernike_matrix as _save_zernike_matrix
        return _General_zernike_matrix, _save_zernike_matrix
        
    except ImportError:
        # Fallback: try to import from mirror_control if it's in the same parent directory
        try:
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            mirror_control_shared = os.path.join(parent_dir, 'mirror_control', 'shared')
            if os.path.exists(mirror_control_shared) and mirror_control_shared not in sys.path:
                sys.path.insert(0, mirror_control_shared)
            
            from General_zernike_matrix import General_zernike_matrix as _General_zernike_matrix
            from General_zernike_matrix import save_zernike_matrix as _save_zernike_matrix
            return _General_zernike_matrix, _save_zernike_matrix
            
        except ImportError:
            raise ImportError(
                "Cannot import General_zernike_matrix functions. Please ensure mirror_control/shared/General_zernike_matrix.py "
                "is available. Functions have been moved from primary_mirror to the shared utilities."
            )

# Import the functions with backward compatibility
_general_zernike_matrix_func, _save_zernike_matrix_func = _import_from_shared_zernike_matrix()

def General_zernike_matrix(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/shared/General_zernike_matrix.py
    Please update your imports to use: from mirror_control.shared.General_zernike_matrix import General_zernike_matrix
    
    Parameters:
    maxTerm (int): highest order term
    R (float): disc radius of measurement 
    a_i (float): inner radius in microns of measurement
    grid_shape (int): grid resolution (default: 500)
    
    Returns:
    tuple: (Z_matrix.transpose(), Z_3D.transpose(1,2,0)) - 2D array and 3D array in tuple form
    """
    warnings.warn(
        "General_zernike_matrix in primary_mirror.General_zernike_matrix is deprecated. "
        "Please import from mirror_control.shared.General_zernike_matrix instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _general_zernike_matrix_func(*args, **kwargs)

def save_zernike_matrix(*args, **kwargs):
    """
    DEPRECATED: This function has been moved to mirror_control/shared/General_zernike_matrix.py
    Please update your imports to use: from mirror_control.shared.General_zernike_matrix import save_zernike_matrix
    
    Parameters:
    matrix: Zernike matrix to save as obj file
    """
    warnings.warn(
        "save_zernike_matrix in primary_mirror.General_zernike_matrix is deprecated. "
        "Please import from mirror_control.shared.General_zernike_matrix instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _save_zernike_matrix_func(*args, **kwargs)    
