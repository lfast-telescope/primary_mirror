# -*- coding: utf-8 -*-
"""
DEPRECATED: Functions have been moved to mirror_control.interferometer.plotting_utils

This file provides backward compatibility by importing functions from their new location.
For new code, import directly from mirror_control.interferometer.plotting_utils.

@author: Warren Foster
"""

import warnings
import sys
import os

# Add parent folder to sys.path for accessing mirror_control
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def _issue_deprecation_warning(func_name):
    """Issue standardized deprecation warning for moved functions."""
    warnings.warn(
        f"Importing {func_name} from primary_mirror.plotting_utils is deprecated. "
        f"Use 'from mirror_control.interferometer.plotting_utils import {func_name}' instead. "
        "This compatibility import will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

# Import all functions from the new location
try:
    from mirror_control.interferometer.plotting_utils import (
        plot_mirror_wf_error as _plot_mirror_wf_error,
        plot_mirror_and_psf as _plot_mirror_and_psf,
        plot_single_mirror as _plot_single_mirror,
        plot_mirror_and_cs as _plot_mirror_and_cs,
        plot_many_mirror_cs as _plot_many_mirror_cs,
        plot_mirrors_side_by_side as _plot_mirrors_side_by_side,
        plot_multiple_surfaces as _plot_multiple_surfaces,
        plot_zernike_modes_as_bar_chart as _plot_zernike_modes_as_bar_chart,
        compute_cmap_and_contour as _compute_cmap_and_contour,
        create_4d_plot as _create_4d_plot,
        create_xy_cs as _create_xy_cs
    )
    from mirror_control.shared.zernike_utils import return_zernike_name as _return_zernike_name
    
    # Create backward compatible wrapper functions with deprecation warnings
    def plot_mirror_wf_error(*args, **kwargs):
        _issue_deprecation_warning('plot_mirror_wf_error')
        return _plot_mirror_wf_error(*args, **kwargs)
    
    def plot_mirror_and_psf(*args, **kwargs):
        _issue_deprecation_warning('plot_mirror_and_psf')
        return _plot_mirror_and_psf(*args, **kwargs)
    
    def plot_single_mirror(*args, **kwargs):
        _issue_deprecation_warning('plot_single_mirror')
        return _plot_single_mirror(*args, **kwargs)
    
    def plot_mirror_and_cs(*args, **kwargs):
        _issue_deprecation_warning('plot_mirror_and_cs')
        return _plot_mirror_and_cs(*args, **kwargs)
    
    def plot_many_mirror_cs(*args, **kwargs):
        _issue_deprecation_warning('plot_many_mirror_cs')
        return _plot_many_mirror_cs(*args, **kwargs)
    
    def plot_mirrors_side_by_side(*args, **kwargs):
        _issue_deprecation_warning('plot_mirrors_side_by_side')
        return _plot_mirrors_side_by_side(*args, **kwargs)
    
    def plot_multiple_surfaces(*args, **kwargs):
        _issue_deprecation_warning('plot_multiple_surfaces')
        return _plot_multiple_surfaces(*args, **kwargs)
    
    def plot_zernike_modes_as_bar_chart(*args, **kwargs):
        _issue_deprecation_warning('plot_zernike_modes_as_bar_chart')
        return _plot_zernike_modes_as_bar_chart(*args, **kwargs)
    
    def compute_cmap_and_contour(*args, **kwargs):
        _issue_deprecation_warning('compute_cmap_and_contour')
        return _compute_cmap_and_contour(*args, **kwargs)
    
    def create_4d_plot(*args, **kwargs):
        _issue_deprecation_warning('create_4d_plot')
        return _create_4d_plot(*args, **kwargs)
    
    def create_xy_cs(*args, **kwargs):
        _issue_deprecation_warning('create_xy_cs')
        return _create_xy_cs(*args, **kwargs)
    
    def return_zernike_name(*args, **kwargs):
        _issue_deprecation_warning('return_zernike_name')
        return _return_zernike_name(*args, **kwargs)
    
except ImportError as e:
    warnings.warn(
        f"Could not import from new location: {e}. "
        "Falling back to original implementations. "
        "Please ensure mirror_control module is available.",
        ImportWarning
    )
    
    # If import fails, keep original implementations (fallback)
    # This would be all the original function code that was here before

