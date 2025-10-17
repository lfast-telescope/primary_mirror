"""
Interferometer utilities - DEPRECATED

This module has been moved to mirror-control/interferometer/interferometer_utils.py
These imports are provided for backward compatibility only.

For new code, use:
    from mirror_control.interferometer.interferometer_utils import take_interferometer_measurements

WARNING: This compatibility layer will be removed in a future version.
"""

import warnings
import os
import sys

def _deprecation_warning(func_name):
    """Issue deprecation warning for moved functions."""
    warnings.warn(
        f"Importing {func_name} from primary_mirror.interferometer_utils is deprecated. "
        f"Use 'from mirror_control.interferometer.interferometer_utils import {func_name}' instead. "
        "This compatibility import will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

def take_interferometer_measurements(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils instead."""
    _deprecation_warning('take_interferometer_measurements')
    
    # Add the mirror-control path to sys.path if not already there
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from interferometer.interferometer_utils import take_interferometer_measurements as _func
    return _func(*args, **kwargs)

def take_interferometer_coefficients(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils instead."""
    _deprecation_warning('take_interferometer_coefficients')
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from interferometer.interferometer_utils import take_interferometer_coefficients as _func
    return _func(*args, **kwargs)

def correct_tip_tilt_power(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils instead."""
    _deprecation_warning('correct_tip_tilt_power')
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from interferometer.interferometer_utils import correct_tip_tilt_power as _func
    return _func(*args, **kwargs)

def hold_alignment(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils instead."""
    _deprecation_warning('hold_alignment')
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from interferometer.interferometer_utils import hold_alignment as _func
    return _func(*args, **kwargs)

def start_alignment(*args, **kwargs):
    """DEPRECATED: Use mirror_control.interferometer.interferometer_utils instead."""
    _deprecation_warning('start_alignment')
    
    mirror_control_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mirror-control')
    if mirror_control_path not in sys.path:
        sys.path.insert(0, mirror_control_path)
    
    from interferometer.interferometer_utils import start_alignment as _func
    return _func(*args, **kwargs)


