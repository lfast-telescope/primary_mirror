# This is bad code, even by my standards, but I'm going to try reading mat files and turning them into TEC recommendations
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize, io
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
from plotting_utils import *
#%%
# Path to the folder of influence functions (delta maps)
tec_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M9/'
mat_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/on-sky/20241204/SHWFS/'

# Mirror parameters
in_to_m = 25.4e-3
OD = 30 * in_to_m  # Outer mirror diameter (m)
ID = 3 * in_to_m  # Central obscuration diameter (m)
clear_aperture_outer = 0.47 * OD
clear_aperture_inner = ID / 2

#Interferometer parameters
number_frames_avg = 30
number_averaged_frames = 5

eigenvectors = np.load(tec_path + 'eigenvectors.npy')

eigenvector_cmd_ref = -0.5 #This was changed from positive 0.5 because it looks like SHWFS positive is 4D negative
#%%
mat_list = os.listdir(mat_path)
reference = 0
for file in mat_list:
    if file.endswith('.mat'):
        val = io.loadmat(mat_path + file)
        if file.endswith('average_0209.mat'):
            reference = val['output_val'].copy()
        plt.imshow(1e3*(val['output_val']-reference))
        plt.title(file)
        plt.colorbar()
        plt.show()
#%%
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

eigenvalues = [0 for i in range(24)]
eigenvalue_bounds = None

tec_grid = make_pupil_grid(eigenvectors[0].shape,OD)

sh_grid = make_pupil_grid(reference.shape,OD)
sh_field = Field(reference.ravel(),sh_grid)
inter = make_linear_interpolator_separated(sh_field)
sh_on_tec_grid = inter(tec_grid)
sampled_tec = np.array(np.reshape(sh_on_tec_grid, eigenvectors[0].shape))
reduced_surface, eigenvalue_delta = optimize_TECs(sampled_tec*1e3, eigenvectors, eigenvalues, eigenvalue_bounds,
                                                  clear_aperture_outer, clear_aperture_inner, Z, metric='rms_neutral')


