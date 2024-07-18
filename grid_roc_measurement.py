# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:09:11 2024

@author: warre

Evaluate surface ROC using spherometer measurments
Measurements using grid that maps to squares on the lapping body
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib import cm
from scipy import ndimage
import csv
from hcipy import *
from plotting_utils import *
from LFAST_wavefront_utils import *

csv_file = 'C:/Users/warre/OneDrive/Documents/LFAST/lap/roc_0716.csv'

#%%

ideal_sag=0.076
cropped_data,smoothed_data,mirror_extent = process_spherometer_grid(csv_file)
curv = ['concave']

if curv == ['convex']:
    roc = 25.4*np.divide(11.5**2/4+np.power(cropped_data,2), 2*cropped_data) - 0.125/2    
    plt.imshow(cropped_data[:95,:95],cmap='viridis')#vmax = ideal_sag + sag_range, vmin = ideal_sag - sag_range)
    plt.colorbar(label = 'Radius of curvature (mm)')
    plt.contour(smoothed_data[:95,:95],colors = 'k',alpha=0.35,levels = 6)
else:
    roc = 25.4*np.divide(11.5**2/4+np.power(cropped_data,2), 2*cropped_data) + 0.125/2
    plt.imshow(roc[4:96,4:96],cmap='magma')
    plt.colorbar(label = 'Radius of curvature (mm)')
    plt.contour(smoothed_data[4:96,4:96],colors = 'k',alpha=0.35,levels = 6)
# Plot data using colormap showing error
sag_error = smoothed_data[mirror_extent] - ideal_sag
sag_range = np.max([np.nanmax(np.abs(sag_error)), np.nanmin(np.abs(sag_error))])

plt.title('29" concave lapping tool ROC 7/16',x=0.65)

plt.xticks([])
plt.yticks([])
plt.show()