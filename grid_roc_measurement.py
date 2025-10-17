# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 11:09:11 2024

@author: warre

Evaluate surface ROC using spherometer measurments
Measurements using grid that maps to squares on the lapping body`
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

#csv_file = 'C:/Users/warre/OneDrive/Documents/LFAST/lap/roc_0716.csv'
csv_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/cv_iron/'
list_cv_names = os.listdir(csv_path)
#%%
for csv_name in list_cv_names:
    csv_file = csv_path + csv_name
    if csv_file.endswith('.csv'):
        date = csv_name.split('_')[0]
        sph_size = csv_name.split('_')[1].split('.csv')[0]

        ideal_sag=0.076
        cropped_data,smoothed_data,mirror_extent = process_spherometer_grid(csv_file, spherometer_diameter=float(sph_size), pixels_per_square=20)
        curv = ['concave']

        cropped_data = cropped_data / 25.4

        if curv == ['convex']:
            roc = 25.4*np.divide(11.5**2/4+np.power(cropped_data,2), 2*cropped_data) - 0.125/2
            plt.imshow(cropped_data,cmap='viridis')#vmax = ideal_sag + sag_range, vmin = ideal_sag - sag_range)
            plt.colorbar(label = 'Radius of curvature (mm)')
            plt.contour(smoothed_data[:95,:95],colors = 'k',alpha=0.35,levels = 6)
        else:
            if sph_size == '12':
                roc = 25.4*np.divide(float(sph_size)**2/4+np.power(cropped_data,2), 2*cropped_data) + 0.25/2
            elif sph_size == '16':
                roc = 25.4 * np.divide(float(sph_size) ** 2 / 4 + np.power(cropped_data, 2), 2 * cropped_data) + 0.375 / 2
            mean_roc = np.nanmean(roc)
            plt.imshow(roc,cmap='viridis_r',vmin=5245,vmax=5305)
            plt.colorbar(label = 'Radius of curvature (mm)')
            plt.contour(smoothed_data,colors = 'k',alpha=0.35,levels = 6)
        # Plot data using colormap showing error
        sag_error = smoothed_data[mirror_extent] - ideal_sag
        sag_range = np.max([np.nanmax(np.abs(sag_error)), np.nanmin(np.abs(sag_error))])

        plt.title('Concave tool measured with ' + sph_size + 'in sph. has mean ROC=' + str(round(mean_roc)) +'mm', x=0.6)

        plt.xticks([])
        plt.yticks([])
        plt.show()