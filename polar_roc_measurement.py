# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26

@author: warrenbfoster

Evaluate surface ROC using spherometer measurments
Measurements using polar grid that maps surface

Assumes a csv file that passes information IN ROWS
Where each row holds equally spaced measurements of a certain radius
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
#%%

pressing = False

if not pressing:
    csv_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M10/'
    csv_filename = ['roc_0726.csv','roc_0730.csv','roc_0801.csv','roc_0802.csv','roc_0806.csv','roc_0807.csv','roc_0808.csv','roc_0809.csv','roc_0812.csv','roc_0813.csv','roc_0814.csv']
    #Doing everything with generic units

    spherometer_diameter=11.5
    object_diameter=32
    number_of_pixels = 100
    measurement_radius = [11.875, 8.5, 5.25, 2]
    overfill = 0
    curv = 'concave'

    title_1 = 'M1N10 ROC '
    title_2 = ['before', 'after 2hrs', 'after 7hrs', 'after 12hrs','after 15hrs','after 19hrs','after 24 hrs','after 29 hrs','after 33 hrs','after 37 hrs','after 42 hrs']
    title_3 = ' grind, spec=5275mm'

else:
    csv_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/pressing/'
    csv_filename = ['roc_0809.csv']
    # Doing everything with generic units
    spherometer_diameter = 11.5
    object_diameter = 37
    number_of_pixels = 100
    measurement_radius = [14.3125,11.4375,8.375,5.4375,2.43]
    overfill = 0
    curv = 'concave'

    title_1 = 'Pressing body '
    title_2 = ['before']
    title_3 = ' grind, spec=5275mm'

#%%
for i in np.arange(0,len(csv_filename)):
    csv_file = csv_path + csv_filename[i]
    cropped_data, smoothed_data, mirror_extent = process_spherometer_concentric(csv_file, measurement_radius=measurement_radius, crop_clear_aperture = True)

    if curv == 'convex':
        roc = 25.4 * np.divide(11.5 ** 2 / 4 + np.power(cropped_data, 2), 2 * cropped_data) - 0.125 / 2
    else:
        roc = 25.4 * np.divide(11.5 ** 2 / 4 + np.power(cropped_data, 2), 2 * cropped_data) + 0.125 / 2

    if i==0:
        bounds = [np.nanmin(roc), np.nanmax(roc)]

    plt.imshow(np.flip(roc,0), cmap='viridis')
    plt.title(title_1 + title_2[i] + ' grind has mean ROC=' + str(int(np.nanmean(roc))) + 'mm' , x=0.65)
    plt.colorbar(label='Radius of curvature (spec=5275mm)')
    plt.contour(smoothed_data, colors='k', alpha=0.35, levels=6)

    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.show()
