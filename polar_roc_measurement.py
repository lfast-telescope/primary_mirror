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
from LFAST_wavefront_utils import *
#%%

def polar_roc_measurement(csv_file, title = 'M1N10 after x hours', spherometer_diameter = 11.5, object_diameter=32, measurement_radius = [11.875, 8.5, 5.25, 2], number_of_pixels = 100, crop_clear_aperture = True, concave = True, output_plots = True, plot_label = 'Radius of curvature (spec=5275mm)'):
#   csv_path: path to csv file with format shown in 20.35/LFAST_MirrorTesting/M10
#   title: for output plot
#   number_of_pixels: size of computed array
#   crop_clear_aperture: Boolean. If true, output is cropped using ID=4" and OD=30"
#   concave : Boolean. Changes roc measurement calculation based on spherometer contact points.
#   output_plots: Boolean. Set to false to suppress plotting.
#   All measurements can be any units that is consistent with sag values. Default values for spherometer_diameter, object_diameter, measurement_radius are inches.
    
    cropped_data, smoothed_data, mirror_extent = process_spherometer_concentric(csv_file, measurement_radius=measurement_radius, spherometer_diameter=spherometer_diameter, object_diameter=object_diameter, number_of_pixels=number_of_pixels, crop_clear_aperture=crop_clear_aperture)

    if concave:
        roc = 25.4 * np.divide(11.5 ** 2 / 4 + np.power(cropped_data, 2), 2 * cropped_data) + 0.125 / 2
    else:
        roc = 25.4 * np.divide(11.5 ** 2 / 4 + np.power(cropped_data, 2), 2 * cropped_data) - 0.125 / 2

    if output_plots:
        plt.imshow(np.flip(roc,0), cmap='viridis')
        plt.title(title + ' grind has mean ROC=' + str(int(np.nanmean(roc))) + 'mm' , x=0.65)
        plt.colorbar(label=plot_label)
        plt.contour(smoothed_data, colors='k', alpha=0.35, levels=6)
    
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    return np.flip(roc,0)