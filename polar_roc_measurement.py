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


# %%

def polar_roc_measurement(csv_file, title='M1N10 after x hours', spherometer_diameter=11.5, object_diameter=32,
                          measurement_radius=[11.875, 8.5, 5.25, 2], number_of_pixels=100, crop_clear_aperture=True,
                          concave=True, output_plots=True, plot_label='Radius of curvature (spec=5275mm)'):
    #   csv_path: path to csv file with format shown in 20.35/LFAST_MirrorTesting/M10
    #   title: for output plot
    #   number_of_pixels: size of computed array
    #   crop_clear_aperture: Boolean. If true, output is cropped using ID=4" and OD=30"
    #   concave : Boolean. Changes roc measurement calculation based on spherometer contact points.
    #   output_plots: Boolean. Set to false to suppress plotting.
    #   All measurements can be any units that is consistent with sag values. Default values for spherometer_diameter, object_diameter, measurement_radius are inches.

    cropped_data, smoothed_data, mirror_extent = process_spherometer_concentric(csv_file,
                                                                                measurement_radius=measurement_radius,
                                                                                spherometer_diameter=spherometer_diameter,
                                                                                object_diameter=object_diameter,
                                                                                number_of_pixels=number_of_pixels,
                                                                                crop_clear_aperture=crop_clear_aperture)

    if concave:
        roc = 25.4 * np.divide(spherometer_diameter ** 2 / 4 + np.power(smoothed_data, 2), 2 * smoothed_data) + 0.125 / 2
    else:
        roc = 25.4 * np.divide(spherometer_diameter ** 2 / 4 + np.power(smoothed_data, 2), 2 * smoothed_data) - 0.125 / 2

    if output_plots:
        plt.imshow(np.flip(roc, 0), cmap='viridis')
        plt.title(title + ' grind has mean ROC=' + str(int(np.nanmean(roc))) + 'mm', x=0.65)
        plt.colorbar(label=plot_label)
        plt.contour(smoothed_data, colors='k', alpha=0.35, levels=6)

        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return np.flip(roc, 0)

pressing = True
thirty = False
spherometer_16 = True

if pressing:
    file_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/pressing/'
    measurement_radius=[12.5, 9.125, 5.75, 2.375]
    spherometer_diameter=16
    object_diameter=37
    crop_clear_aperture = False
    file_suffix = ['roc_1016.csv','roc_1017.csv']
    hours_list = [0,5]

    title = 'Pressing body after '
elif thirty:
    file_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/pressing/'
    measurement_radius=[6,2]
    spherometer_diameter=16
    object_diameter=30
    crop_clear_aperture = True
    concave = False

else:
    file_path = 'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M6/'

    if spherometer_16:
        measurement_radius=[10, 6, 2]
        spherometer_diameter=16
        object_diameter = 32
        crop_clear_aperture = True

    hours_list = [20,40,45,50,52.5,53.5,56.5,59,61,62,64.5,67,69.5,70.5]
    file_suffix = ['roc_0917.csv','roc_0927.csv','roc_0930.csv','roc_1001.csv','roc_1002.csv','roc_1004.csv','roc_1007.csv','roc_1008.csv','roc_1009.csv','roc_1010.csv','roc_1010b.csv','roc_1011.csv','roc_1014.csv','roc_1014b.csv']
    #title = 'M1N6 before'

for num, file in enumerate(file_suffix):

  title = 'Pressing body before '
  val = polar_roc_measurement(file_path + file, title=title + str(hours_list[num]) + 'hrs', measurement_radius=measurement_radius,spherometer_diameter=spherometer_diameter, object_diameter=object_diameter, crop_clear_aperture=crop_clear_aperture,number_of_pixels=256)

# file = 'roc_convex_30in_0930.csv'
# title = '30in glass on convex side'
# val = polar_roc_measurement(file_path + file, title=title, measurement_radius=measurement_radius,spherometer_diameter=spherometer_diameter, object_diameter=object_diameter, crop_clear_aperture=crop_clear_aperture)