import primary_mirror.LFAST_TEC_output
import pandas as pd
import time
from datetime import datetime as dt
# import os


# file_prefix = '230608'
file_prefix = 'uninit'


TEC_FULL_RANGE_CURRENT = 0.8

def get_file_date_prefix():
    t_now = time.time()
    dt_obj = dt.fromtimestamp(t_now)
    year = str(dt_obj.year)[-2:]
    month = str(dt_obj.month).zfill(2)
    day = str(dt_obj.day).zfill(2)
    date_string = f'{year}{month}{day}'
    return date_string

def update_file_prefix(suffix=''):
    global file_prefix
    file_prefix = f'{get_file_date_prefix()}{suffix}'


def save_TEC_table(TEC_locs, output_path = None,file_name = None):
    df = pd.DataFrame(TEC_locs)
    if output_path == None:
        file_path = f'TEC_locs_save.csv'
    else:
        file_path = output_path + file_name + '.csv'
    df.to_csv(path_or_buf=file_path, index=False,lineterminator='\n')

def save_to_csv(tec_percentages, output_path = None,file_name = None):
    tec_i = list( range(1, len(tec_percentages)+1) )
    dict = {'TEC': tec_i, 'cmd': tec_percentages,'enabled':1}
    df = pd.DataFrame(dict)
    if output_path == None:
        file_path = f'Electrical_load.csv'
    else:
        file_path = output_path + file_name + '.csv'
    df.to_csv(path_or_buf=file_path, index=False,lineterminator='\n')


def currents_to_percent(currents):
    tec_percentages = []
    for tec_current in currents:
        percentage = ((tec_current * 100) / TEC_FULL_RANGE_CURRENT) * -1
        tec_percentages.append(percentage)
    return tec_percentages


def run(wavefront_file, t_amb, iteration, heat_loads_prev=None):
    """
    Helper function that calls the surface correction functions from the LFAST Thermal Control Model.

    Saves the output currents, formatted as percentage of the TEC Full Scale current (0.8A), to a CSV file.
    Filename for the CSV file is in the form of 'tec_currents_{file_prefix}_i_{iteration}.csv', where:
        file_prefix: a text string in the form of YYMMDD from the current date,
        iteration: the iteration used as parameter on the function call.

    -- parameters:
        wavefront_file (string): filename for h5 file containing wavefront map.
        t_amb: (float): ambient temperature.
        iteration: (integer): current iteration of the surface correction, start at 0.
        heat_loads_prev: (list): calculated heat loads from the previous iteration. Use when itertation>0.
    -- returns:
        currents: (list): calculated currents for the 132 TECs.
        heat_loads: (list): calculated heat loads for the 132 TECs.
    """
    if iteration == 0:
        (currents, electrical_power, heat_loads ) = LFAST_TEC_output.Full_surface_correction(wavefront_file, t_amb, 10)
        percentages = currents_to_percent(currents)
        save_to_csv(percentages, iteration)
    else:
        (currents, electrical_power, heat_loads ) = LFAST_TEC_output.iterative_correction(wavefront_file, t_amb, 10, heat_loads_prev)
        percentages = currents_to_percent(currents)
        save_to_csv(percentages, iteration)
    return currents, heat_loads


def main():
    print('This file is meant to be used as an import on a Python session, not to be run as a standalone file.')


if __name__ == "__main__":
    main()