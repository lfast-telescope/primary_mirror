import os
import time
import requests
import datetime

def take_interferometer_measurements(path,num_avg=10, onboard_averaging=True, savefile = None):
    current_time = datetime.datetime.now().strftime('%H%M%S')
    if savefile is None:
        savefile = current_time
    if onboard_averaging:
        tic = time.time()
        payload = {"analysis": "analyzed", "fileName": path + savefile, "count": str(num_avg)}
        meas = requests.get('http://localhost/WebService4D/WebService4D.asmx/AverageMeasure', params=payload)
        sav = requests.get('http://localhost/WebService4D/WebService4D.asmx/SaveArray', params=payload)
        print(time.time()-tic, ' seconds to measure, analyze + save')
    else:
        time_folder = mirror_path + folder_name + '/' + savefile + '/'
        os.mkdir(time_folder)
        tic = time.time()
        for i in np.arange(1,11):
            payload = {"analysis": "analyzed", "fileName": time_folder + str(i)}
            meas = requests.get('http://localhost/WebService4D/WebService4D.asmx/Measure', params=payload)
            sav = requests.get('http://localhost/WebService4D/WebService4D.asmx/SaveArray', params=payload)
        print(time.time()-tic, ' seconds to measure, analyze + save')

