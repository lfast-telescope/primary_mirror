import time
import datetime
from interferometer_utils import *
import numpy as np

number_test_steps = 72
number_frames_avg = 30
number_averaged_frames = 5

mirror_path = 'C:/Users/lfast-admin/Documents/mirrors/M9/'
folder_name = datetime.datetime.now().strftime('%Y%m%d')
if not os.path.exists(mirror_path + folder_name): os.mkdir(mirror_path + folder_name)
folder_path = mirror_path + folder_name + '/'

tic = time.time()
take_interferometer_measurements(folder_path, num_avg=number_frames_avg, onboard_averaging=True)
test_duration = time.time() - tic

input('Start TEC test')

list_of_tec_tests = os.listdir(folder_path)
current_test = list_of_tec_tests[-1]
test_path = folder_path + current_test + '/'

for i in [69]:
    step_path = test_path + str(i) + '/'
    while not os.path.exists(step_path):
        print('Now running step ' + str(i))
        time.sleep(1)
    txt = open(step_path + 'step_info.txt','r').read()
    txt_words = txt.split(' ')
    duration = int(txt_words[3])
    tec_num = int(txt_words[7][:-1])
    tec_cmd = float(txt_words[9])
    time.sleep(duration - test_duration*number_averaged_frames*2)
    savefile = 'tec' + str(tec_num) + '_cmd' + txt_words[9].replace('.','-')
    for num in np.arange(number_averaged_frames):
        take_interferometer_measurements(step_path, num_avg=number_frames_avg, onboard_averaging=True, savefile=savefile + '_' + str(num))
