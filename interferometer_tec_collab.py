import time
import datetime
from interferometer_utils import *
import numpy as np
from LFAST_wavefront_utils import *
from General_zernike_matrix import *
from LFASTfiber.libs.libNewport import smc100
from LFASTfiber.libs import libThorlabs


number_test_steps = 50
number_frames_avg = 30
number_averaged_frames = 5

gain = 0.5
#%%
s = smc100('COM3',nchannels=3)
#%%
remove_coef = [0]
in_to_m = 25.4e-3
OD = 31.9*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.47*OD
clear_aperture_inner = ID
s_gain = 0.5

Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

mirror_path = 'C:/Users/lfast-admin/Documents/mirrors/M8/'
folder_name = datetime.datetime.now().strftime('%Y%m%d')
if not os.path.exists(mirror_path + folder_name): os.mkdir(mirror_path + folder_name)
folder_path = mirror_path + folder_name + '/'
#%%
for i in np.arange(10):
    s.setPositionAbs(0,3)
    s.setPositionAbs(25,3)



#start_alignment(3,number_frames_avg,s,s_gain)

#%%
tic = time.time()
for num in np.arange(number_averaged_frames):
    take_interferometer_measurements(folder_path, num_avg=number_frames_avg, onboard_averaging=True,savefile=str(num+1))

test_duration = time.time() - tic

#%%
input('Start TEC test')

list_of_tec_tests = os.listdir(folder_path)

i = -1
current_test = list_of_tec_tests[i]
test_path = folder_path + current_test + '/'
while not os.path.isdir(test_path):
    i = i-1
    current_test = list_of_tec_tests[i]
    test_path = folder_path + current_test + '/'

for i in np.arange(1,51):
    step_path = test_path + str(i) + '/'
    while not os.path.exists(step_path):
        print('Now running step ' + str(i))
        time.sleep(1)
    tic = time.time()
    txt = open(step_path + 'step_info.txt','r').read()
    txt_words = txt.split(' ')
    duration = int(txt_words[3])
    tec_num = int(txt_words[7][:-1])
    tec_cmd = float(txt_words[9])
    savefile = 'tec' + str(tec_num) + '_cmd' + txt_words[9].replace('.','-')
    keep_running = True
    while keep_running:
        start_alignment(1, number_frames_avg, s, s_gain)
        time.sleep(3)
        if time.time() - tic > duration - test_duration * 1.6:
            keep_running = False

    for num in np.arange(number_averaged_frames):
        take_interferometer_measurements(step_path, num_avg=number_frames_avg, onboard_averaging=True, savefile=savefile + '_' + str(num))
