from LFAST_TEC_output import *
from LFAST_wavefront_utils import *
from plotting_utils import *
from General_zernike_matrix import *
import matplotlib.pyplot as plt
#%%
#Mirror parameters
in_to_m = 25.4e-3

OD = 32*in_to_m #Outer mirror diameter (m)
ID = 3*in_to_m #Central obscuration diameter (m)
clear_aperture_outer = 0.5*OD
clear_aperture_inner = 0.5*ID

remove_normal_coef = [0, 1, 2, 4] #Zernike coefficients to subtract

#Set up the Zernike fitting matrix to process the h5 files
Z = General_zernike_matrix(44,int(clear_aperture_outer * 1e6),int(clear_aperture_inner * 1e6))

#%% Load measurements sequentially to perform circle detection. The pupil is then defined based on the coordinate average of the set.
test1 = {"mirror_number":11,
         "file_path":'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M11/20250207/1/',
         "tecs_on":False
         }
test2 = {"mirror_number":10,
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M10/20250212/0/',
         "tecs_on":False
         }
test3 = {"mirror_number":11,
         "file_path":'C:/Users/warrenbfoster/OneDrive - University of Arizona/Documents/LFAST/mirrors/M11/20250205/0/',
         "tecs_on":False
         }
test4 = {"mirror_number":11,
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M11/20250220/0/',
         "tecs_on":False
         }
test5 = {"mirror_number":11,
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M11/20250224/1/',
         "tecs_on":False
         }
test6 = {"mirror_number":11,
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M11/20250225/0/',
         "tecs_on":False
         }
test7 = {"mirror_number":'_test_plate',
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M_test_plate/20250225/1/',
         "tecs_on":False
         }
test8 = {"mirror_number":10,
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M10/20250306/0/',
         "tecs_on":False
         }
test9 = {"mirror_number":10,
         "file_path":'C:/Users/lfast-admin/Documents/mirrors/M10/20250306/3/',
         "tecs_on":False
         }

test_suite = [test9]
#%%
for test in test_suite:
    save_subfolder = test['file_path']
    data_holder = []
    coord_holder = []
    ID_holder = []
    for file in os.listdir(save_subfolder):
        if file.endswith(".h5"):
            data, circle_coord, ID = measure_h5_circle(save_subfolder + file)
            data_holder.append(data)
            coord_holder.append(circle_coord)
            ID_holder.append(ID)

    avg_circle_coord = np.mean(coord_holder, axis=0)
    avg_ID = np.mean(ID_holder)
    test.update({"circle_coord":avg_circle_coord, "ID":avg_ID})

    #Based on the defined pupil, process the measurements
    increased_ID_crop = 1.25

    wf_maps = []

    for data in data_holder:
        wf_maps.append(format_data_from_avg_circle(data, avg_circle_coord, clear_aperture_outer, clear_aperture_inner*increased_ID_crop, Z, normal_tip_tilt_power=True)[1])

    surface = np.flip(np.mean(wf_maps, 0), 0)

    test.update({"surface":surface})

#%%
surface = test["surface"].copy() / 2
surface[surface < -0.065] = np.nan
X,Y = np.meshgrid(np.linspace(-OD/2,OD/2,surface.shape[0]),np.linspace(-OD/2,OD/2,surface.shape[0]))
distance_from_center = np.sqrt(np.square(X)+np.square(Y))
pupil_boolean = (distance_from_center < 7.85*25.4e-3)
surface[~pupil_boolean] = np.nan

vals = surface[~np.isnan(surface)]*1e3
sorted_vals = np.sort(vals)
sorted_index = int(0.0001 * len(sorted_vals))  # For peak-valley, throw out the extreme tails
pv = sorted_vals[-sorted_index] - sorted_vals[sorted_index]
rms = np.sqrt(np.sum(np.power(vals, 2)) / len(vals))

plot_single_mirror('Test plate has ' + str(round(pv)) + 'nm PV surface error',surface)

#%%
for test in test_suite:
    M,C = get_M_and_C(test["surface"], Z)

    coef_normal = [0, 1, 2, 4]
    coef_correctable = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 20]
    coef_radial = list(np.arange(45))
    for val in [12,24,40]:
        coef_radial.remove(val)

    surface_normal = remove_modes(M, C, Z, coef_normal)
    surface_correctable = remove_modes(M, C, Z, coef_correctable)
    surface_radial = remove_modes(M, C, Z, coef_radial)

    wf_foc, throughput, x_foc, y_foc = propagate_wavefront(surface_correctable, clear_aperture_outer, clear_aperture_inner, Z, use_best_focus=True)
    _, throughput_uncorrected, _, _ = propagate_wavefront(surface_normal, clear_aperture_outer, clear_aperture_inner, Z, use_best_focus=True)

    surfaces = [surface_normal, surface_radial]
    surface_titles = ['Measured wavefront','Radially symmetric terms']
    supergrid_rows = 3
    supergrid_cols = 2
    supergrid_col_width_ratios = [1,0.5]

    fig = plt.figure(figsize = (np.sum(supergrid_col_width_ratios)*4,supergrid_rows*4))
    gs0 = gridspec.GridSpec(supergrid_rows, supergrid_cols,height_ratios=[1]*supergrid_rows,width_ratios=[1,0.5])
    number_subgrids = 5
    for i in range(supergrid_rows):
        if i == 0 or i == 1:
            vals = surfaces[i][~np.isnan(surfaces[i])]
            sorted_vals = np.sort(vals)
            sorted_index = int(0.001*len(sorted_vals)) #For peak-valley, throw out the extreme tails
            pv = sorted_vals[-sorted_index]-sorted_vals[sorted_index]
            rms = np.sqrt(np.sum(np.power(vals,2))/len(vals))

            subplot_spec = gs0[i,0]
            create_subplot(fig,number_subgrids,subplot_spec,surfaces[i],surface_titles[i])
            subplot_spec_text = gs0[i,1]
            add_text_for_subplots(fig,1,subplot_spec_text,surfaces[i],pv,rms)
        else:
            subplot_spec_foc = gs0[i,0]
            subplot_spec_foc_text = gs0[i,1]
            add_PSF_for_subplot(fig,subplot_spec_foc,wf_foc,x_foc,y_foc)
            add_text_for_PSF_subplot(fig, subplot_spec_foc_text, throughput_uncorrected, throughput)

    date = test['file_path'].split('/')[-3]
    plt.suptitle('Mirror #' + str(test['mirror_number']) + ' on ' + date,y=0.95)
    plt.show()
#%%
def add_text_for_subplots(fig,number_subgrids,subplot_spec,surface,pv,rms):
    gs00 = gridspec.GridSpecFromSubplotSpec(number_subgrids, number_subgrids, subplot_spec=subplot_spec,width_ratios=[1]*number_subgrids,height_ratios=[1]*number_subgrids)
    ax = fig.add_subplot(gs00[0,0])
    ax.axis('off')
    if pv > 1:
        ax.text(0.1,0.6,'PV: ' + str(round(pv, 3)) + 'um')
    else:
        ax.text(0.1,0.6,'PV: ' + str(round(pv*1e3, 1)) + 'nm')
    if rms > 1:
        ax.text(0.1,0.5,'RMS: ' + str(round(rms, 3)) + 'um')
    else:
        ax.text(0.1,0.5,'RMS: ' + str(round(rms*1e3, 1)) + 'nm')

def create_subplot(fig,number_subgrids,subplot_spec,surface,surface_title):
    gs00 = gridspec.GridSpecFromSubplotSpec(number_subgrids, number_subgrids, subplot_spec=subplot_spec,width_ratios=[1]*number_subgrids,height_ratios=[1]*number_subgrids)
    ax1 = fig.add_subplot(gs00[:-1, :-1])
    ax2 = fig.add_subplot(gs00[-1, :-1])
    ax3 = fig.add_subplot(gs00[:-1,-1])
    ax_ghost = fig.add_subplot(gs00[0,:])
    ax_ghost.axis('off')
    ax_ghost.set_title(surface_title)
    x_cs, y_cs = create_xy_cs(surface)
    ax1.imshow(surface,cmap='jet')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.plot(np.linspace(-OD / 2, OD / 2, x_cs.size), x_cs)
    ax2.yaxis.grid(True)
    ax2.set_ylabel('um',loc='top',rotation='horizontal',labelpad=-8)
    ax3.plot(y_cs, np.linspace(-OD / 2, OD / 2, y_cs.size))
    ax3.invert_xaxis()
    ax3.yaxis.tick_right()
    ax3.xaxis.grid(True)
    ax3.set_ylabel('m',loc='bottom',rotation='horizontal')
    ax3.yaxis.set_label_coords(1.5,-0.06)
    return [ax1,ax2,ax3]

def add_PSF_for_subplot(fig,subplot_spec,wf_foc,x_foc,y_foc):
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)
    ax = fig.add_subplot(gs00[0, 0])
    ax.pcolormesh(x_foc, y_foc, np.log10(wf_foc), cmap='inferno',vmin=np.max(np.log10(wf_foc))-3)

    ax.set_aspect('equal')
    ax.yaxis.tick_left()
    ax.set_xlabel('arcsec')

    center_coord = [0, 0]
    patch = mpatches.Circle(center_coord, color='c', radius=0.7325, fill=False, linewidth=2)
    ax.add_artist(patch)
    ax.set_title('Log scale PSF')

def add_text_for_PSF_subplot(fig,subplot_spec,EE_noTEC,EE_TEC):
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=subplot_spec)
    ax = fig.add_subplot(gs00[0,0])
    ax.axis('off')
    ax.text(-0.2,0.55,'Mirror without TEC correction\ncouples ' + str(round(EE_noTEC*1e2, 1)) + '% light into fiber', )
    ax.text(-0.2,0.4,'With TEC correction:' + str(round(EE_TEC*1e2, 1)) + '%')