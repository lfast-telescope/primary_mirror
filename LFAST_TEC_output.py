import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
from matplotlib import cm
from scipy import interpolate
import h5py
import cv2 as cv
from matplotlib.widgets import EllipseSelector
import csv
import os
import matplotlib.patches as mpatches
from hcipy import *
from scipy.optimize import minimize, minimize_scalar



TEC_location_file = 'TEC_centroids.csv'
thermal_conductance_file = 'Thermal_conductances.csv'
Influence_function_file = 'Binaries/Influence_function matrix.obj'
Zernike_matrix_file = 'Binaries/Zernike_matrix.obj'


E_TEC = 0.476 #W/C Heat sink side thermal conductance
R = 3.2131 #TEC electrical resistance
K = 0.32131 #TEC thermal conductance
Alpha = 0.0298 #V/K Seebeck coefficient



def load_influence_function_matrix(): #load saved influence function matrix object
    
    fileobj = open(Influence_function_file,'rb')
    matrix = pickle.load(fileobj)
    fileobj.close()
    
    return matrix


def load_zernike_matrix(): #load saved Zernike matrix object
    
    fileobj = open(Zernike_matrix_file,'rb')
    matrix = pickle.load(fileobj)
    fileobj.close()
    
    return matrix


def import_TEC_centroids(chosen_file = None): #import x,y coordinates of each TEC
    
    if chosen_file == None:
        file = TEC_location_file
    else:
        file = chosen_file
    
    df = pd.read_csv(file).iloc[0:,:]
    
    return df


def import_thermal_conductance(): #import mirror/heat sink thermal conductance values 
    
    file = thermal_conductance_file
    
    df = pd.read_csv(file).iloc[0:,:]
    
    E_vals = list(df['E'])[0:24]
    E_vals.extend(list(np.ones(108)*E_TEC))
    
    return np.array(E_vals)


def import_4D_map(filename,Z): #import measured surface from 4D h5 file. input is (filename, Zernike matrix)
    
    f = h5py.File(filename,'r')
    data = np.array(list(f['measurement0']['genraw']['data']))
    
    invalid = np.nanmax(data)
    data[data == invalid] = np.nan #remove invalid values
    data = data*(632.8/1000) #convert from waves to nm
    
    def select_callback(eclick, erelease): #callback function for cropping the mirror from background
        """
        Callback for line selection.

        *eclick* and *erelease* are the press and release events.
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print(f"({x1:3.4f}, {y1:3.4f}) --> ({x2:3.4f}, {y2:3.4f})")
        
        global data_crop
        
        data_crop = data[int((1-y2)*len(data[0])):int((1-y1)*len(data[0])),int((x1)*len(data[0])):int((x2)*len(data[0]))]
        
        if eclick.dblclick:
            print('off')
            crop.set_active(False)
    
    zs = data 
    zs_copy = zs.copy()
    zs_copy[np.isnan(zs_copy)] = 0
    
    xs = np.linspace(-387500,387500,len(zs[0]))
    ys = np.linspace(-387500,387500,len(zs.transpose()[0]))
     
    plt.axes().set_aspect('equal')
    
    ax = plt.subplot(111)

    cs = ax.contourf((xs + 387500)/(2*387500),(-ys + 387500)/(2*387500),zs*1000,14,cmap=cm.rainbow) #plot raw measurement on xy grid from 0 to 1

    cbar = plt.colorbar(cs)
    cbar.ax.tick_params(labelsize = 12)
    cbar.set_label('Z (nm)',fontsize = 12)
    
    crop = EllipseSelector(
        ax, select_callback,
        useblit=True,
        button=[1, 3],  # disable middle button
        interactive=True)
    
    plt.xlabel('X',fontsize = 12)
    plt.ylabel('Y',fontsize = 12)
    plt.title('Click and drag to crop disc.  Double click when done.',fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    
    while crop.active == True:  #pause script until cropping is finished
        plt.waitforbuttonpress(1)
    
    zs_cropped = np.flip(data_crop, axis = 0)/2 #cropped image, perform parity flip and divide by 2 for WFE to surface conversion
    zs_cropped_copy = zs_cropped.copy()
    zs_cropped_copy[np.isnan(zs_cropped_copy)] = 0
    
    xs = np.linspace(-387500,387500,len(zs_cropped[0]))
    ys = np.linspace(-387500,387500,len(zs_cropped.transpose()[0]))
    
    Z_int = interpolate.RectBivariateSpline(ys,xs,zs_cropped_copy)
    
    xi = yi = np.linspace(-381000,381000,500)
    X,Y = np.meshgrid(xi,yi)
    
    zi = Z_int(Y,X,grid = False)  #truncate to clear aperture radius
    
    test = np.sqrt(X**2 + Y**2)                         #
    inds = np.where((test > 381000) | (test < 63500))   #
    coords = list(zip(inds[0],inds[1]))                 #remove data points outside of clear aperture
    for j,m in enumerate(coords):                       #
        zi[coords[j][0]][coords[j][1]] = np.nan         #
        
    M = zi.flatten(),zi
    
    C = Zernike_decomposition(Z, M, -1) #Zernike fit
    
    Piston = (Z[1].transpose(2,0,1)[0])*C[2][0] #
    TiltY = (Z[1].transpose(2,0,1)[1])*C[2][1]  #remove piston, tip/tilt, and power
    TiltX = (Z[1].transpose(2,0,1)[2])*C[2][2]  #
    Power = (Z[1].transpose(2,0,1)[4])*C[2][4]  #
    
    Surf = M[1] - Piston - TiltX - TiltY - Power ##remove piston, tip/tilt, and power
       
    return Surf.flatten(),Surf #return 1D flattened surface and 2D surface

    
def import_4D_map_auto(filename,Z,normal_tip_tilt_power=True,remove_coef = []):

    #Mirror radius in um
    pixel_ID = 1.5*25.4*1e3  #original value: 63500 . Changed from 1000 on 8/15/2024
    pixel_OD = 15*25.4*1e3 #original value: 381000. Changed from 381000 on 8/15/2024
    
    f = h5py.File(filename,'r')
    data = np.array(list(f['measurement0']['genraw']['data']))

    invalid = np.nanmax(data)
    
    masked_at_interferometer = len(data[data==invalid]) > 50        
    
    if masked_at_interferometer:    
        data[data == invalid] = np.nan #remove invalid values
    
    data = data*(632.8/1000) #convert from waves to um
    
    scale = 255*(data - np.nanmin(data))/np.nanmax((data - np.nanmin(data)))    #
    scale[np.isnan(scale)] = 255                                                #Convert data array to color scale image of vals 1-255
    scale[scale==0] = 1                                                         #
    
    img = scale.astype('uint8') #convert data type to uint8 for Hough Gradient function
    img = cv.medianBlur(img,5)  #median blurring function to help with detection

    if masked_at_interferometer:        
        circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,10000,                            #Find mirror disc in data array
                                param1=20,param2=15,minRadius=100,maxRadius=800)        #Output in (x_center,y_center,radius)
    else:
        grad = np.gradient(img)[1].astype('uint8')
        img_grad = cv.medianBlur(grad,7)
        circles = cv.HoughCircles(img_grad,cv.HOUGH_GRADIENT,1,10000,param1=20,param2=15,minRadius=100,maxRadius=800)
    
    circles = np.uint16(np.around(circles))                                             #Round vals to nearest integer
    
    r = circles[0][0][2] - 5    #Trim edge to remove noisy data
    
    x1 = (circles[0][0][0] - r)/len(img)    #crop h5 data array using circle params from Hough Gradient function
    x2 = (circles[0][0][0] + r)/len(img)
    y1 = 1 - (circles[0][0][1] + r)/len(img)  #Flipped due to img/contour being mirrored across y-axis
    y2 = 1 - (circles[0][0][1] - r)/len(img)  #
    
    data_crop = data[int((1-y2)*len(data[0])):int((1-y1)*len(data[0])),int((x1)*len(data[0])):int((x2)*len(data[0]))]
    
    zs_cropped = np.flip(data_crop, axis = 0)#/2 #cropped image, perform parity flip
    zs_cropped_copy = zs_cropped.copy()
    zs_cropped_copy[np.isnan(zs_cropped_copy)] = 0

    #Don't you love random ass numbers 200 lines into a piece of code that you didn't write?
    #To my best guess, this is mirror half radius in um
    #The really annoying thing about it though is that this shouldn't be hard-coded - it changes with subtense
    #It'd be better for the user to just establish the mirror size
    #Comments from warrenbfoster, 10/18/2024. Profanity omitted.
    xs = np.linspace(-387500,387500,len(zs_cropped[0]))
    ys = np.linspace(-387500,387500,len(zs_cropped.transpose()[0]))
   
    Z_int = interpolate.RectBivariateSpline(ys,xs,zs_cropped_copy)
    
    xi = yi = np.linspace(-381000,381000,500)
    X,Y = np.meshgrid(xi,yi)
    
    zi = Z_int(Y,X,grid = False)  #truncate to clear aperture radius
    
    test = np.sqrt(X**2 + Y**2)                         #
    inds = np.where((test > pixel_OD) | (test < pixel_ID))   #
    coords = list(zip(inds[0],inds[1]))                 #remove data points outside of clear aperture
    for j,m in enumerate(coords):                       #
        zi[coords[j][0]][coords[j][1]] = np.nan         #
        if False: #plot intermediate steps to figure out wtf this step is doing
            if j % 1000 == 0:
                plt.imshow(zi)
                plt.show()
        
    M = zi.flatten(),zi
    
    C = Zernike_decomposition(Z, M, -1) #Zernike fit

    f.close()
    
    if normal_tip_tilt_power:
        Piston = (Z[1].transpose(2,0,1)[0])*C[2][0] #
        TiltY = (Z[1].transpose(2,0,1)[1])*C[2][1]  #remove piston, tip/tilt, and power
        TiltX = (Z[1].transpose(2,0,1)[2])*C[2][2]  #
        Power = (Z[1].transpose(2,0,1)[4])*C[2][4]  #
        
        Surf = M[1] - Piston - TiltX - TiltY - Power ##remove piston, tip/tilt, and power
    elif len(remove_coef)>0:
        Surf = remove_modes(M,C,Z,remove_coef)
    else:
        print('Strange things are afoot')
    return Surf.flatten(),Surf #return 1D flattened surface and 2D surface    

def import_cropped_4D_map(filename, Z, normal_tip_tilt_power=True, remove_coef=[]):
    #Revision of Nick's import_4D_map_auto, but using the assumption that the user has
    #set an analysis mask that crops out the mirror outside the clear aperture.
    #This only works for coated mirrors where this distinction is obvious.

    clear_aperture_radius = 15.2 * 25.4 * 1e3  # Mirror radius that is not covered by TEC
    pixel_OD = 14.95 * 25.4 * 1e3 # Coated mirror radius
    pixel_ID = 1.8 * 25.4 * 1e3 #Coated ID

    f = h5py.File(filename, 'r')
    data = np.array(list(f['measurement0']['genraw']['data']))

    invalid = np.nanmax(data)

    data[data == invalid] = np.nan  # remove invalid values
    data = data * (632.8 / 1000)  # convert from waves to um

    asymmetry = np.max(data.shape)-np.min(data.shape)
    index = np.argmax(data.shape)
    if index == 0:
        data = data[int(np.floor(asymmetry/2)):-int(np.floor(asymmetry/2)),:]
    else:
        data = data[:,int(np.floor(asymmetry / 2)):-int(np.ceil(asymmetry / 2))]

    scale = 255 * (data - np.nanmin(data)) / np.nanmax((data - np.nanmin(data)))  #
    scale[np.isnan(scale)] = 255  # Convert data array to color scale image of vals 1-255
    scale[scale == 0] = 1  #

    img = scale.astype('uint8')  # convert data type to uint8 for Hough Gradient function
    img = cv.medianBlur(img, 5)  # median blurring function to help with detection

    OD_circle = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10000,  # Find mirror disc in data array
                              param1=20, param2=15, minRadius=100,
                              maxRadius=800)[0][0]  # Output in (x_center,y_center,radius)

    ID_circle = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 10000,  # Find mirror disc in data array
                              param1=20, param2=15, minRadius=10,
                              maxRadius=18)[0][0]  # Output in (x_center,y_center,radius)

    difference = [OD_circle[i]-ID_circle[i] for i in [0,1]]
    if np.sqrt(np.sum(np.power(difference,2))) > 3:
        print('Warning: ID detection has coordinates ' + str(ID_circle) + ' and OD detection has coordinates ' + str(OD_circle))
        if True:
            fig,ax = plt.subplots()
            ax.imshow(img)
            artist1 = plt.Circle((OD_circle[0], OD_circle[1]), OD_circle[2], fill=False)
            artist2 = plt.Circle((ID_circle[0], ID_circle[1]), ID_circle[2], fill=False)
            ax.imshow(data)
            ax.add_artist(artist1)
            ax.add_artist(artist2)
            ax.set_aspect('equal')
            plt.show()

    x = np.mean([OD_circle[0],ID_circle[0]])
    y = np.mean([OD_circle[1],ID_circle[1]])
    r = OD_circle[2] - 3  # Trim smaller edge to remove noisy data, because user has already done this

    x1 = np.round(x-r)
    x2 = np.round(x+r)
    y1 = np.round(y-r)
    y2 = np.round(y+r)

    if False:   #Plot to verify cropping
        fig, ax = plt.subplots()
        img_plot = img.copy()
        img_plot[int(y1):int(y2), int(x1):int(x2)] = 0
        ax.imshow(img_plot)
        artist1 = plt.Circle((OD_circle[0], OD_circle[1]), r, fill=False)
        artist2 = plt.Circle((ID_circle[0], ID_circle[1]), ID_circle[2], fill=False)
        ax.add_artist(artist1)
        ax.add_artist(artist2)
        ax.set_aspect('equal')
        plt.show()

    data_crop = data[int(y1):int(y2)+1, int(x1):int(x2)+1]

    zs_cropped = np.flip(data_crop, axis=0)  # /2 #cropped image, perform parity flip
    zs_cropped_copy = zs_cropped.copy()
    zs_cropped_copy[np.isnan(zs_cropped_copy)] = 0

    #Define grid that the measurement is using
    xs = np.linspace(-clear_aperture_radius, clear_aperture_radius, len(zs_cropped[0]))
    ys = np.linspace(-clear_aperture_radius, clear_aperture_radius, len(zs_cropped.transpose()[0]))

    Z_int = interpolate.RectBivariateSpline(ys, xs, zs_cropped_copy)

    #Interpolate measurement onto a 500x500 grid
    xi = yi = np.linspace(-clear_aperture_radius, clear_aperture_radius, 500)
    X, Y = np.meshgrid(xi, yi)

    zi = Z_int(Y, X, grid=False)  # truncate to clear aperture radius

    test = np.sqrt(X ** 2 + Y ** 2)  #
    inds = np.where((test > pixel_OD) | (test < pixel_ID))  #
    coords = list(zip(inds[0], inds[1]))  # remove data points outside of clear aperture
    for j, m in enumerate(coords):  #
        zi[coords[j][0]][coords[j][1]] = np.nan  #

    M = zi.flatten(), zi

    C = Zernike_decomposition(Z, M, -1)  # Zernike fit

    f.close()

    if normal_tip_tilt_power:
        Piston = (Z[1].transpose(2, 0, 1)[0]) * C[2][0]  #
        TiltY = (Z[1].transpose(2, 0, 1)[1]) * C[2][1]  # remove piston, tip/tilt, and power
        TiltX = (Z[1].transpose(2, 0, 1)[2]) * C[2][2]  #
        Power = (Z[1].transpose(2, 0, 1)[4]) * C[2][4]  #

        Surf = M[1] - Piston - TiltX - TiltY - Power  ##remove piston, tip/tilt, and power
    elif len(remove_coef) > 0:
        Surf = remove_modes(M, C, Z, remove_coef)
    else:
        print('Strange things are afoot')
    return Surf.flatten(), Surf  # return 1D flattened surface and 2D surface


def Zernike_decomposition(Z,M,n):
    
    Z_processed = Z[0].copy()[:,0:n]
    Z_processed[np.isnan(Z_processed)] = 0  #replaces NaN's with 0's
    
    if type(M) == tuple:
        M_processed = M[0].copy()
    else:
        M_processed = M.copy().ravel()
    M_processed[np.isnan(M_processed)] = 0  #replaces NaN's with 0's
    
    Z_t = Z_processed.transpose() #
    
    A = np.dot(Z_t,Z_processed) #
    
    A_inv = np.linalg.inv(A) #
    
    B = np.dot(A_inv,Z_t)       #
    
    Zernike_coefficients = np.dot(B,M_processed) #Solves matrix equation:  Zerninke coefficients = ((Z_t*Z)^-1)*Z_t*M
    
    Surf = np.dot(Z[1][:,:,0:n],Zernike_coefficients) #Calculates best fit surface to Zernike modes
    
    return Surf.flatten(),Surf,Zernike_coefficients #returns the vector containing Zernike coefficients and the generated surface in tuple form       


def Heat_loads(W,I,T_a):  #calculate heat loads, input (influence function matrix, Zernike fit to measured surface, ambient temperature in celsius)
    
    W_processed = W[0].copy()
    W_processed[np.isnan(W_processed)] = 0  #replaces NaN's with 0's
    
    I_processed = -I[0].copy()
    I_processed[np.isnan(I_processed)] = 0  #replaces NaN's with 0's
    
    W_t = W_processed.transpose() #
    
    A = np.dot(W_t,W_processed) #
    
    A_inv = np.linalg.inv(A) #
    
    B = np.dot(A_inv,W_t)       #
    
    Heat_loads = np.dot(B,I_processed) #Solves matrix equation:  Heat_loads = ((W_t*W)^-1)*W_t*I
    
    Surf = np.dot(W[1],Heat_loads) #Calculates best fit surface from heat loads
    Temp = np.dot(W[2],Heat_loads) + T_a
    TEC_temp = np.dot(W[3],Heat_loads) + T_a
    
    return Heat_loads,Surf,Temp,TEC_temp #returns the vector containing 132 heat loads and the generated surface in tuple form


def Heat_loads2(W,I,T_a):  #calculate heat loads such that sum of heat loads = 0, input (influence function matrix, Zernike fit to measured surface,ambient temperature in celsius)
    
    W_processed = W[0].copy()
    W_processed[np.isnan(W_processed)] = 0  #replaces NaN's with 0's
    
    I_processed = -I[0].copy()
    I_processed[np.isnan(I_processed)] = 0  #replaces NaN's with 0's
    
    W_t = W_processed.transpose()
    
    A = np.dot(W_t,W_processed)
    B = np.dot(W_t,I_processed)
    
    C = np.ones((1,132),dtype = 'float')
    C[0][0:24] = 0  #constrain only back TECs to sum to 0
    d = np.zeros((1,),dtype = 'float')
    f = np.zeros((1,),dtype = 'float')
    
    E = np.vstack((A,C))
    F = np.vstack((C.transpose(),f))
    G = np.hstack((B,d))
    
    H = np.hstack((E,F))
    
    H_inv = np.linalg.inv(H)

    Heat_loads = np.dot(H_inv,G)[0:132]
    
    Surf = np.dot(W[1],Heat_loads) #Calculates best fit surface from heat loads
    Temp = np.dot(W[2],Heat_loads) + T_a
    TEC_temp = np.dot(W[3],Heat_loads) + T_a
    
    return Heat_loads,Surf,Temp,TEC_temp #returns the vector containing 132 heat loads and the generated surface in tuple form         


def get_TEC_temperatures(df,temp_map): #extract TEC mirror side temperatures, df is TEC locations, temp_map is the fourth element in the heat_load output H[3]
    
    xs = ys = np.linspace(-0.45914,0.45914,500) #grid that extends out to OD TEC locations
    zs = temp_map.copy()
    zs[np.isnan(zs)] = 0
    
    Z = interpolate.interp2d(xs,ys,zs)
    
    zi = Z(xs,ys)
    
    X,Y = np.meshgrid(xs,-ys)
                
    test = np.sqrt(X**2 + Y**2)                         #
    inds = np.where((test > 0.45914) | (test < .0635))  #
    coords = list(zip(inds[0],inds[1]))                 #remove points outside OD and inside ID
    for j,m in enumerate(coords):                       #
        zi[coords[j][0]][coords[j][1]] = np.nan         #
                
                 
    xs_TEC = np.array(df['X (m)'])
    ys_TEC = np.array(df['Y (m)']) 
    
    TEC_ts = []
    
    for i,n in enumerate(xs_TEC):  #for each TEC location, extract temperature values within a 20 mm radius and average them
        TEC_point_list = []
        test = np.sqrt((X - xs_TEC[i])**2 + (Y - ys_TEC[i])**2)
        inds = np.where(test <= 0.02)
        coords = list(zip(inds[0],inds[1]))
        for j,m in enumerate(coords):
            TEC_point_list.append(zi[coords[j][0]][coords[j][1]])
        TEC_ts.append(np.nanmean(TEC_point_list))           
        
    return np.array(TEC_ts)


def TEC_electrical_load_solve_function_mirror(vals,Qh,T,E,T_a): #TEC implicit equations
    
    Th = T + 273.15
    
    T_o = T_a + 273.15

    Qc,I = vals
    
    a = ((Alpha*I)/(E) + (K)/(E) + 1)*Qc - Alpha*T_o*I + (1/2)*I**2*R + K*Th - K*T_o
    b = ((K)/(E))*Qc + Qh - Alpha*Th*I - (1/2)*I**2*R + K*Th - K*T_o
    
    return a,b



def TEC_electrical_load_mirror(Qh,T,E,T_a): #numerical solve function for current and electrical power
    
    Qc,I = optimize.fsolve(TEC_electrical_load_solve_function_mirror,(1,0.1),(Qh,T,E,T_a))
    
    P = Qh - Qc
    
    return I,P


def get_electrical_output(df,Qh,T,E,T_a): #calculate TEC currents.  Inputs are (TEC locations, heat loads, TEC temperature, thermal conductance, ambient temperature in celsius)
    
    currents = [TEC_electrical_load_mirror(Qh[i],T[i],E[i],T_a)[0] for i,n in enumerate(Qh)]
    power = [TEC_electrical_load_mirror(Qh[i],T[i],E[i],T_a)[1] for i,n in enumerate(Qh)]
    
    return currents,power
    

def Full_surface_correction(filename,T_a,n): #comprehensive function to solve for TEC currents.  Input (4D map file, ambient temperature in celsius, number of Zernike terms to correct)
    
    W = load_influence_function_matrix()

    Z = load_zernike_matrix()

    TEC_locs = import_TEC_centroids()
    
    thermal_c = import_thermal_conductance()

    M = import_4D_map_auto(filename, Z)

    Z_C = Zernike_decomposition(Z,M,n)

    H = Heat_loads2(W,Z_C,T_a)

    TEC_temps = get_TEC_temperatures(TEC_locs,H[3])

    current = get_electrical_output(TEC_locs,H[0],TEC_temps,thermal_c,T_a)[0]
    
    power = get_electrical_output(TEC_locs,H[0],TEC_temps,thermal_c,T_a)[1]
    
    return current,power,H[0]


def iterative_correction(filename,T_a,n,H_prev): #iterative correction function.  Input (4D map file, ambient temperature in celsius, number of Zernike modes to correct, previous set of heat loads)
    
    W = load_influence_function_matrix()

    Z = load_zernike_matrix()

    TEC_locs = import_TEC_centroids()
    
    thermal_c = import_thermal_conductance()

    M = import_4D_map_auto(filename, Z)

    Z_C = Zernike_decomposition(Z,M,n)

    H = Heat_loads2(W,Z_C,T_a)
    
    H_net = H_prev + H[0]
    
    TEC_temp_map = np.dot(W[3],H_net) + T_a
    
    TEC_temps = get_TEC_temperatures(TEC_locs,TEC_temp_map)

    current = get_electrical_output(TEC_locs,H_net,TEC_temps,thermal_c,T_a)[0]
   
    power = get_electrical_output(TEC_locs,H_net,TEC_temps,thermal_c,T_a)[1]
    
    return current,power,H_net

def remove_modes(M,C,Z,remove_coef):
    #Remove Zernike modes from input surface map
    removal = M[1]*0
    for coef in remove_coef:
        term = (Z[1].transpose(2,0,1)[coef])*C[2][coef]
        removal += term
        
        if False: #Plot the removal terms for sanity
            plt.imshow(term)
            plt.title(coef+1)
            plt.show()
    Surf = M[1] - removal
    return Surf