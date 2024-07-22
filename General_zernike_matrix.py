import numpy as np
import pickle
import math


def General_zernike_matrix(maxTerm,R,a_i,grid_shape=500): #highest order term, disc radius of measurement, inner radius in microns of measurement
    
    maxTerm += 1 # (!) Added this because piston is Z=0
    
    xs = ys = np.linspace(-R,R,grid_shape) #disc grid in microns
    X,Y = np.meshgrid(xs,-ys)
    
    Rs = np.sqrt(X**2 + Y**2) #convert to polar coords for Zernike polynomial calculation
    Ts = np.arctan2(Y,X)      #
    
    js = np.arange(0,maxTerm+1) #list of Zernike modes
    
    ns = [math.ceil((-3 + np.sqrt(9+ 8*i))/2) for i in js]     #incides
    ms = [2*js[i] - ns[i]*(ns[i]+2) for i,n in enumerate(js)]  #
    ks = (ns - np.abs(ms))/2                                   #
    
    z_vals = []
    z_mesh = []
   
    for i,n in enumerate(js):  #calculate all Zernike polynomials and add to list
        sum_set = np.arange(0,ks[i]+1)
        R_nm_terms = [ (((-1)**s*math.factorial(ns[i] - s))/(math.factorial(s)*math.factorial(int(0.5*(ns[i] + np.abs(ms[i])) - s))*math.factorial(int(0.5*(ns[i] - np.abs(ms[i])) - s))))*(Rs/R)**(ns[i]-2*s)   for s in sum_set.astype(int)]
        R_nm = sum(R_nm_terms)
        
        #N_nm = 0
        if ms[i] == 0:
            N_nm = np.sqrt((2*(ns[i] + 1)/(1 + 1)))
        else:
            N_nm = np.sqrt((2*(ns[i] + 1)/(1 + 0)))
        
        if ms[i] >= 0:
            Z_nm = N_nm*R_nm*np.cos(ms[i]*Ts)
        else:
            Z_nm = -N_nm*R_nm*np.sin(ms[i]*Ts)
            
        test = np.sqrt(X**2 + Y**2)                     #
        inds = np.where((test > R) | (test < a_i))      #
        coords = list(zip(inds[0],inds[1]))             #remove points outside of OD and inside ID
        for j,m in enumerate(coords):                   #
            Z_nm[coords[j][0]][coords[j][1]] = np.nan   #
        
        z_vals.append(Z_nm)  #add to 3D matrix
        z_mesh.append(Z_nm.flatten()) #add to flattened 2D matrix
        
    Z_3D = np.array(z_vals) #convert list into array 
    Z_matrix = np.array(z_mesh) #convert list into array

    print('Zernike matrix created for ' + str(maxTerm) + ' terms')
    
    return Z_matrix.transpose(),Z_3D.transpose(1,2,0) #return 2D array and 3D in tuple form.  Use Z[0] to call the 2D array and Z[1] to call the 3D array. Transpose functions are used to put the matrix in the correct form for matrix arithmetic.
    
    
def save_zernike_matrix(matrix): #save Zernike matrix as obj file
    fileobj = open('Binaries/Zernike_matrix.obj','wb')
    pickle.dump(matrix,fileobj)
    fileobj.close()    
