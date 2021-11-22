#!/usr/bin/python3
# -*-coding:utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import csv
from PIL import Image
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks


def read_ch(csv_name):
    
    piezo = []
    scan  = []
    
    with open(csv_name+'.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            try :
                row = row[0].split(';')
                piezo.append(float(row[0]))
                scan.append(float(row[1]))
            except:
                pass
    
    scan = savgol_filter(scan, 11, 3)
    
    return piezo, scan
    

def peak(piezo, abso, fluid=False):
    
    if fluid :
        
        maxi   = find_peaks(abso, distance=10, width=5, prominence=0.05)[0]
        
        arg_1  = np.argmin(abso[maxi])
        mini_1 = maxi[arg_1]
        maxi   = np.concatenate((maxi[:arg_1], maxi[arg_1+1:]))
        arg_2  = np.argmin(abso[maxi])
        mini_2 = maxi[arg_2]
        
        delta_v = piezo[mini_2] - piezo[mini_1]
        orig    = piezo[mini_1]
        
        return orig, delta_v


    maxi = find_peaks(abso[:4*len(abso)//5], distance=10, width=3)[0]
    
    arg_1  = np.argmin(abso[maxi])
    mini_1 = maxi[arg_1]
    
    maxi = np.concatenate((maxi[:arg_1], maxi[arg_1+1:]))
    
    arg_2  = np.argmin(abso[maxi])
    mini_2 = maxi[arg_2]

    #plt.plot(piezo, abso)
    #plt.plot(piezo[mini_1], abso[mini_1], 'o')
    #plt.plot(piezo[mini_2], abso[mini_2], 'o')
    #plt.show()
    
    delta_v = piezo[mini_2] - piezo[mini_1]
    orig    = piezo[mini_1]
    
    return orig, delta_v


def piezo_to_hz(orig, delta_v, piezo, fluid=False):
    
    if fluid:
        hz = -1*(np.array(piezo)-orig)*0.15694/delta_v   
        return hz         
    
    hz = -1*(np.array(piezo)-orig)*3.03576/delta_v    
    return hz 


def open_tif(path, name):
    im = plt.imread(path+'/'+'{}.tif'.format(name)).astype(float)
    return im
        
    
    
def gauss_fit(x, maxi, std, rtd):
    return maxi*np.exp(-(x-rtd)**2/std**2)
    
    
def tri(a, b):
    
    tot = []
    for i in range (0,len(a)):
        tot.append([a[i], b[i]])
        
    tot.sort()
    a_tri = []
    b_tri = []
    
    for i in range (0,len(a)):
        a_tri.append(tot[i][0])
        b_tri.append(tot[i][1])
    
    return a_tri, b_tri
    
def rad_avg(im, centre):
    
    moy = []
    
    centre_x, centre_y = centre[0], centre[1]
    
    Rx = np.abs(np.linspace(-centre_x, 2048-centre_x, 2048))
    Ry = np.abs(np.linspace(-centre_y, 2048-centre_y, 2048))
    
    x, y = np.meshgrid(Rx, Ry)
    R = np.sqrt(x**2 + y**2)

    rad_max = np.max(R)
    rad = np.linspace(0, rad_max, round(rad_max)//3)
    
    for i in range(1, len(rad)):
    
        R_i = np.ones((2048, 2048))
        R_i[R<=rad[i-1]] = 0
        R_i[R>rad[i]] = 0
        
        moy.append(np.sum(R_i*im)/np.sum(R_i))
    
    lim = 3*len(rad)//5
    
    return rad[:lim], moy[:lim]
      
  
   
def centre(im):   

    #plt.imshow(im)
    #plt.show()

    out_x = np.sum(im, axis=0)
    out_x = out_x/np.max(out_x)
    out_y = np.sum(im, axis=1)
    out_y = out_y/np.max(out_y)
 
    absc = np.linspace(0, 2048, 2048)
    ptot, pcov = curve_fit(gauss_fit, absc, out_x, p0=[1, 100, 1000])
    centre_x = int(ptot[2])
    
    #plt.figure(6)
    #plt.plot(absc, out_x)
    #plt.plot(absc, gauss_fit(absc, *ptot))
    
    ptot, pcov = curve_fit(gauss_fit, absc, out_y, p0=[1, 100, 1000])
    centre_y = int(ptot[2])
    
    #plt.figure(7)
    #plt.plot(absc, out_y)
    #plt.plot(absc, gauss_fit(absc, *ptot))
    #plt.show()
    
    return centre_x, centre_y  
    
    
