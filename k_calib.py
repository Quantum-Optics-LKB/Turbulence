# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:09:50 2021

@author: Myrann
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from scipy import fftpack
from scipy.optimize import curve_fit
import sys


imag = 50
n0=30
avg = 50
M = 400/75/2 #grandissement du faisceau sur la camera
# ROI=[600:1400,640:1440] TURBULENCE
# ROI=[400:800,440:840] PAS TURBULENCE
dx = 6.5e-3/M #mm on divise par le grandissement
# kx = np.fft.fftshift(np.fft.fftfreq(400, d=dx))
kx = np.fft.fftshift(np.fft.fftfreq(2048, d=dx))
h,w = (2048,2048)
h1,w1 = (800,800)


def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    ext = [np.min(kx/2), np.max(kx/2), np.min(kx/2), np.max(kx/2)]
    im = plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5), extent=ext)
    return im

def f(x,a,b):
    return a*x + b


kmes = []
kwei = np.array([15,17.5,20,22.5])
t = ['15','17.5','20','22.5']
# t = ['12.5']
nn=5
avg=50
ims_fft_av = np.zeros((h,w))
for k in t:
    print(f'\nTreating dataset : {k}')
    ims = np.empty((avg//nn, h, w))
    oms = np.empty((avg//nn, h1, w1))
    # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
    for l in range(nn):
        sys.stdout.write(f'\rBatch {l+1}/{nn}')
        for i in range(avg//nn):   
            ims[i, :, :] = plt.imread("k_calib2/v{}/v{}_{:05d}.tif".format(k,k, l*(avg//nn)+i)).astype(float)
        ims_fft = np.fft.fftshift(np.fft.fft2(ims))
        ims_fft_av += np.mean(np.abs(ims_fft), axis=0)
    ims_fft_av /= nn
    
    
    ky_0_av = ims_fft_av[len(ims_fft_av)//2-5:len(ims_fft_av)//2+5,:]
    ky_0_av = np.sum(ky_0_av,0)
    ky_0_av = ky_0_av[1050:2048] #ROI 
    kx_0 = kx[1050:2048] #ROI
    
    kymax = np.max(ky_0_av) #pour faire le fit lineaire calibration
    kymax_pos = np.where(ky_0_av == kymax)[0][0] #prend position dans le tableau du max
    kmes = kmes + [kx_0[kymax_pos]]
    
    
    # mean = kx_0[kymax_pos]
    # sigma = 2
    # popt1,pcov1 = curve_fit(gaus, kx_0, ky_0_av,p0=[kymax,mean,sigma]) #pour fit les porifles
    
    plt.figure(1,[15,8])
    plt.plot(kx_0, np.abs(ky_0_av),'-',label=f'v{k}')
    # plt.plot(kx_0, gaus(kx_0,*popt1),'--',label=f'fit v{n} :-/')
    # plt.yscale('log')
    plt.legend()
    plt.xlabel('kx (1/mm)')
    plt.ylabel('Density')
    # plt.title(f'Density profile (v={n}, avg={avg})')
    plt.title(f'Density profile (avg={avg})')
    # plt.ylim(0,1e8)
    # plt.xlim(4,40)
    plt.savefig(f'Profile {imag} images.pdf')
    
popt,pcov = curve_fit(f, kwei, kmes)


plt.figure(5)
plt.plot(kwei,kmes,'ro')
plt.title('Calibration kwei vs kmes')
plt.xlabel('k wei (1/mm)')
plt.ylabel('k mesure (1/mm)')
plt.plot(kwei,f(kwei,*popt),'g--',label=f'fit: a={popt[0]}, b={popt[1]}')
plt.legend()
plt.savefig('kwei vs kmes.pdf')

##FIN PLOT DE TOUS LES PORIFLE OVERLAP

#k calibration
# n=30
# avg=10
# ims = np.empty((avg, h, w))
# oms = np.empty((avg,h1,w1))
# # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
# for i in range(avg):
#     ims[i, :, :] = plt.imread("v{}_{:05d}.tif".format(n, i+1)).astype(float)
#     oms[i, :, :] = ims[i,:,:][600:1400,640:1440]
# ims_fft = np.fft.fftshift(np.fft.fft2(oms)) 
# ims_fft_av = np.mean(np.abs(ims_fft), axis=0)


# ky_0_av = ims_fft_av[len(ims_fft_av)//2,:]
# ky_0_av = ky_0_av[400:800] #ROI AVEC TURB
# # ky_0_av = ky_0_av[200:400] #POUR ROI SANS TURB
# kx_0 = kx[400:800]
# # kx_0 = kx[200:400] #ROI SANS TURB

# plt.figure(1,[20,14])
# plt.plot(kx_0, np.abs(ky_0_av),'-',label=f'v{n}')
# plt.yscale('log')
# plt.legend()
# plt.xlabel('kx (1/mm)')
# plt.ylabel('Density')
# # plt.title(f'Density profile (v={n}, avg={avg})')
# plt.title(f'Density profile (avg={avg})')
# # plt.ylim(0,1e8)
# plt.xlim(1,10)
# plt.savefig('Profile no turbulence')