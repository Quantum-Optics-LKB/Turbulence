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


M = 4/2 #grandissement du faisceau sur la camera
# ROI=[600:1400,640:1440] TURBULENCE
# ROI=[400:800,440:840] PAS TURBULENCE
dx = 6.5e-3/M #mm on divise par le grandissement
# kx = np.fft.fftshift(np.fft.fftfreq(400, d=dx))
kx = np.fft.fftshift(np.fft.fftfreq(2048, d=dx)) #POUR ROI
# kx = np.fft.fftshift(np.fft.fftfreq(2048, d=dx)) #POUR PAS ROI
h,w = (2048,2048)
h1,w1 = (800,800)
delta_n = 1.1e-4
hl = 1/((2*np.pi*384e12/(3e8))*np.sqrt(2*np.abs(delta_n))) #healing lenght
hl= hl*1e3 #en mm



def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    # A logarithmic colormap
    ext = [np.min(kx/2), np.max(kx/2), np.min(kx/2), np.max(kx/2)]
    im = plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5), extent=ext)
    return im

def f(x,a,b):
    return a*x+b

def kolv(X,a):
    return X**(-a)

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def power_law(x, a, b):
    """ Calcul la loi de puissance avec les constantes a et b """
    return a*np.power(x, b)

t = ['15','20','25','30']
t = ['30']
nn=5
avg=30
ims_fft_av = np.zeros((h,w))
for k in t:
    print(f'\nTreating dataset : {k}')
    ims = np.empty((avg//nn, h, w))
    oms = np.empty((avg//nn, h1, w1))
    # oms = np.empty((avg,200,200)) ici je prend un petit ROI où y a pas turbulence
    for l in range(nn):
        sys.stdout.write(f'\rBatch {l+1}/{nn}')
        for i in range(avg//nn):   
            ims[i, :, :] = plt.imread("v{}/v{}_{:05d}.tif".format(k,k, l*(avg//nn)+i+1)).astype(float)
        ims_fft = np.fft.fftshift(np.fft.fft2(ims))
        ims_fft_av += np.mean(np.abs(ims_fft), axis=0)
    ims_fft_av /= nn
    
    
    ky_0_av = ims_fft_av[:,len(ims_fft_av)//2-5:len(ims_fft_av)//2+5]
    ky_0_av = np.sum(ky_0_av,1)
    ky_0_av = ky_0_av[1026:2048] #ROI 
    kx_0 = kx[1026:2048] #ROI

    
    start = 27
    stop = 63
    power = 2.8
        
    pars, cov = curve_fit(power_law, kx_0[start:stop], np.abs(ky_0_av[start:stop]), p0=[0, 0], bounds=(-np.inf, np.inf))


    plt.figure(2)
    plot_spectrum(ims_fft_av)
    plt.xlabel('kx (1/mm)')
    plt.ylabel('ky (1/mm)')
    plt.title(f'FT of turbulence (v={k})')

    plt.figure(1,[16,9])
    plt.scatter(kx_0, np.abs(ky_0_av),label=f'{k}')
    # plt.scatter(kx_0, kx_0**(power-1)*np.abs(ky_0_av),label=f'{k}')
    plt.plot(kx_0[start:stop], power_law(kx_0[start:stop], *pars),'r--',label=f'{pars[1]}')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True, which="both")
    plt.legend()
    plt.xlabel('$k_y$ mm-1',fontsize=15)
    plt.ylabel('Density',fontsize=15)
    # plt.title(f'Density profile (v={n}, avg={avg})')
    plt.title(f'Density profile (power law)',fontsize=15)
    # plt.ylim(1e4,10e8)
    plt.xlim(1,40)
    # plt.savefig('Profile_powerlaw.pdf')
    ims_fft_av = 0

    




